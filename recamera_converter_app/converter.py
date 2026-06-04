from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import onnx

from tools.make_recamera_transform_cmd import derive_outputs, all_tensor_names
from .model_info import read_classes, write_model_info

REPO_ROOT = Path(__file__).resolve().parents[1]
JOBS_ROOT = REPO_ROOT / "jobs"
CACHE_VOLUME = os.environ.get("RECAMERA_TPUMLIR_CACHE_VOLUME", "recamera_tpumlir_cache")
DOCKER_IMAGE = os.environ.get("TPUC_DOCKER_IMAGE", "sophgo/tpuc_dev:v3.4")

Precision = Literal["INT8", "F16", "BOTH"]
Task = Literal["detection"]


@dataclass
class ConvertResult:
    job_id: str
    job_dir: Path
    status: str
    logs: list[str] = field(default_factory=list)
    cvimodel: Path | None = None
    model_json: Path | None = None
    zip_path: Path | None = None
    output_names: list[str] = field(default_factory=list)

    def log(self, text: str) -> None:
        self.logs.append(text)
        (self.job_dir / "logs" / "conversion.log").write_text("\n".join(self.logs) + "\n", encoding="utf-8")


def safe_job_prefix(model_name: str | None) -> str:
    value = (model_name or "").strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = value.strip("._-")
    return value or "model"


def new_job_dir(model_name: str | None = None) -> tuple[str, Path]:
    JOBS_ROOT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{safe_job_prefix(model_name)}-{stamp}"
    job_id = base
    i = 1
    while (JOBS_ROOT / job_id).exists():
        i += 1
        job_id = f"{base}-{i}"
    job_dir = JOBS_ROOT / job_id
    (job_dir / "input").mkdir(parents=True)
    (job_dir / "output").mkdir()
    (job_dir / "logs").mkdir()
    return job_id, job_dir


def save_upload(src, dst: Path) -> None:
    with dst.open("wb") as fh:
        shutil.copyfileobj(src, fh)


def write_job_status(job_dir: Path, *, status: str, message: str = "") -> None:
    payload = {
        "job_id": job_dir.name,
        "status": status,
        "message": message,
        "job_dir": str(job_dir),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (job_dir / "logs" / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_job_status(job_dir: Path) -> dict:
    status_path = job_dir / "logs" / "status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            status = {"job_id": job_dir.name, "status": "unknown", "message": "Could not read status.json"}
    else:
        status = {"job_id": job_dir.name, "status": "unknown", "message": "No status.json yet"}

    log_path = job_dir / "logs" / "conversion.log"
    status["log"] = log_path.read_text(errors="replace")[-40000:] if log_path.exists() else ""

    downloads = []
    output_dir = job_dir / "output"
    if output_dir.exists():
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path.suffix in {".cvimodel", ".json", ".zip", ".log", ".txt", ".sh"}:
                downloads.append({"name": path.name, "url": f"/download/{job_dir.name}/{path.name}", "size": path.stat().st_size})
    status["downloads"] = downloads
    return status


def inspect_outputs(onnx_path: Path) -> list[str]:
    model = onnx.load(onnx_path)
    outputs = derive_outputs(all_tensor_names(model))
    if len(outputs) != 6:
        raise RuntimeError(
            "Could not derive six YOLO detection pre-head outputs. "
            "Currently v1 supports YOLO11/YOLO26 detection-style graphs only."
        )
    return outputs


def normalize_precision(precision: str | None) -> Precision:
    value = (precision or "INT8").strip().upper()
    if value == "BOTH":
        return "BOTH"
    if value in {"INT8", "F16"}:
        return value  # type: ignore[return-value]
    raise RuntimeError(f"Unsupported precision mode: {precision!r}. Use INT8, F16, or BOTH.")


def precision_modes(precision: Precision) -> list[Literal["INT8", "F16"]]:
    if precision == "BOTH":
        return ["INT8", "F16"]
    return [precision]


def find_docker_cli(explicit_path: str | None = None) -> str | None:
    """Find Docker CLI even when Docker Desktop is installed but not in PATH."""
    candidates: list[str] = []
    if explicit_path:
        candidates.append(explicit_path)
    env_path = os.environ.get("DOCKER_CLI") or os.environ.get("DOCKER_PATH")
    if env_path:
        candidates.append(env_path)

    which = shutil.which("docker") or shutil.which("docker.exe")
    if which:
        candidates.append(which)

    system = platform.system().lower()
    if system == "windows":
        program_files = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]
        local_app_data = os.environ.get("LOCALAPPDATA")
        for base in program_files:
            if base:
                candidates.append(str(Path(base) / "Docker" / "Docker" / "resources" / "bin" / "docker.exe"))
        if local_app_data:
            candidates.append(str(Path(local_app_data) / "Docker" / "resources" / "bin" / "docker.exe"))
    elif system == "darwin":
        candidates.extend([
            "/Applications/Docker.app/Contents/Resources/bin/docker",
            "/usr/local/bin/docker",
            "/opt/homebrew/bin/docker",
        ])
    else:
        candidates.extend(["/usr/bin/docker", "/usr/local/bin/docker", "/snap/bin/docker"])

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        path = Path(candidate).expanduser()
        if path.exists() and path.is_file():
            return str(path)
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def run_checked(cmd: list[str], *, cwd: Path, result: ConvertResult) -> None:
    result.log("$ " + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        clean = line.rstrip("\n")
        result.log(clean)
        if "[reCamera converter]" in clean:
            if "starting model_transform" in clean:
                write_job_status(result.job_dir, status="running", message="Running model_transform")
            elif "model_transform finished" in clean:
                write_job_status(result.job_dir, status="running", message="model_transform finished; continuing deploy flow")
            elif "starting run_calibration" in clean:
                write_job_status(result.job_dir, status="running", message="Running INT8 calibration")
            elif "run_calibration finished" in clean:
                write_job_status(result.job_dir, status="running", message="Calibration finished; continuing deploy flow")
            elif "starting INT8 model_deploy" in clean:
                write_job_status(result.job_dir, status="running", message="Running INT8 model_deploy")
            elif "starting F16 model_deploy" in clean:
                write_job_status(result.job_dir, status="running", message="Running F16 model_deploy")
            elif "model_deploy finished" in clean:
                write_job_status(result.job_dir, status="running", message="model_deploy finished; collecting outputs")
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}")


def shell_script(*, model_name: str, output_names: list[str], test_image_name: str, calibration_count: int, precision: Precision) -> str:
    output_arg = ",".join(output_names)
    calibration_count = max(1, calibration_count)
    deploy_parts: list[str] = []
    modes = precision_modes(precision)
    if "INT8" in modes:
        deploy_parts.append(f"""
echo "[reCamera converter] $(date -Is) starting run_calibration with {calibration_count} image(s)"
run_tool run_calibration \\
  /tmp/onnx_cvimodel_work/{shlex.quote(model_name)}.mlir \\
  --dataset /workspace/calibration \\
  --input_num {calibration_count} \\
  -o /workspace/output/{shlex.quote(model_name)}_calib_table
echo "[reCamera converter] $(date -Is) run_calibration finished"
echo "[reCamera converter] $(date -Is) starting INT8 model_deploy"
run_tool model_deploy \\
  --mlir /tmp/onnx_cvimodel_work/{shlex.quote(model_name)}.mlir \\
  --quant_input \\
  --quantize INT8 \\
  --customization_format RGB_PACKED \\
  --processor cv181x \\
  --calibration_table /workspace/output/{shlex.quote(model_name)}_calib_table \\
  --test_input /workspace/input/{shlex.quote(test_image_name)} \\
  --test_reference /workspace/output/{shlex.quote(model_name)}_top_outputs.npz \\
  --fuse_preprocess \\
  --aligned_input \\
  --tolerance 0.98,0.8 \\
  --model /workspace/output/{shlex.quote(model_name)}_cv181x_int8.cvimodel
echo "[reCamera converter] $(date -Is) INT8 model_deploy finished"
""".strip())
    if "F16" in modes:
        deploy_parts.append(f"""
echo "[reCamera converter] $(date -Is) starting F16 model_deploy"
run_tool model_deploy \\
  --mlir /tmp/onnx_cvimodel_work/{shlex.quote(model_name)}.mlir \\
  --quant_input \\
  --quantize F16 \\
  --customization_format RGB_PACKED \\
  --processor cv181x \\
  --test_input /workspace/input/{shlex.quote(test_image_name)} \\
  --test_reference /workspace/output/{shlex.quote(model_name)}_top_outputs.npz \\
  --fuse_preprocess \\
  --tolerance 0.99,0.9 \\
  --model /workspace/output/{shlex.quote(model_name)}_cv181x_f16.cvimodel
echo "[reCamera converter] $(date -Is) F16 model_deploy finished"
""".strip())
    deploy_script = "\n".join(deploy_parts)
    return f"""
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_PROGRESS_BAR=off
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export TERM=dumb
export NO_COLOR=1
export TQDM_ASCII=1
echo "[reCamera converter] $(date -Is) container started"
echo "[reCamera converter] precision mode: {precision}"
echo "[reCamera converter] probing TPU-MLIR environment"
bootstrap_tpumlir() {{
  for setup in \
    /workspace/tpu-mlir/envsetup.sh \
    /workspace/tpu_mlir/envsetup.sh \
    /workspace/tpu-mlir*/envsetup.sh \
    /workspace/tpu_mlir*/envsetup.sh \
    /opt/tpu-mlir/envsetup.sh \
    /opt/tpu_mlir/envsetup.sh \
    /opt/tpu-mlir*/envsetup.sh \
    /opt/tpu_mlir*/envsetup.sh \
    /root/tpu-mlir/envsetup.sh \
    /root/tpu_mlir/envsetup.sh \
    /root/tpu-mlir*/envsetup.sh \
    /root/tpu_mlir*/envsetup.sh; do
    if [ -f "$setup" ]; then
      echo "[reCamera converter] sourcing $setup"
      # shellcheck disable=SC1090
      source "$setup"
      break
    fi
  done
  if ! command -v model_transform >/dev/null 2>&1 || ! command -v model_deploy >/dev/null 2>&1; then
    found_setup="$(find /opt /root /usr/local /workspace -maxdepth 5 -name envsetup.sh 2>/dev/null | head -1 || true)"
    if [ -n "$found_setup" ]; then
      echo "[reCamera converter] sourcing discovered $found_setup"
      # shellcheck disable=SC1090
      source "$found_setup"
    fi
  fi
  configure_tpumlir_env() {{
    TPUMLIR_PYTHON="${{TPUMLIR_PYTHON:-$fallback_venv/bin/python3}}"
    export TPUMLIR_PYTHON
    if [ ! -x "$TPUMLIR_PYTHON" ]; then
      echo "[reCamera converter] fallback Python missing: $TPUMLIR_PYTHON" >&2
      return 1
    fi
    if ! tpumlir_pkg_dir="$($TPUMLIR_PYTHON - <<'PYENV'
from pathlib import Path
import tpu_mlir
print(Path(tpu_mlir.__file__).resolve().parent)
PYENV
)"; then
      echo "[reCamera converter] tpu_mlir import failed with $TPUMLIR_PYTHON" >&2
      return 1
    fi
    for candidate in       "$fallback_venv/bin"       "$tpumlir_pkg_dir/bin"       "$tpumlir_pkg_dir/../bin"       "$tpumlir_pkg_dir/../../bin"; do
      if [ -d "$candidate" ]; then
        PATH="$candidate:$PATH"
      fi
    done
    for candidate in       "$tpumlir_pkg_dir/lib"       "$tpumlir_pkg_dir/../lib"       "$tpumlir_pkg_dir/../../lib"; do
      if [ -d "$candidate" ]; then
        LD_LIBRARY_PATH="$candidate:${{LD_LIBRARY_PATH:-}}"
      fi
    done
    export PATH LD_LIBRARY_PATH
  }}
  fallback_import_check() {{
    TPUMLIR_PYTHON="${{TPUMLIR_PYTHON:-$fallback_venv/bin/python3}}"
    export TPUMLIR_PYTHON
    "$TPUMLIR_PYTHON" - <<'PYDEP'
import pkg_resources, flatbuffers, onnx, onnxruntime, numpy, cv2, yaml, requests, tqdm, scipy, skimage, pycocotools, tpu_mlir
PYDEP
    configure_tpumlir_env
    command -v tpuc-opt >/dev/null 2>&1
  }}
  install_fallback_deps() {{
    python3 -m pip install --no-input --progress-bar off --upgrade pip wheel
    python3 -m pip install --no-input --progress-bar off --upgrade --force-reinstall 'setuptools>=65,<81'
    python3 -m pip install --no-input --progress-bar off \
      'setuptools>=65,<81' \
      'tpu_mlir==1.7' \
      'flatbuffers>=23,<25' \
      'onnx>=1.16,<2' \
      'onnxruntime>=1.16,<2' \
      'onnxsim>=0.4,<1' \
      'numpy<2' \
      'opencv-python-headless>=4.8,<5' \
      'PyYAML>=6,<7' \
      'requests>=2.31,<3' \
      'tqdm>=4,<5' \
      'transformers>=4,<5' \
      'scipy>=1.10,<2' \
      'scikit-image>=0.21,<1' \
      'pycocotools>=2,<3' \
      'torch==2.0.1' \
      'torchvision==0.15.2'
  }}
  build_fallback_venv() {{
    tmp_venv="/cache/tpu_mlir_venv.tmp.$$"
    old_venv="/cache/tpu_mlir_venv.broken-$(date +%Y%m%d-%H%M%S)-$$"
    rm -rf "$tmp_venv"
    "$system_python" -m venv "$tmp_venv"
    # shellcheck disable=SC1091
    source "$tmp_venv/bin/activate"
    TPUMLIR_PYTHON="$tmp_venv/bin/python3"
    export TPUMLIR_PYTHON
    install_fallback_deps
    configure_tpumlir_env
    fallback_import_check
    if [ -e "$fallback_venv" ]; then
      mv "$fallback_venv" "$old_venv" || rm -rf "$fallback_venv"
    fi
    mv "$tmp_venv" "$fallback_venv"
    # shellcheck disable=SC1091
    source "$fallback_venv/bin/activate"
    TPUMLIR_PYTHON="$fallback_venv/bin/python3"
    export TPUMLIR_PYTHON
    configure_tpumlir_env
    echo "[reCamera converter] fallback venv install/import check OK"
  }}
  ensure_tpumlir_cli() {{
    TPUMLIR_PYTHON="${{TPUMLIR_PYTHON:-$fallback_venv/bin/python3}}"
    export TPUMLIR_PYTHON
    configure_tpumlir_env
    wrapper_dir=/cache/tpumlir_wrappers/bin
    mkdir -p "$wrapper_dir"
    PATH="$wrapper_dir:$PATH"
    export PATH
    cat > "$wrapper_dir/model_transform" <<'WRAP'
#!/usr/bin/env bash
exec "$TPUMLIR_PYTHON" -m tpu_mlir.python.tools.model_transform "$@"
WRAP
    cat > "$wrapper_dir/model_deploy" <<'WRAP'
#!/usr/bin/env bash
exec "$TPUMLIR_PYTHON" -m tpu_mlir.python.tools.model_deploy "$@"
WRAP
    cat > "$wrapper_dir/run_calibration" <<'WRAP'
#!/usr/bin/env bash
exec "$TPUMLIR_PYTHON" -m tpu_mlir.python.tools.run_calibration "$@"
WRAP
    chmod +x "$wrapper_dir/model_transform" "$wrapper_dir/model_deploy" "$wrapper_dir/run_calibration"
    hash -r 2>/dev/null || true
  }}
  if ! command -v model_transform >/dev/null 2>&1 || ! command -v model_deploy >/dev/null 2>&1; then
    fallback_venv=/cache/tpu_mlir_venv
    system_python="$(command -v python3)"
    mkdir -p /cache/pip
    export PIP_CACHE_DIR=/cache/pip
    echo "[reCamera converter] TPU-MLIR commands not on PATH; using cached fallback venv at $fallback_venv"
    if [ -x "$fallback_venv/bin/python3" ]; then
      # shellcheck disable=SC1091
      source "$fallback_venv/bin/activate"
      if fallback_import_check; then
        echo "[reCamera converter] cached fallback venv and tpuc-opt check OK"
      else
        echo "[reCamera converter] cached fallback venv is incomplete or missing tpuc-opt; rebuilding safely"
        deactivate 2>/dev/null || true
        build_fallback_venv
      fi
    else
      build_fallback_venv
    fi
    ensure_tpumlir_cli
  fi
}}
bootstrap_tpumlir
describe_cmd() {{
  if command -v "$1" >/dev/null 2>&1; then
    type -t "$1"
  else
    echo missing
  fi
}}
echo "[reCamera converter] model_transform: $(describe_cmd model_transform)"
echo "[reCamera converter] model_deploy: $(describe_cmd model_deploy)"
echo "[reCamera converter] run_calibration: $(describe_cmd run_calibration)"
python3 -m pip show tpu_mlir 2>/dev/null | sed 's/^/[tpu_mlir package] /' || true
run_tool() {{
  # stdbuf cannot execute shell functions; TPU-MLIR fallback commands may be
  # functions wrapping `python3 -m ...` when the package does not expose CLI
  # entrypoints on PATH.
  if [ "$(type -t "$1" 2>/dev/null || true)" = "function" ]; then
    "$@"
  elif command -v stdbuf >/dev/null 2>&1; then
    stdbuf -oL -eL "$@"
  else
    "$@"
  fi
}}
mkdir -p /tmp/onnx_cvimodel_work /workspace/output /workspace/logs
echo "[reCamera converter] $(date -Is) starting model_transform"
run_tool model_transform \\
  --model_name {shlex.quote(model_name)} \\
  --model_def /workspace/input/model.onnx \\
  --input_shapes '[[1,3,640,640]]' \\
  --mean 0.0,0.0,0.0 \\
  --scale 0.0039216,0.0039216,0.0039216 \\
  --keep_aspect_ratio \\
  --pixel_format rgb \\
  --output_names {shlex.quote(output_arg)} \\
  --test_input /workspace/input/{shlex.quote(test_image_name)} \\
  --test_result /workspace/output/{shlex.quote(model_name)}_top_outputs.npz \\
  --mlir /tmp/onnx_cvimodel_work/{shlex.quote(model_name)}.mlir
echo "[reCamera converter] $(date -Is) model_transform finished"
{deploy_script}
echo "[reCamera converter] output directory:"
ls -lh /workspace/output
""".strip()


def make_zip(result: ConvertResult) -> Path:
    zip_path = result.job_dir / "output" / f"{result.job_id}-recamera-model.zip"
    log_path = result.job_dir / "logs" / "conversion.log"
    output_log = result.job_dir / "output" / "conversion.log"
    if log_path.exists():
        shutil.copy2(log_path, output_log)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted((result.job_dir / "output").iterdir()):
            if path.is_file() and path.suffix in {".cvimodel", ".json", ".log", ".txt", ".sh"}:
                zf.write(path, arcname=path.name)
    result.zip_path = zip_path
    return zip_path


def finalize_outputs(*, job_id: str, job_dir: Path, model_name: str, precision: Precision, classes_text: str | None = None, status: str = "ok") -> ConvertResult:
    result = ConvertResult(job_id=job_id, job_dir=job_dir, status=status)
    modes = precision_modes(precision)
    cvimodels = [job_dir / "output" / f"{model_name}_cv181x_{mode.lower()}.cvimodel" for mode in modes]
    missing = [str(path) for path in cvimodels if not path.exists()]
    if missing:
        raise RuntimeError("Expected output missing: " + ", ".join(missing))
    result.cvimodel = cvimodels[0]

    output_names_path = job_dir / "output" / "output_names.txt"
    if output_names_path.exists():
        result.output_names = [line.strip() for line in output_names_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]

    classes = [c.strip() for c in (classes_text or "").replace("\n", ",").split(",") if c.strip()] or read_classes()
    for mode, cvimodel in zip(modes, cvimodels):
        model_json = job_dir / "output" / f"model_{mode.lower()}.json"
        write_model_info(
            model_json,
            model_name=model_name,
            task="detection",
            classes=classes,
            description=f"{model_name} Detection {mode} converted locally for Seeed reCamera CV181x",
            author="reCamera ONNX converter",
            model_file=cvimodel,
        )
        if mode == "INT8":
            shutil.copy2(model_json, job_dir / "output" / "model.json")
            result.model_json = job_dir / "output" / "model.json"
    if result.model_json is None:
        result.model_json = job_dir / "output" / "model_f16.json"
    result.log(f"Wrote model metadata with {len(classes)} classes")
    make_zip(result)
    result.log(f"Done: {result.zip_path}")
    write_job_status(job_dir, status=status, message=f"Done. Outputs are in {job_dir / 'output'}")
    return result


def convert_prepared(
    *,
    job_id: str,
    job_dir: Path,
    onnx_path: Path,
    image_path: Path,
    image_name: str,
    calibration_count: int = 1,
    model_name: str = "model",
    task: Task = "detection",
    precision: Precision = "INT8",
    classes_text: str | None = None,
    allow_pull: bool = True,
    docker_cli: str | None = None,
) -> ConvertResult:
    precision = normalize_precision(precision)
    if task != "detection":
        raise RuntimeError("v1 supports detection only")
    result = ConvertResult(job_id=job_id, job_dir=job_dir, status="running")
    write_job_status(job_dir, status="running", message="Preparing conversion")
    result.log(f"Created {job_id}")
    result.log(f"Precision mode: {precision}")
    result.log(f"Saved ONNX: {onnx_path.stat().st_size} bytes")
    result.log(f"Using test image: {image_path.name} ({image_path.stat().st_size} bytes)")

    write_job_status(job_dir, status="running", message="Inspecting ONNX outputs")
    outputs = inspect_outputs(onnx_path)
    result.output_names = outputs
    (job_dir / "output" / "output_names.txt").write_text("\n".join(outputs) + "\n", encoding="utf-8")
    result.log("Derived output_names:\n" + "\n".join(outputs))

    script = shell_script(model_name=model_name, output_names=outputs, test_image_name=image_name, calibration_count=calibration_count, precision=precision)
    (job_dir / "output" / "commands.sh").write_text(script + "\n", encoding="utf-8")

    docker_bin = find_docker_cli(docker_cli)
    if not docker_bin:
        raise RuntimeError(
            "Docker CLI not found. Docker Desktop may be installed but docker.exe is not visible to this Python process. "
            "Set DOCKER_CLI/DOCKER_PATH or fill the Docker CLI path field, e.g. "
            "C:\\Program Files\\Docker\\Docker\\resources\\bin\\docker.exe"
        )
    result.log(f"Using Docker CLI: {docker_bin}")

    if allow_pull:
        write_job_status(job_dir, status="running", message=f"Pulling {DOCKER_IMAGE}; first run can take a while")
        run_checked([docker_bin, "pull", DOCKER_IMAGE], cwd=REPO_ROOT, result=result)

    result.log(f"Using TPU-MLIR Docker cache volume: {CACHE_VOLUME}")

    write_job_status(job_dir, status="running", message="Running TPU-MLIR conversion in Docker")
    docker_cmd = [
        docker_bin, "run", "--privileged", "--rm",
        "-v", f"{job_dir}:/workspace",
        "-v", f"{CACHE_VOLUME}:/cache",
        "-w", "/workspace",
        DOCKER_IMAGE,
        "bash", "-lc", script,
    ]
    run_checked(docker_cmd, cwd=REPO_ROOT, result=result)

    return finalize_outputs(job_id=job_id, job_dir=job_dir, model_name=model_name, precision=precision, classes_text=classes_text, status="ok")


def convert(
    *,
    onnx_file,
    test_image,
    model_name: str,
    task: Task = "detection",
    precision: Precision = "INT8",
    classes_text: str | None = None,
    allow_pull: bool = True,
    docker_cli: str | None = None,
) -> ConvertResult:
    job_id, job_dir = new_job_dir(model_name)
    onnx_path = job_dir / "input" / "model.onnx"
    image_suffix = Path(getattr(test_image, "filename", "test.jpg") or "test.jpg").suffix or ".jpg"
    image_name = f"test{image_suffix}"
    image_path = job_dir / "input" / image_name
    save_upload(onnx_file, onnx_path)
    save_upload(test_image, image_path)
    calibration_dir = job_dir / "calibration"
    calibration_dir.mkdir()
    shutil.copy2(image_path, calibration_dir / "calibration_0001.jpg")
    return convert_prepared(
        job_id=job_id,
        job_dir=job_dir,
        onnx_path=onnx_path,
        image_path=image_path,
        image_name=image_name,
        calibration_count=1,
        model_name=model_name,
        task=task,
        precision=precision,
        classes_text=classes_text,
        allow_pull=allow_pull,
        docker_cli=docker_cli,
    )
