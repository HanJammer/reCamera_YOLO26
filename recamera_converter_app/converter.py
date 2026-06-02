from __future__ import annotations

import json
import os
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
DOCKER_IMAGE = "sophgo/tpuc_dev:v3.1"

Precision = Literal["F16"]
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


def new_job_dir() -> tuple[str, Path]:
    JOBS_ROOT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"job-{stamp}"
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


def inspect_outputs(onnx_path: Path) -> list[str]:
    model = onnx.load(onnx_path)
    outputs = derive_outputs(all_tensor_names(model))
    if len(outputs) != 6:
        raise RuntimeError(
            "Could not derive six YOLO detection pre-head outputs. "
            "Currently v1 supports YOLO11/YOLO26 detection-style graphs only."
        )
    return outputs


def docker_available() -> bool:
    return shutil.which("docker") is not None


def run_checked(cmd: list[str], *, cwd: Path, result: ConvertResult) -> None:
    result.log("$ " + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result.log(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")


def shell_script(*, model_name: str, output_names: list[str], test_image_name: str) -> str:
    output_arg = ",".join(output_names)
    # The container sees job dir as /workspace. Keep outputs inside /workspace/output so
    # the host can collect them without digging in container /tmp.
    return f"""
set -euo pipefail
python3 -m pip install -q 'tpu_mlir[all]==1.7'
mkdir -p /tmp/onnx_cvimodel_work /workspace/output /workspace/logs
model_transform \\
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
model_deploy \\
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
""".strip()


def make_zip(result: ConvertResult) -> Path:
    zip_path = result.job_dir / "output" / f"{result.job_id}-recamera-model.zip"
    log_path = result.job_dir / "logs" / "conversion.log"
    output_log = result.job_dir / "output" / "conversion.log"
    if log_path.exists():
        shutil.copy2(log_path, output_log)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [result.cvimodel, result.model_json, output_log, result.job_dir / "output" / "commands.sh", result.job_dir / "output" / "output_names.txt"]:
            if path and path.exists():
                zf.write(path, arcname=path.name)
    result.zip_path = zip_path
    return zip_path


def convert(
    *,
    onnx_file,
    test_image,
    model_name: str,
    task: Task = "detection",
    precision: Precision = "F16",
    classes_text: str | None = None,
    allow_pull: bool = True,
) -> ConvertResult:
    if task != "detection" or precision != "F16":
        raise RuntimeError("v1 supports detection/F16 only")
    job_id, job_dir = new_job_dir()
    result = ConvertResult(job_id=job_id, job_dir=job_dir, status="created")
    result.log(f"Created {job_id}")

    onnx_path = job_dir / "input" / "model.onnx"
    image_suffix = Path(getattr(test_image, "filename", "test.jpg") or "test.jpg").suffix or ".jpg"
    image_name = f"test{image_suffix}"
    image_path = job_dir / "input" / image_name
    save_upload(onnx_file, onnx_path)
    save_upload(test_image, image_path)
    result.log(f"Saved ONNX: {onnx_path.stat().st_size} bytes")
    result.log(f"Saved test image: {image_path.stat().st_size} bytes")

    outputs = inspect_outputs(onnx_path)
    result.output_names = outputs
    (job_dir / "output" / "output_names.txt").write_text("\n".join(outputs) + "\n", encoding="utf-8")
    result.log("Derived output_names:\n" + "\n".join(outputs))

    script = shell_script(model_name=model_name, output_names=outputs, test_image_name=image_name)
    (job_dir / "output" / "commands.sh").write_text(script + "\n", encoding="utf-8")

    if not docker_available():
        raise RuntimeError("Docker CLI not found. Install Docker Desktop/Engine or run generated commands.sh in TPU-MLIR manually.")

    if allow_pull:
        run_checked(["docker", "pull", DOCKER_IMAGE], cwd=REPO_ROOT, result=result)

    docker_cmd = [
        "docker", "run", "--privileged", "--rm",
        "-v", f"{job_dir}:/workspace",
        "-w", "/workspace",
        DOCKER_IMAGE,
        "bash", "-lc", script,
    ]
    run_checked(docker_cmd, cwd=REPO_ROOT, result=result)

    cvimodel = job_dir / "output" / f"{model_name}_cv181x_f16.cvimodel"
    if not cvimodel.exists():
        raise RuntimeError(f"Expected output missing: {cvimodel}")
    result.cvimodel = cvimodel

    classes = [c.strip() for c in (classes_text or "").replace("\n", ",").split(",") if c.strip()] or read_classes()
    model_json = job_dir / "output" / "model.json"
    write_model_info(
        model_json,
        model_name=model_name,
        task=task,
        classes=classes,
        description=f"{model_name} Detection F16 converted locally for Seeed reCamera CV181x",
        author="reCamera ONNX converter",
        model_file=cvimodel,
    )
    result.model_json = model_json
    result.log(f"Wrote model.json with {len(classes)} classes")
    make_zip(result)
    result.status = "ok"
    result.log(f"Done: {result.zip_path}")
    return result
