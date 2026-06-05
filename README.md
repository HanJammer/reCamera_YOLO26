# reCamera ONNX Converter

A local, browser-based converter for building Seeed Studio reCamera-compatible `.cvimodel` files from YOLO ONNX models.

The goal is to make custom YOLO-to-reCamera conversion approachable for users who do not want to hand-edit TPU-MLIR commands, tensor names, Docker paths, or Node-RED metadata.

> Status: early V1/prototype. The detection path works for the supported YOLO11/YOLO26-style graphs, but the project is still evolving.

The first supported workflow is:

```text
YOLO detection ONNX -> TPU-MLIR -> CV181x INT8/F16 .cvimodel + model metadata
```

It is intended for reCamera users who want to try newer YOLO exports, custom-trained detection models, or models whose ONNX graph does not match Seeed's stock conversion assumptions.

**Video presentation of the tool** ([source video](https://www.youtube.com/watch?v=XgPjyj2U8YQ)).

## Why this exists

Some reCamera conversion flows and examples are YOLO11-oriented and may hardcode internal ONNX tensor names such as:

```text
/model.23/cv2.*/...
/model.23/cv3.*/...
```

YOLO26 detection exports can use different internal names, for example:

```text
/model.23/one2one_cv2.*/...
/model.23/one2one_cv3.*/...
```

If the converter asks for tensors that do not exist in the ONNX graph, conversion fails before the model is ever deployed.

This project inspects the uploaded ONNX, derives the correct detection head output names when possible, and runs the TPU-MLIR conversion locally.

## Features

- Local FastAPI WebUI.
- Upload ONNX + test image in a browser.
- Auto-derive YOLO11/YOLO26-style detection output names.
- Run Sophgo TPU-MLIR inside Docker.
- Detect common Docker CLI locations and allow a manual Docker CLI path in the WebUI.
- Use `sophgo/tpuc_dev:v3.4` by default, with `TPUC_DOCKER_IMAGE` override for future Sophgo tag changes.
- Generate:
  - `.cvimodel`,
  - `model.json`,
  - `output_names.txt`,
  - `commands.sh`,
  - `conversion.log`,
  - ZIP bundle.
- Includes COCO80 class list for Node-RED/reCamera metadata.

## Project direction

This repository is meant to become a practical WebUI around the reCamera conversion workflow:

1. upload/export a YOLO ONNX model,
2. inspect the ONNX graph,
3. derive the correct output tensor names where possible,
4. run TPU-MLIR in a reproducible container,
5. return a ready-to-upload `.cvimodel` bundle with `model.json`, logs, and the generated commands.

The first target is INT8 detection because reCamera/CV181x is memory-constrained and Seeed documents INT8 as the practical/required deployment precision. F16 remains available as an experimental comparison/debug output and `Both` generates both suffixes in one job. Segmentation, pose, classification, quantization presets, and direct reCamera deployment are candidates for later versions.

## Current limitations

V1 is intentionally narrow:

- detection models only,
- CV181x target only,
- INT8 precision by default; optional F16 or Both output mode,
- Docker Desktop/Engine required,
- no automatic camera deployment yet,
- segmentation/pose/classification not supported yet.

YOLO26 segmentation is **not** solved by the detection path; segmentation models have additional mask/proto outputs that need a separate runtime/postprocess contract.

## Requirements

- Python 3.10+
- Docker Desktop or Docker Engine
- Internet access for first Docker image pull
- Enough disk space for intermediate ONNX/NPZ/MLIR artifacts

The converter uses this Docker image by default:

```text
sophgo/tpuc_dev:v3.4
```

You can override the image with `TPUC_DOCKER_IMAGE` if Sophgo changes Docker Hub tags again.

The image normally contains TPU-MLIR. The WebUI first tries to source the image's TPU-MLIR environment. If the CLI tools are still missing, it installs base `tpu_mlir==1.7` plus explicit runtime dependencies (`setuptools`/`pkg_resources`, `onnx`, `onnxruntime`, `onnxsim`, `flatbuffers`, `torch`, `torchvision`, etc.) into an isolated `/tmp/tpu_mlir_venv` fallback. It does **not** install `tpu_mlir[all]`, because that extra can request unavailable legacy dependencies such as `paddlepaddle==2.5.0`.

## Quick start: WebUI

```bash
git clone https://github.com/HanJammer/reCamera_YOLO26.git
cd reCamera_YOLO26

python -m venv .venv

# Linux/macOS:
. .venv/bin/activate

# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements-app.txt
python -m recamera_converter_app
```

If the repeated `/api/job/...` polling lines are too noisy, disable Uvicorn access logs:

```bash
# Linux/macOS
RECAMERA_ACCESS_LOG=0 python -m recamera_converter_app

# Windows PowerShell
$env:RECAMERA_ACCESS_LOG="0"; python -m recamera_converter_app
```

The TPU-MLIR fallback cache uses Docker volume `recamera_tpumlir_cache` by default. To reset it:

```bash
docker volume rm recamera_tpumlir_cache
```

Open:

```text
http://127.0.0.1:7860
```

Then upload:

1. model name, for example `yolo26n` — used for TPU-MLIR output names and the job directory prefix,
2. ONNX model,
3. precision/output mode — `INT8` default/recommended for reCamera, `F16` experimental, or `Both`,
4. optional replacement test image — if you leave it empty, the bundled author-provided `test.jpg` is used,
5. optional INT8 calibration images — representative images are recommended; if empty, the test image is reused as a minimal fallback,
6. optional comma-separated class names.

The first run can take several minutes because Docker may need to pull the `sophgo/tpuc_dev` container image and install TPU-MLIR Python dependencies inside it. The Docker image pull checkbox controls only the Docker image update. Fallback TPU-MLIR Python dependencies are cached in the Docker named volume `recamera_tpumlir_cache` and reused across conversion jobs. This avoids very slow Windows bind-mount writes during large Python installs. After you click **Convert**, the app opens a job page with live status, logs, output directory, and download links. Job directories are named like `<model-name>-YYYYMMDD-HHMMSS`. Do not click **Convert** again unless you intentionally want to start another conversion job.

The app returns a ZIP bundle with the converted model and metadata.

### Precision/output mode

- **INT8**: default and recommended for reCamera/CV181x deployment. Requires calibration images; if none are supplied, the app uses the selected/default test image as a minimal fallback.
- **F16**: experimental/debug mode. It can be useful for comparison but may be too large for reCamera memory constraints.
- **Both**: runs one `model_transform`, then emits both `*_cv181x_int8.cvimodel` and `*_cv181x_f16.cvimodel`.

When INT8 is generated, the app also writes `model.json` as an alias for `model_int8.json`, because that is the normal reCamera/Node-RED metadata name.


### Docker Desktop installed, but app says "Docker CLI not found"

Docker Desktop can be installed while the `docker` CLI is not visible in the `PATH` inherited by Python.

Fix options:

1. Restart your terminal after installing Docker Desktop.
2. Start the app from a terminal where `docker version` works.
3. Fill the **Docker CLI path** field in the WebUI.
4. Or set an environment variable before starting the app:

Windows PowerShell:

```powershell
$env:DOCKER_CLI="C:\Program Files\Docker\Docker\resources\bin\docker.exe"
python -m recamera_converter_app
```

Linux/macOS:

```bash
export DOCKER_CLI=/usr/bin/docker
python -m recamera_converter_app
```

Typical Docker Desktop CLI path on Windows:

```text
C:\Program Files\Docker\Docker\resources\bin\docker.exe
```

## Manual conversion workflow

If you do not want to use the WebUI, the underlying helper scripts can be used directly.

### 1. Install Python requirements

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

### 2. Inspect an ONNX model

```bash
python tools/inspect_onnx.py models/yolo26n.onnx
```

### 3. Generate a `model_transform` command

```bash
python tools/make_recamera_transform_cmd.py models/yolo26n.onnx \
  --model-name yolo26n \
  --test-input /workspace/test.jpg \
  --mlir /tmp/onnx_cvimodel_work/yolo26n.mlir
```

### 4. Run TPU-MLIR in Docker

```bash
docker pull sophgo/tpuc_dev:v3.4

docker run --privileged --rm -it \
  -v "$PWD:/workspace" \
  -w /workspace \
  sophgo/tpuc_dev:v3.4 \
  bash
```

Inside the container:

```bash
mkdir -p /tmp/onnx_cvimodel_work
```

The Sophgo image already includes TPU-MLIR, so do not reinstall `tpu_mlir[all]` unless you are intentionally debugging package versions.

Run the generated `model_transform` command. A successful first stage should end with something like:

```text
npz compare PASSED
```

Then deploy to `.cvimodel`, for example:

```bash
model_deploy \
  --mlir /tmp/onnx_cvimodel_work/yolo26n.mlir \
  --quant_input \
  --quantize INT8 \
  --customization_format RGB_PACKED \
  --processor cv181x \
  --test_input /workspace/test.jpg \
  --test_reference /workspace/yolo26n_top_outputs.npz \
  --fuse_preprocess \
  --tolerance 0.98,0.8 \
  --calibration_table /workspace/yolo26n_calib_table \
  --model /workspace/yolo26n_cv181x_int8.cvimodel
```

## Using the model in reCamera / Node-RED

The generated `.cvimodel` can be uploaded through the reCamera/Node-RED model node.

Important operational details:

1. The Node-RED model node `classes` field populates `model.json`.
2. Class names should be comma-separated.
3. After changing/uploading the model node configuration, press **Deploy** in Node-RED.

If the `classes` field is empty, `model.json` may contain an empty class list such as:

```json
{"model_id":"0","model_name":"your_model.cvimodel","classes":[""]}
```

That is not a converter failure; fill the model node `classes` field and deploy again.

## COCO80 comma-separated classes

For COCO detection models, paste this into the Node-RED model node `classes` field:

```text
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush
```

## Optional helper: upload a ready `.cvimodel`

There is an experimental helper for uploading a ready `.cvimodel` and explicit `model_info` through the firmware API:

```bash
python tools/upload_cvimodel.py path/to/model.cvimodel \
  --host <recamera-host-or-ip> \
  --model-name "My Detection Model" \
  --dry-run
```

Remove `--dry-run` to upload.

This is not the main recommended public workflow yet; Node-RED upload + explicit `classes` + Deploy is easier to inspect.

## Optional helper: probe WebSocket output

To inspect raw preview/inference messages:

```bash
python tools/probe_recamera_ws.py --host <recamera-host-or-ip> --frames 3
```

This can help distinguish UI overlay confusion from actual runtime output.

## Project layout

```text
recamera_converter_app/     Local WebUI application
tools/inspect_onnx.py       Print ONNX graph/output candidates
tools/make_recamera_transform_cmd.py
                           Derive detection output names and print model_transform command
tools/upload_cvimodel.py    Experimental upload helper
tools/probe_recamera_ws.py  WebSocket inspection helper
data/coco80.txt             COCO80 class names
docs/                       Investigation notes and design docs
```

## Notes for custom-trained models

For custom detection models:

- export fixed-shape ONNX, preferably `1x3x640x640`,
- avoid NMS-only exports if you need internal head tensors,
- provide a representative test image,
- provide your own comma-separated class list,
- verify runtime output after Node-RED Deploy.

If the tool cannot derive six detection head outputs, open the model in Netron and inspect the detection head. The current v1 supports YOLO11/YOLO26-style detection graphs; other architectures may need a new output-name strategy.

## References

- Seeed reCamera model conversion: <https://wiki.seeedstudio.com/recamera_model_conversion/>
- Ultralytics reCamera integration: <https://docs.ultralytics.com/integrations/seeedstudio-recamera>
- Ultralytics ONNX export docs: <https://docs.ultralytics.com/integrations/onnx>
- HuggingFace YOLO26 ONNX exports: <https://huggingface.co/zwh20081/yolo26-onnx>
