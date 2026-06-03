# One-click reCamera ONNX -> CVI model converter design

## Goal

Create a local one-click converter for Seeed reCamera users:

1. User opens a local web page.
2. User uploads an `.onnx` model.
3. Tool detects/derives correct YOLO output names when possible.
4. Tool runs Sophgo TPU-MLIR conversion.
5. User downloads a ready `.cvimodel` plus matching metadata JSON for reCamera.

The first supported target should be **YOLO26n detection** for reCamera/CV181x, because this is the currently validated path.

## Important reality check

A truly pure-Python, no-container, cross-platform converter is not realistic as a first version.

TPU-MLIR is a native compiler/toolchain, not just a Python package. The practical user-friendly approach is:

- Python web app for UX,
- Docker Desktop / Docker Engine / WSL2 backend for the actual conversion,
- Sophgo `sophgo/tpuc_dev:v3.4` image running `model_transform` and `model_deploy`.
  The app can override this with `TPUC_DOCKER_IMAGE` because Docker Hub tags have changed before.

From the user's perspective it can still be one-click: install Docker Desktop, run one Python command, upload ONNX, download CVI model.

## Proposed user experience

```bash
git clone <this-repo-url>
cd reCamera_YOLO26
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-app.txt
python -m recamera_converter_app
```

Then browser opens:

```text
http://127.0.0.1:7860
```

UI fields:

- ONNX file upload
- task type: detection / segmentation / pose / classify; initially detection only
- target: `cv181x`
- precision: F16 first, INT8 later
- test/calibration image upload
- model display name
- class list upload/edit; default COCO 80 for detection
- Convert button
- Download:
  - `.cvimodel`
  - `model.json`
  - logs
  - generated commands

## Backend pipeline

### 1. Save upload

Create a per-job working directory:

```text
jobs/<job_id>/
  input/model.onnx
  input/test.jpg
  output/
  logs/
```

### 2. Inspect ONNX

Use existing tools:

```bash
python tools/inspect_onnx.py input/model.onnx
python tools/make_recamera_transform_cmd.py input/model.onnx --model-name <name>
```

For YOLO26 detection, derive:

```text
/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0
/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0
/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0
/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0
/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0
/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0
```

For YOLO11 detection, support the older Seeed names:

```text
/model.23/cv2.0/cv2.0.2/Conv_output_0
/model.23/cv3.0/cv3.0.2/Conv_output_0
...
```

### 3. Run Docker conversion

The Python app shells out to Docker or uses Docker SDK.

Equivalent command:

```bash
docker run --rm \
  -v "$JOB_DIR:/workspace" \
  -w /workspace \
  sophgo/tpuc_dev:v3.4 \
  bash -lc '
    mkdir -p /tmp/onnx_cvimodel_work &&
    model_transform ... &&
    model_deploy ...
  '
```

Conversion outputs are copied to `jobs/<job_id>/output/`.

### 4. Generate reCamera metadata

For detection, generate `model.json` similar to factory metadata:

```json
{
  "model_id": 0,
  "version": "1.0.0",
  "model_name": "YOLO26n Detection",
  "model_format": "cvimodel",
  "ai_framwork": 6,
  "description": "YOLO26n detection model converted for reCamera CV181x.",
  "author": "Ultralytics / converted locally",
  "classes": ["person", "bicycle", "car"]
}
```

Include checksum/model_md5 if the firmware upload path benefits from it.

### 5. Optional direct upload to reCamera

Later phase, not v1:

- detect camera IP,
- call `/api/deviceMgr/uploadModel`,
- send multipart:
  - `model_file=@model.cvimodel`
  - `model_info=<json>`
- optionally update/deploy flow if required.

This needs careful rollback and should not be in first release.

## Supported modes roadmap

### Phase 1: YOLO detection

- YOLO11n detection: known Seeed path
- YOLO26n detection: validated `model_transform` and `model_deploy` path
- F16 precision first

### Phase 2: better packaging

- Docker auto-detection
- Docker image pull/check
- progress logs via Server-Sent Events or WebSocket
- zip download with model, JSON, logs, commands

### Phase 3: INT8

- calibration dataset upload
- `run_calibration`
- INT8 `model_deploy`

### Phase 4: segmentation/pose/classify

Requires confirming reCamera/SSCMA output contracts.

Current warning: YOLO26 segmentation has extra `one2one_cv4.*` and proto outputs, so a detection-style six-output conversion is not enough to claim segmentation support.

## Risks

- Seeed firmware/runtime may have postprocess assumptions tied to YOLO11.
- F16 model may run but be slower/larger than expected.
- INT8 may require mixed precision if head accuracy degrades.
- Firmware Model Conversion panel uses cloud/task APIs and may not match local TPU-MLIR behavior.
- Direct camera upload may need auth/CSRF/session headers depending on firmware state. The firmware Model Conversion tab itself is not sufficient for YOLO26 today; it accepts ONNX and starts the broken converter path. Locally converted cvimodels should use `/api/deviceMgr/uploadModel`.

## Current validated evidence

- `yolo26n.onnx` from HuggingFace converted through `model_transform` after replacing YOLO11 `cv2/cv3` output names with YOLO26 `one2one_cv2/one2one_cv3` names.
- `model_transform` produced `npz compare PASSED`.
- `model_deploy` produced a CV181x F16 `.cvimodel` around 6.9 MB.

## Recommendation

Build the community tool as a small local web app with Docker hidden behind it. Market it as:

```text
reCamera ONNX -> CVI Model Converter
```

Not as pure Python-only conversion. Python should orchestrate; TPU-MLIR/Docker should compile.
