# reCamera ONNX Converter

A local, browser-based converter for building Seeed Studio reCamera-compatible `.cvimodel` files from YOLO ONNX models.

The first supported workflow is:

```text
YOLO detection ONNX -> TPU-MLIR -> CV181x F16 .cvimodel + model.json
```

It is intended for reCamera users who want to try newer YOLO exports, custom-trained detection models, or models whose ONNX graph does not match Seeed's stock conversion assumptions.

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
- Generate:
  - `.cvimodel`,
  - `model.json`,
  - `output_names.txt`,
  - `commands.sh`,
  - `conversion.log`,
  - ZIP bundle.
- Includes COCO80 class list for Node-RED/reCamera metadata.

## Current limitations

V1 is intentionally narrow:

- detection models only,
- CV181x target only,
- F16 precision only,
- Docker Desktop/Engine required,
- no automatic camera deployment yet,
- segmentation/pose/classification not supported yet.

YOLO26 segmentation is **not** solved by the detection path; segmentation models have additional mask/proto outputs that need a separate runtime/postprocess contract.

## Requirements

- Python 3.10+
- Docker Desktop or Docker Engine
- Internet access for first Docker image pull
- Enough disk space for intermediate ONNX/NPZ/MLIR artifacts

The converter uses this Docker image:

```text
sophgo/tpuc_dev:v3.1
```

Inside the container it installs:

```text
tpu_mlir[all]==1.7
```

## Quick start: WebUI

```bash
git clone <this-repo-url>
cd reCamera_YOLO26

python -m venv .venv-app

# Linux/macOS:
. .venv-app/bin/activate

# Windows PowerShell:
# .venv-app\Scripts\Activate.ps1

pip install -r requirements-app.txt
python -m recamera_converter_app
```

Open:

```text
http://127.0.0.1:7860
```

Then upload:

1. ONNX model,
2. test image,
3. optional comma-separated class names.

The app returns a ZIP bundle with the converted model and metadata.

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
docker pull sophgo/tpuc_dev:v3.1

docker run --privileged --rm -it \
  -v "$PWD:/workspace" \
  -w /workspace \
  sophgo/tpuc_dev:v3.1 \
  bash
```

Inside the container:

```bash
pip install 'tpu_mlir[all]==1.7'
mkdir -p /tmp/onnx_cvimodel_work
```

Run the generated `model_transform` command. A successful first stage should end with something like:

```text
npz compare PASSED
```

Then deploy to `.cvimodel`, for example:

```bash
model_deploy \
  --mlir /tmp/onnx_cvimodel_work/yolo26n.mlir \
  --quant_input \
  --quantize F16 \
  --customization_format RGB_PACKED \
  --processor cv181x \
  --test_input /workspace/test.jpg \
  --test_reference /workspace/yolo26n_top_outputs.npz \
  --fuse_preprocess \
  --tolerance 0.99,0.9 \
  --model /workspace/yolo26n_cv181x_f16.cvimodel
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
