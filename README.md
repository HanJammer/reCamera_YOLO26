# reCamera YOLO26 conversion notes

This repo investigates and works around the current Seeed Studio reCamera YOLO26 conversion problem.

Short version: the shipped reCamera image and Seeed conversion flow are still YOLO11-centric, while some Ultralytics documentation now talks about YOLO26. The firmware ONNX converter can fail on YOLO26 because it uses YOLO11 internal tensor names. This repo derives the correct YOLO26 names, converts locally with TPU-MLIR, and uploads the ready `.cvimodel`.

## Status

Validated on HanJammer's reCamera at `10.21.37.9`:

- Factory models on the camera are **YOLO11n**, not YOLO26.
- Firmware/Seeed ONNX conversion fails for YOLO26 when it looks for YOLO11 tensor names such as `/model.23/cv2.*` and `/model.23/cv3.*`.
- YOLO26n detection uses `/model.23/one2one_cv2.*` and `/model.23/one2one_cv3.*` instead.
- Local TPU-MLIR conversion succeeded for `yolo26n.onnx` from <https://huggingface.co/zwh20081/yolo26-onnx>.
- `model_deploy` produced `yolo26n_cv181x_f16.cvimodel` around 6.9 MB.
- The generated F16 `.cvimodel` was uploaded to reCamera and WebSocket `8090` returned detection boxes.

Current caveat: reCamera UI/Node-RED upload can reset `model.json` to a minimal form with empty classes:

```json
{"model_id":"0","model_name":"yolo26n_cv181x_f16.cvimodel","classes":[""]}
```

Use `tools/upload_cvimodel.py` after UI upload, or instead of it, to restore metadata and COCO80 classes.

## Device findings

Device inspected: `recamera1` at `10.21.37.9`.

Observed open services:

- `22/tcp` SSH
- `80/tcp` reCamera web UI
- `1880/tcp` Node-RED editor
- `8090/tcp` WebSocket stream/data endpoint used by the web demo
- `9090/tcp` ttyd terminal

Runtime model state:

- OS: Buildroot 2021.05, RISC-V, kernel `5.10.4-tag-`
- inference process: `/usr/local/bin/sscma-node --start`
- active model storage: `/userdata/Models/model.cvimodel` + `/userdata/Models/model.json`
- compatibility/alternate model storage also exists: `/userdata/MODEL/model.*`
- factory presets: `/usr/share/supervisor/models/`
- factory presets are YOLO11n:
  - `yolo11n_detection_cv181x_int8.*`
  - `yolo11n_segment_cv181x_int8.*`
  - `yolo11n_pose_cv181x_int8.*`
  - `yolo11n_classify_cv181x_int8.*`

The device does **not** include `model_transform` or `model_deploy`; conversion is expected to happen off-device using Sophgo TPU-MLIR tooling.

## Why the firmware conversion fails

The Seeed conversion command hardcodes YOLO11n ONNX internal tensor names:

```text
/model.23/cv2.0/cv2.0.2/Conv_output_0
/model.23/cv3.0/cv3.0.2/Conv_output_0
/model.23/cv2.1/cv2.1.2/Conv_output_0
/model.23/cv3.1/cv3.1.2/Conv_output_0
/model.23/cv2.2/cv2.2.2/Conv_output_0
/model.23/cv3.2/cv3.2.2/Conv_output_0
```

YOLO26n detection from the HuggingFace ONNX repo instead uses:

```text
/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0
/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0
/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0
/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0
/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0
/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0
```

So the immediate failure is not mysterious: the converter is asking for tensors that do not exist in the YOLO26 graph.

## Convert YOLO26n detection locally

This repo currently assumes Docker/Sophgo TPU-MLIR for conversion. A pure Python-only converter is not realistic because TPU-MLIR is a native compiler toolchain.

### 1. Start TPU-MLIR container

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
```

### 2. Download YOLO26n ONNX

```bash
mkdir -p models
curl -L \
  https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n.onnx \
  -o models/yolo26n.onnx
```

Model artifacts are intentionally ignored by git.

### 3. Generate transform command

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python tools/make_recamera_transform_cmd.py models/yolo26n.onnx \
  --model-name yolo26n \
  --test-input /workspace/test.jpg \
  --mlir /tmp/onnx_cvimodel_work/yolo26n.mlir
```

Or use the already generated command in:

```text
hf_analysis/yolo26n.transform_cmd.txt
```

### 4. Run `model_transform`

Important: create the output directory first.

```bash
mkdir -p /tmp/onnx_cvimodel_work
```

Then run the generated `model_transform` command. Expected good sign:

```text
npz compare PASSED
```

### 5. Run `model_deploy`

```bash
model_deploy \
  --mlir /tmp/onnx_cvimodel_work/yolo26n.mlir \
  --quant_input \
  --quantize F16 \
  --customization_format RGB_PACKED \
  --processor cv181x \
  --test_input /workspace/test.jpg \
  --test_reference yolo26n_top_outputs.npz \
  --fuse_preprocess \
  --tolerance 0.99,0.9 \
  --model yolo26n_cv181x_f16.cvimodel
```

Expected output:

```text
yolo26n_cv181x_f16.cvimodel
```

## Upload ready `.cvimodel` to reCamera

The firmware **Model Conversion** panel is for ONNX conversion. It is the broken path for YOLO26 today.

For locally converted `.cvimodel`, use the firmware upload endpoint:

```text
/api/deviceMgr/uploadModel
```

Helper:

```bash
python tools/upload_cvimodel.py yolo26n_cv181x_f16.cvimodel \
  --host 10.21.37.9 \
  --model-name "YOLO26n Detection F16" \
  --dry-run
```

If the dry run metadata looks sane:

```bash
python tools/upload_cvimodel.py yolo26n_cv181x_f16.cvimodel \
  --host 10.21.37.9 \
  --model-name "YOLO26n Detection F16"
```

The helper sends:

- `model_file=@...cvimodel`
- `model_info=<json>`
- COCO80 classes from `data/coco80.txt`

## Known issues and workarounds

### UI upload resets classes

Uploading through the firmware UI or Node-RED model node can produce:

```json
{"model_id":"0","model_name":"yolo26n_cv181x_f16.cvimodel","classes":[""]}
```

Workaround: re-upload with `tools/upload_cvimodel.py`, which sends explicit `model_info` with COCO80 classes.

### Bounding boxes can look unchanged

YOLO11n and YOLO26n detection on the same scene may produce similar bounding boxes. Identical-looking UI boxes do not prove the model did not change.

Better verification:


Probe WebSocket output directly:

```bash
python tools/probe_recamera_ws.py --host 10.21.37.9 --frames 3
```

- compare `/userdata/Models/model.cvimodel` size/checksum,
- inspect `/userdata/Models/model.json`,
- inspect WebSocket `8090` raw `boxes` values,
- test with controlled scenes/classes,
- compare confidence/class outputs across factory YOLO11n and YOLO26n.

### Segmentation is not solved yet

YOLO26 segmentation has additional mask/proto outputs, e.g. `output1 [1,32,160,160]` and `one2one_cv4.*` branches. A detection-style six-output conversion is not enough to claim segmentation support.

## Toward a user-friendly converter

The intended community tool is a local web app:

1. user runs a Python command,
2. browser opens,
3. user uploads ONNX and test image,
4. app runs Docker/Sophgo TPU-MLIR in the background,
5. app returns `.cvimodel`, `model.json`, logs, and generated commands.

See:

```text
docs/one-click-converter-design.md
```

Recommended v1 scope:

- YOLO11/YOLO26 detection only,
- CV181x target,
- F16 precision,
- Docker backend hidden behind local web UI,
- no direct camera upload until conversion and metadata behavior are stable.

## References

- Seeed reCamera conversion guide: <https://wiki.seeedstudio.com/recamera_model_conversion/>
- Ultralytics reCamera integration: <https://docs.ultralytics.com/integrations/seeedstudio-recamera>
- Ultralytics ONNX export docs: <https://docs.ultralytics.com/integrations/onnx>
- HuggingFace YOLO26 ONNX exports: <https://huggingface.co/zwh20081/yolo26-onnx>
