# Simple reCamera YOLO26 conversion runbook

## What is "Sophgo / Seeed TPU-MLIR"?

reCamera does not run normal `.onnx` files directly.

It has a small edge AI chip/runtime that wants a Seeed/Sophgo-specific file:

```text
.cvimodel
```

To get that file, a normal model must be compiled:

```text
YOLO26 .onnx  ->  .mlir  ->  .cvimodel
```

The compiler/converter is called **TPU-MLIR**. It provides commands such as:

- `model_transform` - converts ONNX to MLIR
- `model_deploy` - converts MLIR to CVI model (`.cvimodel`)

This is not installed on the reCamera. It runs on a normal Linux PC/workstation, usually inside Docker.

## Recommended environment

Seeed's guide recommends Sophgo's Docker image:

```bash
docker pull sophgo/tpuc_dev:v3.1
```

Then run a container with this repo mounted:

```bash
cd reCamera_YOLO26

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

Verify tools exist:

```bash
model_transform --help | head
model_deploy --help | head
```

## Download YOLO26n ONNX

Inside or outside the container:

```bash
mkdir -p models
curl -L \
  https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n.onnx \
  -o models/yolo26n.onnx
```

## Generate the correct `model_transform` command

The Seeed wiki uses YOLO11 tensor names. For YOLO26, use this repo's helper:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python tools/make_recamera_transform_cmd.py models/yolo26n.onnx --model-name yolo26n
```

Or use the already generated command file:

```bash
cat hf_analysis/yolo26n.transform_cmd.txt
```

The important difference is that YOLO26 uses:

```text
one2one_cv2 / one2one_cv3
```

not the old YOLO11:

```text
cv2 / cv3
```

## First conversion step: ONNX -> MLIR

Inside the TPU-MLIR container, run the command from:

```bash
hf_analysis/yolo26n.transform_cmd.txt
```

If your calibration/test image path is different, replace:

```text
/app/default_dataset/000000179487.jpg
```

with an actual image available inside the container, for example:

```text
/workspace/test_images/example.jpg
```

## Second conversion step: MLIR -> cvimodel

If `model_transform` succeeds, the next approximate command is:

```bash
model_deploy \
  --mlir /tmp/onnx_cvimodel_work/yolo26n.mlir \
  --quant_input \
  --quantize F16 \
  --customization_format RGB_PACKED \
  --processor cv181x \
  --test_input /app/default_dataset/000000179487.jpg \
  --test_reference yolo26n_top_outputs.npz \
  --fuse_preprocess \
  --tolerance 0.99,0.9 \
  --model yolo26n_cv181x_f16.cvimodel
```

This mirrors Seeed's YOLO11 flow, but uses YOLO26-derived output names in the first step.

## Important caution

Do not overwrite the live reCamera model yet.

Before deployment, back up:

```text
/userdata/Models/model.cvimodel
/userdata/Models/model.json
```

The current camera runtime is proven to load YOLO11n models. YOLO26 conversion may succeed but still need runtime/postprocess compatibility testing.

## Current known status

- BORG9 confirmed SSH key login to reCamera works.
- reCamera factory models are YOLO11n.
- `yolo26n.onnx` from HuggingFace has valid YOLO26 internal tensors.
- Correct detection output names were derived and saved in `hf_analysis/yolo26n.transform_cmd.txt`.
- BORG9 node cannot run TPU-MLIR because it lacks `model_transform`, `model_deploy`, `tpu_mlir`, and Docker.
