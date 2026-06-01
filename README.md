# reCamera YOLO26 conversion notes

Working repo for investigating Seeed Studio reCamera model conversion, especially the current documentation mismatch between Ultralytics YOLO26 claims and the YOLO11-based conversion/runtime found on the device.

## Current field findings

Device inspected: `recamera1` at `10.21.37.9`.

Observed open services from BORG9 node:

- `22/tcp` SSH
- `80/tcp` reCamera web UI
- `1880/tcp` Node-RED editor
- `8090/tcp` WebSocket stream/data endpoint used by the web demo
- `9090/tcp` ttyd terminal

Runtime model state on the device:

- OS: Buildroot 2021.05, RISC-V, kernel `5.10.4-tag-`
- Inference process: `/usr/local/bin/sscma-node --start`
- Active model storage: `/userdata/Models/model.cvimodel` + `/userdata/Models/model.json`
- Factory presets: `/usr/share/supervisor/models/`
- Factory presets are YOLO11n, not YOLO26:
  - `yolo11n_detection_cv181x_int8.*`
  - `yolo11n_segment_cv181x_int8.*`
  - `yolo11n_pose_cv181x_int8.*`
  - `yolo11n_classify_cv181x_int8.*`

The device does **not** include `model_transform` or `model_deploy`; conversion is expected to happen off-device using Sophgo TPU-MLIR tooling, producing a `cvimodel` that the reCamera runtime then loads.

## Why the reported conversion failed

The Seeed conversion command hardcodes YOLO11n ONNX internal tensor names:

```text
/model.23/cv2.0/cv2.0.2/Conv_output_0
/model.23/cv3.0/cv3.0.2/Conv_output_0
/model.23/cv2.1/cv2.1.2/Conv_output_0
/model.23/cv3.1/cv3.1.2/Conv_output_0
/model.23/cv2.2/cv2.2.2/Conv_output_0
/model.23/cv3.2/cv3.2.2/Conv_output_0
```

`model_transform` fails immediately if these names are not present in the ONNX graph. That means the error is usually caused by one of these:

1. The model is not the exact YOLO11n ONNX expected by the Seeed wiki.
2. The model is YOLO26, whose head/index/names differ.
3. The Ultralytics exporter version/opset/simplification changed internal node names.
4. The model was exported with graph simplification or another tool that rewrote tensor names.

Seeed's own note says their wiki ONNX is based on IR v8 / opset v17, while newer Ultralytics exports may need adjustment.

## Suggested workflow

1. Export YOLO26 ONNX with stable settings:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(
    format="onnx",
    imgsz=640,
    opset=14,
    dynamic=False,
    simplify=False,
    nms=False,
)
```

2. Inspect the actual ONNX tensor names:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python tools/inspect_onnx.py yolo26n.onnx
```

3. Try to derive the six pre-head detection outputs:

```bash
python tools/make_recamera_transform_cmd.py yolo26n.onnx \
  --model-name yolo26n \
  --test-input ../image/dog.jpg \
  --mlir yolo26n.mlir
```

4. Run the generated `model_transform` command inside the Sophgo TPU-MLIR environment.

If the script cannot derive exactly six outputs, inspect the candidate list manually in Netron. Do **not** blindly reuse YOLO11 output names.

## References

- Seeed reCamera conversion guide: <https://wiki.seeedstudio.com/recamera_model_conversion/>
- Ultralytics reCamera integration: <https://docs.ultralytics.com/integrations/seeedstudio-recamera>
- Ultralytics ONNX export docs: <https://docs.ultralytics.com/integrations/onnx>
