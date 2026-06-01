# reCamera device findings - 2026-06-02

## Scope

Read-only inspection of HanJammer's reCamera at `10.21.37.9`, focused on the YOLO26 vs YOLO11 confusion and the conversion failure caused by missing ONNX output names.

No services were restarted and no files were changed on the camera.

## Network/services

From the BORG9 OpenClaw node:

- ICMP reachable
- `22/tcp` SSH
- `80/tcp` web UI
- `1880/tcp` Node-RED editor
- `8090/tcp` WebSocket endpoint used by the demo UI
- `9090/tcp` ttyd terminal

The undocumented `8090/tcp` port is required by the browser demo. The web UI JavaScript connects to `ws://${window.location.hostname}:8090`.

## Device basics

- Hostname: `recamera1`
- OS: Buildroot 2021.05
- Kernel: `Linux 5.10.4-tag-` on `riscv64`
- User: `recamera`, groups include `dialout`, `audio`, `input`, `i2c`, `spi`, `gpio`
- Main AI process observed: `/usr/local/bin/sscma-node --start`
- Node-RED process observed: `node-red`

## Runtime model locations

Active/current model:

- `/userdata/Models/model.cvimodel`
- `/userdata/Models/model.json`

A compatibility/alternate directory also exists:

- `/userdata/MODEL/model.cvimodel`
- `/userdata/MODEL/model.json`

Factory presets:

- `/usr/share/supervisor/models/yolo11n_detection_cv181x_int8.cvimodel`
- `/usr/share/supervisor/models/yolo11n_detection_cv181x_int8.json`
- `/usr/share/supervisor/models/yolo11n_segment_cv181x_int8.cvimodel`
- `/usr/share/supervisor/models/yolo11n_segment_cv181x_int8.json`
- `/usr/share/supervisor/models/yolo11n_pose_cv181x_int8.cvimodel`
- `/usr/share/supervisor/models/yolo11n_pose_cv181x_int8.json`
- `/usr/share/supervisor/models/yolo11n_classify_cv181x_int8.cvimodel`
- `/usr/share/supervisor/models/yolo11n_classify_cv181x_int8.json`

## Model metadata evidence

The active model JSON reports:

- `model_name`: `YOLO11n Detection`
- `model_format`: `cvimodel`
- `author`: `Ultralytics`
- `description`: `YOLO11n official model from Ultralytics.`
- `classes`: COCO 80-class list

Factory presets are also named `YOLO11n ...`, not YOLO26.

## Conversion tooling location

The camera itself does not include:

- `model_transform`
- `model_deploy`

So conversion is not expected to happen on-device. The reCamera consumes the resulting `cvimodel` and metadata JSON.

## Conversion failure interpretation

The reported error:

```text
RuntimeError: Error, can't find [
  '/model.23/cv2.0/cv2.0.2/Conv_output_0',
  '/model.23/cv3.0/cv3.0.2/Conv_output_0',
  ...
] in model
```

means TPU-MLIR was explicitly asked to select YOLO11n internal tensors that are not present in the provided ONNX file.

This is an ONNX graph/name mismatch before actual cvimodel deployment. It does not prove the camera cannot run the model; it proves the copied `--output_names` list is wrong for that ONNX.

## Current hypothesis

The Seeed conversion wiki remains YOLO11n-centric, while Ultralytics' reCamera page currently claims YOLO26 preinstalled/custom support. The inspected hardware image and presets are YOLO11n.

For YOLO26, either:

1. Ultralytics/Seeed have newer firmware/tooling not present on this unit, or
2. their documentation is ahead of the shipped image, or
3. custom YOLO26 conversion requires manually selecting new pre-head outputs and likely validating postprocessing compatibility in `sscma-node`.

## Next safe steps

1. Inspect the YOLO26 ONNX using `tools/inspect_onnx.py`.
2. Generate model_transform command using actual tensor names via `tools/make_recamera_transform_cmd.py`.
3. Convert in Sophgo TPU-MLIR environment.
4. Do not overwrite `/userdata/Models/model.*` until the resulting metadata and runtime compatibility are understood.
5. Prefer backing up active `/userdata/Models/model.*` before any deployment.
