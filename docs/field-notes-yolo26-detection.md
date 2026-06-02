# Field notes: YOLO26 detection on reCamera

These notes summarize the investigation that led to the local converter.

## Problem

A YOLO26 ONNX conversion failed early in TPU-MLIR with an error like:

```text
RuntimeError: Error, can't find [
  '/model.23/cv2.0/cv2.0.2/Conv_output_0',
  '/model.23/cv3.0/cv3.0.2/Conv_output_0',
  ...
] in model
```

The failing output names match YOLO11-style detection head tensors. YOLO26 detection exports can use `one2one_cv2` / `one2one_cv3` tensors instead.

## Detection output names

For the tested YOLO26n detection ONNX, the working pre-head outputs were:

```text
/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0
/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0
/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0
/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0
/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0
/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0
```

## Confirmed conversion path

A local Sophgo TPU-MLIR Docker workflow successfully produced an F16 CV181x model:

```text
ONNX -> model_transform -> MLIR -> model_deploy -> .cvimodel
```

The successful deployment target was a reCamera-compatible F16 `.cvimodel` around 6.9 MB.

## Node-RED behavior

The Node-RED model node `classes` field is used to populate `model.json`. If the field is empty, the generated metadata can contain an empty classes list. Fill it with a comma-separated class list and press **Deploy** after changing model node configuration.

## Segmentation status

YOLO26 segmentation is not covered by the detection-only path. Segmentation exports include additional mask/proto outputs and need separate runtime/postprocess handling.
