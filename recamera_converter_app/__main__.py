from __future__ import annotations

import io
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .converter import JOBS_ROOT, convert
from .model_info import read_classes

APP_DIR = Path(__file__).resolve().parent
app = FastAPI(title="reCamera ONNX Converter")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


class UploadStream:
    def __init__(self, upload: UploadFile):
        self.upload = upload
        self.filename = upload.filename

    def read(self, n: int = -1) -> bytes:
        return self.upload.file.read(n)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    coco_csv = ", ".join(read_classes())
    docker_hint = "Docker Desktop/Engine must be installed and running."
    return templates.TemplateResponse(request, "index.html", {"coco_csv": coco_csv, "docker_hint": docker_hint})


@app.post("/convert", response_class=HTMLResponse)
def convert_route(
    request: Request,
    onnx_file: UploadFile = File(...),
    test_image: UploadFile = File(...),
    model_name: str = Form("yolo26n"),
    classes_text: str = Form(""),
    pull_image: str | None = Form(None),
):
    try:
        result = convert(
            onnx_file=UploadStream(onnx_file),
            test_image=UploadStream(test_image),
            model_name=model_name.strip() or "model",
            classes_text=classes_text,
            allow_pull=bool(pull_image),
        )
        return templates.TemplateResponse(request, "result.html", {"result": result, "ok": True})
    except Exception as exc:
        # Best-effort: show newest job log if one exists.
        logs = str(exc)
        if JOBS_ROOT.exists():
            jobs = sorted([p for p in JOBS_ROOT.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
            if jobs:
                log_path = jobs[0] / "logs" / "conversion.log"
                if log_path.exists():
                    logs += "\n\n--- latest conversion.log ---\n" + log_path.read_text(errors="replace")[-12000:]
        return templates.TemplateResponse(request, "result.html", {"ok": False, "error": logs})


@app.get("/download/{job_id}/{filename}")
def download(job_id: str, filename: str):
    path = JOBS_ROOT / job_id / "output" / filename
    if not path.exists():
        return HTMLResponse("Not found", status_code=404)
    return FileResponse(path, filename=filename)


def main() -> None:
    url = "http://127.0.0.1:7860"
    print(f"Opening {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    uvicorn.run(app, host="127.0.0.1", port=7860)


if __name__ == "__main__":
    main()
