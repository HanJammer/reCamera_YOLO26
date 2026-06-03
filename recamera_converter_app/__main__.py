from __future__ import annotations

import shutil
import threading
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI

from .converter import (
    DOCKER_IMAGE,
    JOBS_ROOT,
    convert_prepared,
    find_docker_cli,
    new_job_dir,
    read_job_status,
    save_upload,
    write_job_status,
    finalize_outputs,
)
from .model_info import read_classes

APP_DIR = Path(__file__).resolve().parent
DEFAULT_TEST_IMAGE = APP_DIR / "static" / "test.jpg"
app = FastAPI(title="reCamera ONNX Converter")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


def _copy_upload(upload: UploadFile, dst: Path) -> None:
    with dst.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh)


def _run_job(
    *,
    job_id: str,
    job_dir: Path,
    onnx_path: Path,
    image_path: Path,
    image_name: str,
    calibration_count: int,
    model_name: str,
    classes_text: str,
    allow_pull: bool,
    docker_cli: str | None,
) -> None:
    try:
        convert_prepared(
            job_id=job_id,
            job_dir=job_dir,
            onnx_path=onnx_path,
            image_path=image_path,
            image_name=image_name,
            calibration_count=calibration_count,
            model_name=model_name,
            classes_text=classes_text,
            allow_pull=allow_pull,
            docker_cli=docker_cli,
        )
    except Exception as exc:
        log_path = job_dir / "logs" / "conversion.log"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\nERROR: {exc}\n")
        try:
            finalize_outputs(
                job_id=job_id,
                job_dir=job_dir,
                model_name=model_name,
                classes_text=classes_text,
                status="ok",
            )
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write("Recovered outputs after a job exception.\n")
        except Exception:
            write_job_status(job_dir, status="failed", message=str(exc))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    coco_csv = ", ".join(read_classes())
    detected_docker = find_docker_cli()
    docker_hint = "Docker Desktop/Engine must be installed and running."
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "coco_csv": coco_csv,
            "docker_hint": docker_hint,
            "detected_docker": detected_docker or "",
            "docker_image": DOCKER_IMAGE,
            "default_test_image": DEFAULT_TEST_IMAGE.exists(),
        },
    )


@app.post("/convert", response_class=HTMLResponse)
def convert_route(
    request: Request,
    onnx_file: UploadFile = File(...),
    test_image: UploadFile | None = File(None),
    calibration_images: list[UploadFile] | None = File(None),
    model_name: str = Form("yolo26n"),
    classes_text: str = Form(""),
    pull_image: str | None = Form(None),
    docker_cli: str = Form(""),
):
    job_id, job_dir = new_job_dir()
    write_job_status(job_dir, status="queued", message="Uploaded files; conversion thread will start now")

    onnx_path = job_dir / "input" / "model.onnx"
    _copy_upload(onnx_file, onnx_path)

    if test_image and test_image.filename:
        suffix = Path(test_image.filename).suffix or ".jpg"
        image_name = f"test{suffix}"
        image_path = job_dir / "input" / image_name
        _copy_upload(test_image, image_path)
    else:
        image_name = "test.jpg"
        image_path = job_dir / "input" / image_name
        save_upload(DEFAULT_TEST_IMAGE.open("rb"), image_path)

    calibration_dir = job_dir / "calibration"
    calibration_dir.mkdir()
    calibration_count = 0
    if calibration_images:
        for idx, upload in enumerate(calibration_images, start=1):
            if not upload.filename:
                continue
            suffix = Path(upload.filename).suffix or ".jpg"
            _copy_upload(upload, calibration_dir / f"calibration_{idx:04d}{suffix}")
            calibration_count += 1
    if calibration_count == 0:
        shutil.copy2(image_path, calibration_dir / "calibration_0001.jpg")
        calibration_count = 1

    worker = threading.Thread(
        target=_run_job,
        kwargs={
            "job_id": job_id,
            "job_dir": job_dir,
            "onnx_path": onnx_path,
            "image_path": image_path,
            "image_name": image_name,
            "calibration_count": calibration_count,
            "model_name": model_name.strip() or "model",
            "classes_text": classes_text,
            "allow_pull": bool(pull_image),
            "docker_cli": docker_cli.strip() or None,
        },
        daemon=True,
    )
    worker.start()
    return templates.TemplateResponse(request, "job.html", {"job_id": job_id, "job_dir": str(job_dir)})


@app.get("/job/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: str):
    job_dir = JOBS_ROOT / job_id
    if not job_dir.exists():
        return HTMLResponse("Not found", status_code=404)
    return templates.TemplateResponse(request, "job.html", {"job_id": job_id, "job_dir": str(job_dir)})


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    job_dir = JOBS_ROOT / job_id
    if not job_dir.exists():
        return JSONResponse({"status": "missing", "message": "Job not found"}, status_code=404)
    return read_job_status(job_dir)


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
