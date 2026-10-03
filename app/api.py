"""Local FastAPI demo: one detector per server process, loaded at startup."""
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from app.detector import DetectionResult, YOLOv4Detector

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
APP_DIR = Path(__file__).resolve().parent

class UploadResult(DetectionResult):
    """Keep the existing detection fields and add the original upload filename."""
    filename: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load weights once per process, then share the detector for its lifetime.

    A missing model leaves the frontend/docs usable; health and inference
    report 503 so the browser can explain how to restore the local demo.
    """
    app.state.detector = None
    try:
        app.state.detector = await run_in_threadpool(YOLOv4Detector)
    except (OSError, ValueError, cv2.error) as exc:
        logging.getLogger(__name__).warning('Detector unavailable: %s', exc)
    yield
    app.state.detector = None

app = FastAPI(title='YOLOv4 Object Detection Demo', lifespan=lifespan)
# Absolute paths keep assets accessible regardless of the server's working directory.
app.mount('/static', StaticFiles(directory=APP_DIR / 'static'), name='static')

@app.get('/')
def frontend() -> FileResponse:
    """Serve a plain HTML interface without a template engine or build step."""
    return FileResponse(APP_DIR / 'templates' / 'index.html', media_type='text/html')

@app.get('/favicon.ico', include_in_schema=False)
def favicon() -> RedirectResponse:
    """Handle browsers that request the conventional favicon URL automatically."""
    return RedirectResponse('/static/favicon.svg')

@app.get('/api/info')
def info() -> dict[str, str]:
    """Expose API metadata separately from the browser homepage."""
    return {'name': 'YOLOv4 Object Detection Demo', 'docs': '/docs',
            'description': 'Local image inference using pretrained YOLOv4 on COCO.'}

@app.get('/health')
def health(request: Request) -> dict[str, str]:
    """Report model readiness, rather than only checking that the server is alive."""
    if getattr(request.app.state, 'detector', None) is None:
        raise HTTPException(503, 'Model unavailable. Check model files and restart; see models/README.md.')
    return {'status': 'ok', 'model': 'YOLOv4'}

@app.post('/detect', response_model=UploadResult)
def detect(request: Request, file: UploadFile = File(...)) -> UploadResult:
    """Return counts, original-pixel boxes, joint confidences, and local timing.

    This synchronous route runs in FastAPI's worker thread pool, avoiding a
    blocked async event loop during CPU inference. The shared detector guards
    its mutable network with a lock; weights are never reloaded per upload.
    """
    try:
        filename = (file.filename or '').replace('\\', '/').split('/')[-1]
        suffix = Path(filename).suffix.lower()
        if suffix not in {'.jpg', '.jpeg', '.png'}:
            raise HTTPException(415, 'Upload a JPG, JPEG, or PNG image.')
        if file.content_type not in {'image/jpeg', 'image/png'}:
            raise HTTPException(415, 'Content type must be image/jpeg or image/png.')
        # Extension/MIME checks alone cannot establish that a file is an image.
        # Limit compressed bytes, check its signature, and require valid decoding.
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, 'Image upload must be at most 10 MiB.')
        is_png = data.startswith(b'\x89PNG\r\n\x1a\n')
        is_jpeg = data.startswith(b'\xff\xd8\xff')
        if not data or not (is_png if suffix == '.png' else is_jpeg):
            raise HTTPException(400, 'File contents do not match the image extension.')
        if file.content_type != ('image/png' if is_png else 'image/jpeg'):
            raise HTTPException(400, 'Content type does not match the image contents.')
        try:
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        except cv2.error as exc:
            raise HTTPException(400, 'Cannot decode uploaded image.') from exc
        if image is None:
            raise HTTPException(400, 'Cannot decode uploaded image.')
        detector = getattr(request.app.state, 'detector', None)
        if detector is None:
            raise HTTPException(503, 'Model unavailable. Place weights in models/ and restart the API.')
        return {'filename': filename, **detector.detect(image)}
    finally:
        file.file.close()
