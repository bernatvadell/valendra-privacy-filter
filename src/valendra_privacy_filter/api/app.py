"""FastAPI service that never logs inputs or detected values."""

from contextlib import asynccontextmanager
import os
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator

from valendra_privacy_filter.inference import Detector


class InputBatch(BaseModel):
    text: str | None = None
    texts: list[str] | None = None

    @model_validator(mode="after")
    def exactly_one_input(self):
        if (self.text is None) == (self.texts is None):
            raise ValueError("Provide exactly one of text or texts")
        return self


class BodyLimitMiddleware:
    def __init__(self, app, max_body_bytes):
        self.app, self.max_body_bytes = app, max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_body_bytes:
                return await JSONResponse({"detail": "Request body too large"}, status_code=413)(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        return await self.app(scope, replay, send)


def create_app(detector=None, max_chars=100_000, max_batch=32, max_body_bytes=1_000_000):
    if min(max_chars, max_batch, max_body_bytes) <= 0:
        raise ValueError("Limits must be positive")
    lock = threading.Lock()
    metrics = {"requests": 0, "documents": 0, "errors": 0}
    started = time.monotonic()

    @asynccontextmanager
    async def lifespan(app):
        if detector is not None:
            app.state.detector = detector
        else:
            path = os.environ.get("PRIVACY_FILTER_MODEL")
            if not path:
                raise RuntimeError("Set PRIVACY_FILTER_MODEL to a trained checkpoint")
            app.state.detector = Detector.from_pretrained(
                path, device=os.environ.get("PRIVACY_FILTER_DEVICE", "cpu"),
                precision=os.environ.get("PRIVACY_FILTER_PRECISION", "fp32"),
                max_length=int(os.environ.get("PRIVACY_FILTER_MAX_LENGTH", "512")),
                overlap=int(os.environ.get("PRIVACY_FILTER_OVERLAP", "64")),
            )
        yield

    app = FastAPI(title="Valendra Privacy Filter", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware, max_body_bytes=max_body_bytes)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        return JSONResponse({"detail": "Invalid request"}, status_code=422)

    def run(payload, redact=False):
        texts = [payload.text] if payload.text is not None else payload.texts
        if not texts or len(texts) > max_batch or any(len(text) > max_chars for text in texts):
            raise HTTPException(status_code=413, detail="Input limit exceeded")
        with lock:
            metrics["requests"] += 1
            try:
                if redact:
                    result = [{"text": app.state.detector.redact(text)} for text in texts]
                else:
                    result = [app.state.detector.detect(text) for text in texts]
                metrics["documents"] += len(texts)
            except Exception:
                metrics["errors"] += 1
                raise HTTPException(status_code=500, detail="Detection failed") from None
        return result[0] if payload.text is not None else {"results": result}

    @app.get("/health")
    def health():
        return {"status": "ok", "uptime_seconds": time.monotonic() - started, **metrics}

    @app.post("/detect")
    def detect(payload: InputBatch):
        return run(payload)

    @app.post("/redact")
    def redact(payload: InputBatch):
        return run(payload, redact=True)

    return app
