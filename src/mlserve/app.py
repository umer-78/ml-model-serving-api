"""The HTTP service."""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from .metrics import metrics
from .registry import InvalidBundle, ModelNotFound, ModelRegistry
from .schemas import BatchRequest, BatchResponse, Prediction, PredictRequest, PredictResponse

MODELS_DIR = Path(os.environ.get("MLSERVE_MODELS", Path(__file__).resolve().parents[2] / "models"))
logger = logging.getLogger("mlserve")
logging.basicConfig(level=os.environ.get("MLSERVE_LOG_LEVEL", "INFO"),
                    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')


def create_app(models_dir: Path | None = None) -> FastAPI:
    registry = ModelRegistry(models_dir or MODELS_DIR)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load the model at start-up, not on the first request, so a broken
        # bundle fails the deploy instead of the first customer.
        try:
            bundle = registry.active
            logger.info(f"loaded model {bundle.version}")
        except (ModelNotFound, InvalidBundle) as exc:
            logger.warning(f"starting without a model: {exc}")
        yield

    app = FastAPI(
        title="mlserve",
        version="1.0.0",
        description="Serves a scikit-learn model with validation, versioning and metrics.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_id_and_timing(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = f"{elapsed:.2f}"
        logger.info(f"{request.method} {request.url.path} -> {response.status_code} in {elapsed:.1f}ms "
                    f"[{request_id}]")
        return response

    def _bundle(version: str | None):
        try:
            return registry.load(version) if version else registry.active
        except ModelNotFound as exc:
            metrics.record_error("model_not_found")
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InvalidBundle as exc:
            metrics.record_error("invalid_bundle")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/health", tags=["ops"])
    def health() -> dict:
        try:
            bundle = registry.active
        except (ModelNotFound, InvalidBundle) as exc:
            return JSONResponse(status_code=503, content={"status": "degraded", "detail": str(exc)})
        return {"status": "ok", "model_version": bundle.version, "versions": registry.versions()}

    @app.get("/model", tags=["ops"])
    def model_info() -> dict:
        bundle = _bundle(None)
        return {"version": bundle.version, "features": bundle.features, "classes": bundle.classes,
                "metrics": bundle.metadata["metrics"], "created_at": bundle.metadata["created_at"],
                "algorithm": bundle.metadata.get("algorithm")}

    @app.post("/model/activate/{version}", tags=["ops"])
    def activate(version: str) -> dict:
        try:
            bundle = registry.activate(version)
        except ModelNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        logger.info(f"active model is now {bundle.version}")
        return {"active_version": bundle.version}

    @app.post("/predict", response_model=PredictResponse, tags=["inference"])
    def predict(body: PredictRequest) -> PredictResponse:
        bundle = _bundle(body.model_version)
        start = time.perf_counter()
        try:
            labels, probabilities = bundle.predict([body.features])
        except ValueError as exc:
            metrics.record_error("bad_feature_count")
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        elapsed = (time.perf_counter() - start) * 1000
        metrics.observe("/predict", elapsed, label=labels[0])
        return PredictResponse(
            prediction=Prediction(label=labels[0], confidence=max(probabilities[0].values()),
                                  probabilities=probabilities[0]),
            model_version=bundle.version, latency_ms=round(elapsed, 3),
        )

    @app.post("/predict/batch", response_model=BatchResponse, tags=["inference"])
    def predict_batch(body: BatchRequest) -> BatchResponse:
        bundle = _bundle(body.model_version)
        start = time.perf_counter()
        try:
            labels, probabilities = bundle.predict(body.rows)
        except ValueError as exc:
            metrics.record_error("bad_feature_count")
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        elapsed = (time.perf_counter() - start) * 1000
        for label in labels:
            metrics.observe("/predict/batch", elapsed / len(labels), label=label)
        return BatchResponse(
            predictions=[Prediction(label=lab, confidence=max(p.values()), probabilities=p)
                         for lab, p in zip(labels, probabilities, strict=True)],
            model_version=bundle.version, latency_ms=round(elapsed, 3), count=len(labels),
        )

    @app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
    def prometheus_metrics() -> str:
        return metrics.prometheus()

    @app.get("/stats", tags=["ops"])
    def stats() -> dict:
        return metrics.snapshot()

    return app


app = create_app()
