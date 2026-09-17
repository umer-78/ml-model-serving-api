"""Request and response models. Pydantic does the validating, not the handlers."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator

Features = Annotated[list[float], Field(min_length=1, max_length=512,
                                        description="feature values, in the order given by /model")]


class PredictRequest(BaseModel):
    features: Features
    model_version: str | None = Field(default=None, description="defaults to the active version")

    @field_validator("features")
    @classmethod
    def finite_values(cls, values: list[float]) -> list[float]:
        for v in values:
            if v != v or v in (float("inf"), float("-inf")):  # NaN or inf
                raise ValueError("features must be finite numbers")
        return values


class BatchRequest(BaseModel):
    rows: list[Features] = Field(min_length=1, max_length=1000)
    model_version: str | None = None


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    prediction: Prediction
    model_version: str
    latency_ms: float


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str
    latency_ms: float
    count: int
