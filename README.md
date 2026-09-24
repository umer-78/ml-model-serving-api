# ML Model Serving API

**Live demo:** https://umer-78.github.io/ml-model-serving-api/

[![CI](https://github.com/umer-78/ml-model-serving-api/actions/workflows/ci.yml/badge.svg)](https://github.com/umer-78/ml-model-serving-api/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)
![Docker](https://img.shields.io/badge/docker-ready-2496ed)
![License](https://img.shields.io/badge/license-MIT-green)

Training a model is the easy half. This is the other half: a FastAPI service that
serves a scikit-learn model with **input validation, model versioning, health
checks, Prometheus metrics, structured logs and a Docker image** — the things a
model needs before anyone else can depend on it.

```bash
mlserve train          # writes models/<version>/ with the pipeline + metadata
mlserve serve          # http://127.0.0.1:8000/docs
```

```bash
$ curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
    -d '{"features":[13.2,1.78,2.14,11.2,100,2.65,2.76,0.26,1.28,4.38,1.05,3.4,1050]}'
{"prediction":{"label":"class_0","confidence":0.961,
 "probabilities":{"class_0":0.961,"class_1":0.022,"class_2":0.017}},
 "model_version":"2026.09.17.1","latency_ms":12.652}
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | 200 with the active version, **503 when no model can be loaded** |
| `GET` | `/model` | features (in order), classes, training metrics, algorithm |
| `POST` | `/predict` | one row → label, confidence, full probability spread |
| `POST` | `/predict/batch` | up to 1,000 rows in one call |
| `POST` | `/model/activate/{version}` | switch the served version with no restart |
| `GET` | `/metrics` | Prometheus text format |
| `GET` | `/stats` | the same counters as JSON |
| `GET` | `/docs` | generated OpenAPI documentation |

## What makes it production-shaped

- **Versioned bundles.** Each `models/<version>/` holds the pipeline, and metadata
  with the feature order, class names, metrics, scikit-learn version and
  timestamp. The registry refuses to load a bundle missing any of that, so a
  half-written model never reaches traffic.
- **Fail at start-up, not on the first request.** The model loads during the
  lifespan hook; a broken bundle fails the deploy while `/health` reports 503.
- **Validation at the edge.** Pydantic rejects empty, non-numeric, NaN and
  oversized inputs; the handler only ever sees a clean row, and the wrong number
  of features returns 422 with the count it expected.
- **Rollback in one call.** `POST /model/activate/2026.01.01.0001` switches back
  to a known-good version without redeploying.
- **Observability.** Every response carries `x-request-id` and
  `x-response-time-ms`, logs are one JSON object per line, and `/metrics` exposes
  request counts, error reasons, a latency histogram and prediction counts by label.
- **Small, non-root image.** Multi-stage build; the model is trained in the build
  stage and baked in, the runtime stage has no build tools and runs as uid 10001,
  with a `HEALTHCHECK`.

## Run it

```bash
git clone https://github.com/umer-78/ml-model-serving-api.git
cd ml-model-serving-api
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

mlserve train --version 2026.09.17.1
mlserve versions
mlserve serve
```

With Docker:

```bash
docker compose up --build      # or: docker build -t mlserve . && docker run -p 8000:8000 mlserve
curl localhost:8000/health
```

## Load test

```bash
$ python scripts/load_test.py --requests 300 --concurrency 12
300 requests, 12 concurrent, 8.25s (36 req/s)
mean 325.8 ms   p50 331.1   p95 404.9   p99 439.2   max 448.7
```

Measured on one uvicorn worker in a small sandbox container, with a 250-tree
random forest. It is a baseline to improve, not a benchmark to quote: more
workers (`--workers 4`), a smaller forest or batching all move it a long way.

## The model

scikit-learn's wine dataset (178 samples, 13 features, 3 classes) through a
`StandardScaler` + `RandomForestClassifier`. Test accuracy is 1.00 and
five-fold cross-validated accuracy 0.967 — the dataset is small and easy, which
is fine, because the subject here is the serving, not the model. Point `train.py`
at your own data and the rest of the service is unchanged.

## Tests

```bash
ruff check .
python -m pytest -q     # 15 tests
```

They cover the bundle written at training time, the registry (listing, loading,
activation, and every failure mode), all endpoints, validation rejecting four
kinds of bad input, the metrics arithmetic, and the CLI. CI additionally builds
the Docker image and calls the running container.

## License

[MIT](LICENSE)
