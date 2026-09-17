import json

import pytest
from fastapi.testclient import TestClient

from mlserve.app import create_app
from mlserve.metrics import Metrics
from mlserve.registry import InvalidBundle, ModelNotFound, ModelRegistry
from mlserve.train import train

WINE_ROW = [13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050.0]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    train(version="2026.01.01.0001", output_dir=root)
    train(version="2026.02.02.0002", seed=7, output_dir=root)
    return root


@pytest.fixture
def client(models):
    with TestClient(create_app(models)) as c:
        yield c


# ------------------------------------------------------------------- training
def test_training_writes_a_complete_bundle(models):
    folder = models / "2026.01.01.0001"
    assert (folder / "model.joblib").exists()
    meta = json.loads((folder / "metadata.json").read_text())
    assert meta["version"] == "2026.01.01.0001"
    assert len(meta["features"]) == 13 and len(meta["classes"]) == 3
    assert meta["metrics"]["accuracy"] > 0.9
    assert (folder / "classification_report.txt").read_text().strip()


# ------------------------------------------------------------------- registry
def test_registry_lists_loads_and_activates(models):
    registry = ModelRegistry(models)
    assert registry.versions() == ["2026.01.01.0001", "2026.02.02.0002"]
    assert registry.latest() == "2026.02.02.0002"
    assert registry.active.version == "2026.02.02.0002"
    assert registry.activate("2026.01.01.0001").version == "2026.01.01.0001"
    assert registry.active.version == "2026.01.01.0001"


def test_registry_errors(tmp_path, models):
    with pytest.raises(ModelNotFound):
        ModelRegistry(tmp_path).latest()
    with pytest.raises(ModelNotFound):
        ModelRegistry(models).load("does-not-exist")
    broken = tmp_path / "1.0"
    broken.mkdir(parents=True)
    (broken / "model.joblib").write_bytes(b"x")
    (broken / "metadata.json").write_text('{"version": "1.0"}')
    with pytest.raises(InvalidBundle):
        ModelRegistry(tmp_path).load("1.0")


def test_bundle_rejects_the_wrong_feature_count(models):
    bundle = ModelRegistry(models).active
    with pytest.raises(ValueError, match="expected 13 features"):
        bundle.predict([[1.0, 2.0]])


# ----------------------------------------------------------------------- API
def test_health_and_model_info(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == "2026.02.02.0002"
    info = client.get("/model").json()
    assert len(info["features"]) == 13
    assert info["classes"] == ["class_0", "class_1", "class_2"]
    assert info["metrics"]["accuracy"] > 0.9


def test_health_is_503_without_a_model(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        response = c.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"


def test_predict(client):
    response = client.post("/predict", json={"features": WINE_ROW})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["label"] in {"class_0", "class_1", "class_2"}
    assert 0 < body["prediction"]["confidence"] <= 1
    assert sum(body["prediction"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert body["model_version"] == "2026.02.02.0002"
    assert "x-request-id" in response.headers


def test_predict_with_an_explicit_version(client):
    body = client.post("/predict", json={"features": WINE_ROW, "model_version": "2026.01.01.0001"}).json()
    assert body["model_version"] == "2026.01.01.0001"
    assert client.post("/predict", json={"features": WINE_ROW, "model_version": "nope"}).status_code == 404


def test_validation_rejects_bad_input(client):
    assert client.post("/predict", json={"features": []}).status_code == 422          # empty
    assert client.post("/predict", json={"features": [1, 2, 3]}).status_code == 422   # wrong count
    assert client.post("/predict", json={"features": ["a"] * 13}).status_code == 422  # not numbers
    assert client.post("/predict", json={}).status_code == 422                        # missing field


def test_batch_prediction(client):
    body = client.post("/predict/batch", json={"rows": [WINE_ROW, WINE_ROW]}).json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert client.post("/predict/batch", json={"rows": []}).status_code == 422


def test_activate_switches_the_default_version(client):
    assert client.post("/model/activate/2026.01.01.0001").json()["active_version"] == "2026.01.01.0001"
    assert client.post("/predict", json={"features": WINE_ROW}).json()["model_version"] == "2026.01.01.0001"
    assert client.post("/model/activate/missing").status_code == 404
    client.post("/model/activate/2026.02.02.0002")


def test_metrics_endpoints(client):
    client.post("/predict", json={"features": WINE_ROW})
    text = client.get("/metrics").text
    assert 'mlserve_requests_total{endpoint="/predict"}' in text
    assert "mlserve_latency_ms_bucket" in text
    stats = client.get("/stats").json()
    assert stats["total_predictions"] >= 1
    assert stats["average_latency_ms"] >= 0


# ------------------------------------------------------------------- metrics
def test_metrics_counters_and_histogram():
    m = Metrics()
    m.observe("/predict", 3.0, label="class_1")
    m.observe("/predict", 120.0, label="class_2")
    m.record_error("bad_feature_count")
    assert m.requests["/predict"] == 2
    assert m.predictions["class_1"] == 1
    assert m.average_latency_ms == pytest.approx(61.5)
    assert m.latency_buckets[5] == 1 and m.latency_buckets[250] == 2
    assert "mlserve_errors_total{reason=\"bad_feature_count\"} 1" in m.prometheus()


def test_cli_train_and_versions(tmp_path, capsys):
    from mlserve.cli import main
    assert main(["train", "--version", "9.9.9", "--models", str(tmp_path)]) == 0
    assert "9.9.9" in capsys.readouterr().out
    assert main(["versions", "--models", str(tmp_path)]) == 0
    assert "accuracy" in capsys.readouterr().out
    assert main(["versions", "--models", str(tmp_path / "empty")]) == 1
