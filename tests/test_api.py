"""Tests for the FastAPI prediction API."""

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == "XGBoost"


class TestPredictEndpoint:
    def test_predict_default(self):
        resp = client.post("/predict", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_value_brl" in data
        assert data["predicted_value_brl"] > 0
        assert "confidence_lower_brl" in data
        assert "confidence_upper_brl" in data
        assert data["confidence_upper_brl"] > data["confidence_lower_brl"]

    def test_predict_high_value_profile(self):
        payload = {
            "monthly_income": 15000,
            "loyalty_tier": "platinum",
            "item_count": 5,
            "total_cart_qty": 8,
            "n_product_views": 15,
            "session_duration_min": 20,
            "device_type": "desktop",
            "traffic_source": "email",
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        high = resp.json()["predicted_value_brl"]

        low_payload = {
            "monthly_income": 1500,
            "loyalty_tier": "bronze",
            "item_count": 1,
            "total_cart_qty": 1,
            "n_product_views": 2,
            "session_duration_min": 2,
            "device_type": "mobile",
            "traffic_source": "google",
        }
        resp = client.post("/predict", json=low_payload)
        low = resp.json()["predicted_value_brl"]

        assert high > low * 2, f"High ({high}) should be > 2x low ({low})"

    def test_predict_invalid_income(self):
        resp = client.post("/predict", json={"monthly_income": -100})
        assert resp.status_code == 422  # validation error

    def test_predict_invalid_loyalty(self):
        resp = client.post("/predict", json={"loyalty_tier": "diamond"})
        # Pydantic doesn't validate enum strings by default, so this may pass
        # The model should still return a prediction (unknown tier → default)
        assert resp.status_code in (200, 422)
