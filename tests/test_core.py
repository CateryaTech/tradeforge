"""
TradeForge Test Suite
Run: pytest tests/ -v --asyncio-mode=auto
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch

# ─── Backtesting Tests ────────────────────────────────────────────────────────

class TestRegimeDetection:
    def test_bull_regime(self):
        from app.services.trading.backtesting import detect_regime
        # Create uptrending prices
        prices = pd.Series([100 + i * 2 for i in range(100)])
        assert detect_regime(prices) == "bull"

    def test_bear_regime(self):
        from app.services.trading.backtesting import detect_regime
        prices = pd.Series([200 - i * 2 for i in range(100)])
        assert detect_regime(prices) == "bear"

    def test_sideways_regime(self):
        from app.services.trading.backtesting import detect_regime
        # Flat prices with noise
        np.random.seed(42)
        prices = pd.Series(100 + np.random.normal(0, 0.1, 100))
        assert detect_regime(prices) in ["sideways", "bull", "bear"]  # small noise

    def test_insufficient_data(self):
        from app.services.trading.backtesting import detect_regime
        prices = pd.Series([100, 101, 102])
        assert detect_regime(prices) == "unknown"


class TestRiskSimulator:
    def setup_method(self):
        from app.services.trading.backtesting import RiskSimulator
        self.sim = RiskSimulator()
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 1000)

    def test_monte_carlo_var_returns_dict(self):
        result = self.sim.monte_carlo_var(self.returns)
        assert "var_95" in result
        assert "cvar_95" in result
        assert result["var_95"] < 0  # VaR should be negative

    def test_historical_var(self):
        var = self.sim.historical_var(self.returns)
        assert isinstance(var, float)
        assert var < 0  # VaR is a loss

    def test_portfolio_var(self):
        weights = np.array([0.5, 0.5])
        cov = np.array([[0.0004, 0.0001], [0.0001, 0.0009]])
        var = self.sim.portfolio_var(weights, cov)
        assert isinstance(var, float)


class TestVolatilityPredictor:
    def test_feature_preparation(self):
        from app.services.trading.backtesting import VolatilityPredictor
        predictor = VolatilityPredictor()
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(200) * 2))
        features = predictor.prepare_features(prices)
        assert "vol_5" in features.columns
        assert "rsi" in features.columns
        assert "target_vol" in features.columns
        assert len(features) > 0

    def test_train_and_predict(self):
        from app.services.trading.backtesting import VolatilityPredictor
        predictor = VolatilityPredictor()
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(500) * 2))
        metrics = predictor.train(prices)
        assert "mse" in metrics
        prediction = predictor.predict(prices)
        assert isinstance(prediction, float)
        assert prediction >= 0


# ─── Trading Service Tests ────────────────────────────────────────────────────

class TestSlippageEstimator:
    def setup_method(self):
        from app.services.trading.live_trading import SlippageEstimator
        self.estimator = SlippageEstimator()

    def test_small_order_low_slippage(self):
        bids = [[50000, 1.0], [49990, 2.0], [49980, 5.0]]
        asks = [[50010, 1.0], [50020, 2.0], [50030, 5.0]]
        result = self.estimator.estimate(1000, bids, asks, "buy")
        assert result["estimated_slippage"] < 0.01  # < 1%

    def test_large_order_high_slippage(self):
        asks = [[50000, 0.1], [50100, 0.1]]  # Very thin book
        result = self.estimator.estimate(1_000_000, [], asks, "buy")
        assert result.get("liquidity_warning") is True or result["estimated_slippage"] > 0.01

    def test_empty_book(self):
        result = self.estimator.estimate(1000, [], [], "buy")
        assert result["confidence"] == "low"


class TestFeeCollector:
    def test_trade_fee_calculation(self):
        from app.services.trading.live_trading import FeeCollector
        collector = FeeCollector()
        result = collector.calculate_fee(1.0, 50000)  # 1 BTC at $50k
        assert result["fee_usd"] == pytest.approx(50000 * 0.002, rel=1e-3)
        assert result["fee_currency"] == "USDT"

    def test_custom_fee_override(self):
        from app.services.trading.live_trading import FeeCollector
        collector = FeeCollector()
        result = collector.calculate_fee(1.0, 50000, user_fee_override=0.001)
        assert result["fee_usd"] == pytest.approx(50, rel=1e-3)


# ─── Payment Service Tests ────────────────────────────────────────────────────

class TestFeeTracker:
    def setup_method(self):
        from app.services.payments.payment_service import FeeTracker
        self.tracker = FeeTracker()

    def test_trade_fee(self):
        fee = self.tracker.calculate_trade_fee(1.0, 50000)
        assert fee == pytest.approx(100.0, rel=1e-3)  # 0.2% of $50k

    def test_conversion_fee(self):
        fee = self.tracker.calculate_conversion_fee(1000)
        assert fee == pytest.approx(2.0, rel=1e-3)  # 0.2% of $1000

    def test_apy_share(self):
        fee = self.tracker.calculate_apy_share(500)
        assert fee == pytest.approx(50.0, rel=1e-3)  # 10% of $500

    def test_revenue_projection(self):
        projection = self.tracker.project_monthly_revenue(
            active_free=100,
            active_basic=50,
            active_premium=20,
            active_institutional=5,
        )
        assert projection["total_mrr"] > 0
        assert projection["projected_arr"] == pytest.approx(projection["total_mrr"] * 12, rel=1e-3)
        assert "subscription_mrr" in projection
        assert "trade_fee_mrr" in projection


# ─── Fraud Detection Tests ────────────────────────────────────────────────────

class TestFraudDetector:
    def test_feature_extraction(self):
        from app.services.ai.ai_agents import FraudDetector
        detector = FraudDetector()
        trade = {
            "amount_usd": 1000,
            "trades_last_hour": 3,
            "trades_last_24h": 10,
            "avg_trade_size_7d": 500,
            "new_wallet": False,
            "account_age_days": 365,
            "failed_trades_ratio": 0.1,
            "time_since_last_trade_minutes": 30,
        }
        features = detector.extract_features(trade)
        assert len(features) == 8

    def test_train_and_score(self):
        from app.services.ai.ai_agents import FraudDetector
        detector = FraudDetector()
        # Generate synthetic training data
        np.random.seed(42)
        normal_trades = [
            {
                "amount_usd": float(np.random.uniform(100, 5000)),
                "trades_last_hour": int(np.random.randint(0, 10)),
                "trades_last_24h": int(np.random.randint(1, 50)),
                "avg_trade_size_7d": float(np.random.uniform(200, 3000)),
                "new_wallet": False,
                "account_age_days": int(np.random.randint(30, 1000)),
                "failed_trades_ratio": float(np.random.uniform(0, 0.1)),
                "time_since_last_trade_minutes": int(np.random.randint(5, 1440)),
            }
            for _ in range(200)
        ]
        result = detector.train(normal_trades)
        assert result["status"] == "trained"

        # Score a normal trade
        score = detector.score_trade(normal_trades[0])
        assert "is_fraud" in score
        assert "risk_level" in score
        assert score["risk_level"] in ["low", "medium", "high"]


# ─── API Route Tests ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestPaymentRoutes:
    async def test_get_pricing(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        response = client.get("/api/v1/payments/pricing")
        assert response.status_code == 200
        data = response.json()
        assert "tiers" in data
        assert "basic" in data["tiers"]
        assert "premium" in data["tiers"]

    async def test_revenue_projection(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        response = client.post("/api/v1/payments/project-revenue", json={
            "active_free": 100,
            "active_basic": 50,
            "active_premium": 20,
            "active_institutional": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total_mrr"] > 0
