"""
TradeForge Backtesting Engine
- Parallel processing with multiprocessing/Dask
- Market regime detection (bull/bear/sideways)
- VaR and Monte Carlo risk simulation
- Vectorbt integration
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

logger = logging.getLogger(__name__)

# ─── Market Regime Detection ─────────────────────────────────────────────────

def detect_regime(prices: pd.Series, lookback: int = 50) -> str:
    """
    Detect bull/bear/sideways regime using SMA crossover + volatility.
    Returns: 'bull' | 'bear' | 'sideways'
    """
    if len(prices) < lookback:
        return "unknown"

    sma_fast = prices.rolling(20).mean()
    sma_slow = prices.rolling(50).mean()
    volatility = prices.pct_change().rolling(20).std().iloc[-1]

    last_fast = sma_fast.iloc[-1]
    last_slow = sma_slow.iloc[-1]

    trend_strength = abs(last_fast - last_slow) / last_slow

    if trend_strength < 0.01:
        return "sideways"
    elif last_fast > last_slow:
        return "bull"
    else:
        return "bear"


# ─── Single Strategy Backtest (runs in subprocess) ───────────────────────────

def run_single_backtest(params: dict) -> dict:
    """
    Isolated backtest function for multiprocessing.
    Avoids vectorbt import at module level for subprocess safety.
    """
    try:
        import vectorbt as vbt

        symbol = params["symbol"]
        strategy_type = params["strategy_type"]
        config = params.get("config", {})
        start_date = params["start_date"]
        end_date = params["end_date"]
        initial_capital = params.get("initial_capital", 10000)

        # Download price data
        data = vbt.YFData.download(
            symbol,
            start=start_date,
            end=end_date,
            missing_index="drop"
        )
        close = data.get("Close")

        # Detect regime
        regime = detect_regime(close)

        # Build signals based on strategy type
        if strategy_type == "sma_crossover":
            fast = config.get("fast_period", 20)
            slow = config.get("slow_period", 50)
            fast_ma = vbt.MA.run(close, fast)
            slow_ma = vbt.MA.run(close, slow)
            entries = fast_ma.ma_crossed_above(slow_ma)
            exits = fast_ma.ma_crossed_below(slow_ma)

        elif strategy_type == "rsi_mean_reversion":
            period = config.get("rsi_period", 14)
            oversold = config.get("oversold", 30)
            overbought = config.get("overbought", 70)
            rsi = vbt.RSI.run(close, period)
            entries = rsi.rsi_crossed_below(oversold)
            exits = rsi.rsi_crossed_above(overbought)

        elif strategy_type == "bollinger_breakout":
            period = config.get("period", 20)
            std = config.get("std", 2.0)
            bb = vbt.BBANDS.run(close, period, std)
            entries = close > bb.upper
            exits = close < bb.middle

        else:
            # Default: buy and hold
            entries = pd.Series([True] + [False] * (len(close) - 1), index=close.index)
            exits = pd.Series([False] * (len(close) - 1) + [True], index=close.index)

        # Run portfolio simulation
        pf = vbt.Portfolio.from_signals(
            close,
            entries,
            exits,
            init_cash=initial_capital,
            fees=0.001,
            freq="D"
        )

        stats = pf.stats()
        return {
            "symbol": symbol,
            "regime": regime,
            "total_return": float(stats.get("Total Return [%]", 0)),
            "sharpe_ratio": float(stats.get("Sharpe Ratio", 0) or 0),
            "max_drawdown": float(stats.get("Max Drawdown [%]", 0)),
            "win_rate": float(stats.get("Win Rate [%]", 0) or 0),
            "total_trades": int(stats.get("Total Trades", 0)),
            "final_capital": initial_capital * (1 + float(stats.get("Total Return [%]", 0)) / 100),
            "status": "completed"
        }

    except Exception as e:
        return {"status": "failed", "error": str(e), "symbol": params.get("symbol")}


# ─── Parallel Backtest Runner ─────────────────────────────────────────────────

class BacktestEngine:

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)

    async def run_backtest(self, params: dict) -> dict:
        """Run a single backtest asynchronously in a process pool."""
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(executor, run_single_backtest, params)
        return result

    async def run_parallel_backtests(self, params_list: list[dict]) -> list[dict]:
        """
        Run multiple backtests in parallel (e.g., portfolio optimization,
        parameter sweep, or multi-symbol scan).
        """
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                loop.run_in_executor(executor, run_single_backtest, params)
                for params in params_list
            ]
            results = await asyncio.gather(*futures, return_exceptions=True)

        return [
            r if not isinstance(r, Exception) else {"status": "failed", "error": str(r)}
            for r in results
        ]

    async def parameter_sweep(
        self,
        symbol: str,
        strategy_type: str,
        param_grid: dict,
        start_date: str,
        end_date: str,
        initial_capital: float = 10000,
    ) -> list[dict]:
        """
        Grid search over parameter combinations to find optimal settings.

        Example param_grid:
        {
            "fast_period": [10, 20, 30],
            "slow_period": [50, 100, 200]
        }
        """
        import itertools

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))

        params_list = []
        for combo in combinations:
            config = dict(zip(keys, combo))
            params_list.append({
                "symbol": symbol,
                "strategy_type": strategy_type,
                "config": config,
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": initial_capital,
            })

        results = await self.run_parallel_backtests(params_list)

        # Sort by Sharpe ratio
        completed = [r for r in results if r.get("status") == "completed"]
        completed.sort(key=lambda x: x.get("sharpe_ratio", 0), reverse=True)
        return completed


# ─── Monte Carlo Risk Simulation ─────────────────────────────────────────────

class RiskSimulator:

    @staticmethod
    def monte_carlo_var(
        returns: np.ndarray,
        confidence: float = 0.95,
        simulations: int = 10000,
        horizon_days: int = 1,
    ) -> dict:
        """
        Monte Carlo Value at Risk simulation.
        Returns VaR, CVaR, and distribution stats.
        """
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Simulate future returns
        simulated_returns = np.random.normal(
            mean_return * horizon_days,
            std_return * np.sqrt(horizon_days),
            simulations
        )

        var = np.percentile(simulated_returns, (1 - confidence) * 100)
        cvar = simulated_returns[simulated_returns <= var].mean()

        return {
            "var_95": float(var),
            "cvar_95": float(cvar),
            "mean_return": float(mean_return),
            "std_return": float(std_return),
            "worst_case": float(simulated_returns.min()),
            "best_case": float(simulated_returns.max()),
            "simulations": simulations,
            "horizon_days": horizon_days,
        }

    @staticmethod
    def historical_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Historical VaR - percentile of actual return distribution."""
        return float(np.percentile(returns, (1 - confidence) * 100))

    @staticmethod
    def portfolio_var(
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """Parametric VaR for a portfolio using covariance matrix."""
        from scipy import stats
        portfolio_std = np.sqrt(weights @ cov_matrix @ weights)
        z_score = stats.norm.ppf(1 - confidence)
        return float(z_score * portfolio_std)


# ─── Volatility Prediction (ML) ──────────────────────────────────────────────

class VolatilityPredictor:
    """
    Simple ML-based volatility predictor using scikit-learn.
    Used for predictive alerts and adaptive position sizing.
    """

    def __init__(self):
        self.model = None
        self.is_trained = False

    def prepare_features(self, prices: pd.Series) -> pd.DataFrame:
        """Build feature matrix from price series."""
        df = pd.DataFrame({"close": prices})
        df["returns"] = df["close"].pct_change()
        df["vol_5"] = df["returns"].rolling(5).std()
        df["vol_10"] = df["returns"].rolling(10).std()
        df["vol_20"] = df["returns"].rolling(20).std()
        df["rsi"] = self._compute_rsi(df["close"])
        df["return_lag1"] = df["returns"].shift(1)
        df["return_lag2"] = df["returns"].shift(2)
        df["return_lag5"] = df["returns"].shift(5)
        df["target_vol"] = df["returns"].rolling(5).std().shift(-5)  # future vol
        return df.dropna()

    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def train(self, prices: pd.Series) -> dict:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error

        df = self.prepare_features(prices)
        feature_cols = ["vol_5", "vol_10", "vol_20", "rsi", "return_lag1", "return_lag2", "return_lag5"]
        X = df[feature_cols].values
        y = df["target_vol"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return {"mse": float(mse), "rmse": float(np.sqrt(mse))}

    def predict(self, prices: pd.Series) -> float:
        """Predict next 5-day volatility."""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        df = self.prepare_features(prices)
        feature_cols = ["vol_5", "vol_10", "vol_20", "rsi", "return_lag1", "return_lag2", "return_lag5"]
        X_last = df[feature_cols].iloc[-1:].values
        return float(self.model.predict(X_last)[0])
