"""
TradeForge AI Agent Service
- LangChain-based autonomous trading agents
- CrewAI multi-agent system for yield farming
- LLM integration (Groq/OpenRouter) for strategy generation
- Fraud detection using ML
"""

import logging
import json
import asyncio
from typing import Optional
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


# ─── LLM Client (Multi-Provider) ─────────────────────────────────────────────

class LLMClient:
    """
    Unified LLM client supporting Groq, OpenRouter, Together AI.
    Keys loaded from environment variables only (never hardcoded).
    """

    async def complete(
        self,
        prompt: str,
        system: str = "You are a professional quantitative trading analyst.",
        provider: str = "groq",
        max_tokens: int = 1024,
    ) -> str:
        if provider == "groq":
            return await self._call_groq(prompt, system, max_tokens)
        elif provider == "openrouter":
            return await self._call_openrouter(prompt, system, max_tokens)
        elif provider == "together":
            return await self._call_together(prompt, system, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def _call_groq(self, prompt: str, system: str, max_tokens: int) -> str:
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not configured")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
                json={
                    "model": "mixtral-8x7b-32768",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _call_openrouter(self, prompt: str, system: str, max_tokens: int) -> str:
        if not settings.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not configured")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://tradeforge.io",
                    "X-Title": "TradeForge AaaS",
                },
                json={
                    "model": "anthropic/claude-3-haiku",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _call_together(self, prompt: str, system: str, max_tokens: int) -> str:
        if not settings.TOGETHER_API_KEY:
            raise ValueError("TOGETHER_API_KEY not configured")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.TOGETHER_API_KEY}"},
                json={
                    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]


# ─── Strategy Generator Agent ─────────────────────────────────────────────────

class StrategyGeneratorAgent:
    """
    AI agent that generates trading strategy parameters based on
    market conditions and user preferences.
    """

    def __init__(self):
        self.llm = LLMClient()

    async def generate_strategy(
        self,
        market_data_summary: dict,
        user_preferences: dict,
        provider: str = "groq",
    ) -> dict:
        """
        Generate a trading strategy config using LLM analysis.
        Returns JSON config compatible with the backtesting engine.
        """
        prompt = f"""
You are analyzing market data to generate an optimal trading strategy.

MARKET DATA:
- Symbol: {market_data_summary.get('symbol')}
- Current Price: ${market_data_summary.get('price')}
- 24h Change: {market_data_summary.get('change_24h')}%
- Volatility (30d): {market_data_summary.get('volatility_30d')}%
- Market Regime: {market_data_summary.get('regime')}
- Volume 24h: ${market_data_summary.get('volume_24h')}

USER PREFERENCES:
- Risk Tolerance: {user_preferences.get('risk_tolerance', 'medium')}
- Investment Horizon: {user_preferences.get('horizon', '1_month')}
- Max Drawdown Tolerance: {user_preferences.get('max_drawdown', 15)}%
- Capital: ${user_preferences.get('capital', 10000)}

Generate a JSON strategy config with these fields:
- strategy_type: one of [sma_crossover, rsi_mean_reversion, bollinger_breakout, arbitrage]
- config: strategy-specific parameters
- reasoning: brief explanation of why this strategy fits current conditions
- expected_sharpe: estimated Sharpe ratio
- risk_level: low/medium/high

Respond ONLY with valid JSON, no markdown.
"""

        response = await self.llm.complete(prompt, provider=provider)

        # Clean potential markdown
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "strategy_type": "sma_crossover",
                "config": {"fast_period": 20, "slow_period": 50},
                "reasoning": "Fallback to default SMA crossover due to parse error",
                "expected_sharpe": 0.5,
                "risk_level": "medium",
            }


# ─── Multi-Agent Yield Farming System ────────────────────────────────────────

class YieldFarmingAgent:
    """
    Autonomous yield farming agent that:
    1. Scans for best APY opportunities
    2. Rebalances positions automatically
    3. Compounds rewards
    4. Reports performance
    """

    def __init__(self):
        self.llm = LLMClient()

    async def analyze_opportunities(
        self,
        opportunities: list[dict],
        current_positions: list[dict],
        total_capital_usd: float,
    ) -> dict:
        """
        Use LLM to analyze yield opportunities and decide on reallocation.
        """
        prompt = f"""
You are an autonomous DeFi yield optimization agent.

CURRENT POSITIONS:
{json.dumps(current_positions, indent=2)}

AVAILABLE OPPORTUNITIES (sorted by APY):
{json.dumps(opportunities[:10], indent=2)}

TOTAL CAPITAL: ${total_capital_usd:,.2f}

Analyze the opportunities and current positions.
Return a JSON action plan with:
- "reallocations": list of {{from_protocol, to_protocol, amount_usd, reason}}
- "keep_positions": list of position IDs to maintain
- "estimated_apy_improvement": percentage improvement
- "risk_assessment": summary of risks

Consider: gas costs, impermanent loss risk, protocol security, diversification.
Respond ONLY with valid JSON.
"""
        response = await self.llm.complete(
            prompt,
            system="You are an expert DeFi yield optimizer. Always prioritize capital preservation.",
            provider="groq",
        )

        try:
            response = response.strip()
            if "```" in response:
                response = response.split("```")[1].lstrip("json").strip()
            return json.loads(response)
        except Exception:
            return {
                "reallocations": [],
                "keep_positions": [p.get("id") for p in current_positions],
                "estimated_apy_improvement": 0,
                "risk_assessment": "Could not parse AI response, maintaining current positions",
            }

    async def should_compound(
        self,
        protocol: str,
        pending_rewards_usd: float,
        gas_cost_usd: float,
    ) -> bool:
        """
        Decide if compounding is profitable after gas costs.
        Rule: compound if rewards > 5x gas cost.
        """
        return pending_rewards_usd > (gas_cost_usd * 5)


# ─── Fraud Detection ─────────────────────────────────────────────────────────

class FraudDetector:
    """
    ML-based fraud detection for suspicious trading activity.
    Uses feature engineering + Isolation Forest.
    """

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.scaler = None

    def extract_features(self, trade_data: dict) -> list:
        """Extract features from a trade for fraud scoring."""
        return [
            trade_data.get("amount_usd", 0),
            trade_data.get("trades_last_hour", 0),
            trade_data.get("trades_last_24h", 0),
            trade_data.get("avg_trade_size_7d", 0),
            1 if trade_data.get("new_wallet", False) else 0,
            trade_data.get("account_age_days", 365),
            trade_data.get("failed_trades_ratio", 0),
            trade_data.get("time_since_last_trade_minutes", 60),
        ]

    def train(self, historical_trades: list[dict]) -> dict:
        """Train Isolation Forest on historical trade data."""
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        X = np.array([self.extract_features(t) for t in historical_trades])
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            contamination=0.01,  # 1% expected fraud rate
            random_state=42,
            n_estimators=200,
        )
        self.model.fit(X_scaled)
        self.is_trained = True

        return {"status": "trained", "samples": len(X)}

    def score_trade(self, trade_data: dict) -> dict:
        """
        Score a trade for fraud risk.
        Returns: {is_fraud: bool, confidence: float, risk_level: str}
        """
        if not self.is_trained:
            return {"is_fraud": False, "confidence": 0.0, "risk_level": "unknown"}

        import numpy as np
        features = np.array([self.extract_features(trade_data)])
        features_scaled = self.scaler.transform(features)

        prediction = self.model.predict(features_scaled)[0]  # 1=normal, -1=anomaly
        score = self.model.score_samples(features_scaled)[0]

        # Convert score to probability (more negative = more anomalous)
        normalized = max(0, min(1, (score + 0.5) / 1.0))

        is_fraud = prediction == -1
        risk_level = "low" if normalized > 0.7 else ("medium" if normalized > 0.4 else "high")

        return {
            "is_fraud": is_fraud,
            "anomaly_score": float(score),
            "confidence": float(1 - normalized) if is_fraud else float(normalized),
            "risk_level": risk_level,
        }


# Global instances
llm_client = LLMClient()
strategy_agent = StrategyGeneratorAgent()
fraud_detector = FraudDetector()
