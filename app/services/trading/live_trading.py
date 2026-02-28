"""
TradeForge Live Trading Service
- Async order execution (<1s latency)
- Real-time slippage/volatility prediction
- Fee collection logic
- Multi-exchange via CCXT
"""

import asyncio
import logging
import time
from typing import Optional
import ccxt.async_support as ccxt
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


# ─── Platform Fee Collector ───────────────────────────────────────────────────

class FeeCollector:
    """Calculates and records platform fees on every trade."""

    def __init__(self):
        self.fee_percent = settings.TRADE_FEE_PERCENT

    def calculate_fee(self, amount: float, price: float, user_fee_override: Optional[float] = None) -> dict:
        fee_pct = user_fee_override or self.fee_percent
        fee_usd = amount * price * fee_pct
        return {
            "fee_percent": fee_pct,
            "fee_usd": fee_usd,
            "fee_currency": "USDT",
        }


# ─── Slippage Estimator ───────────────────────────────────────────────────────

class SlippageEstimator:
    """
    Estimates expected slippage before order submission.
    Uses order book depth + recent volatility.
    """

    def estimate(
        self,
        order_size_usd: float,
        orderbook_bids: list,
        orderbook_asks: list,
        side: str = "buy",
    ) -> dict:
        """
        Walk the order book to estimate fill price and slippage.
        """
        book = orderbook_asks if side == "buy" else orderbook_bids
        if not book:
            return {"estimated_slippage": 0.005, "confidence": "low"}

        mid_price = book[0][0]
        remaining = order_size_usd
        total_cost = 0.0
        levels_used = 0

        for price, size in book:
            available = price * size
            if remaining <= available:
                total_cost += remaining
                break
            else:
                total_cost += available
                remaining -= available
                levels_used += 1

        if remaining > 0:
            # Order size exceeds available liquidity
            return {"estimated_slippage": 0.05, "confidence": "low", "liquidity_warning": True}

        avg_fill_price = total_cost / order_size_usd
        slippage = abs(avg_fill_price - mid_price) / mid_price

        return {
            "estimated_slippage": float(slippage),
            "avg_fill_price": float(avg_fill_price),
            "mid_price": float(mid_price),
            "levels_used": levels_used,
            "confidence": "high" if levels_used < 5 else "medium",
        }


# ─── Exchange Manager ─────────────────────────────────────────────────────────

class ExchangeManager:
    """
    Manages CCXT async exchange connections.
    Supports Binance, Coinbase, Kraken, and more.
    """

    def __init__(self):
        self._exchanges: dict[str, ccxt.Exchange] = {}
        self.slippage_estimator = SlippageEstimator()
        self.fee_collector = FeeCollector()

    async def get_exchange(self, exchange_id: str, api_key: str = "", secret: str = "") -> ccxt.Exchange:
        key = f"{exchange_id}:{api_key[:8] if api_key else 'public'}"
        if key not in self._exchanges:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                "apiKey": api_key,
                "secret": secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
            self._exchanges[key] = exchange
        return self._exchanges[key]

    async def close_all(self):
        for exchange in self._exchanges.values():
            await exchange.close()
        self._exchanges.clear()

    async def execute_order(
        self,
        exchange_id: str,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "market",
        price: Optional[float] = None,
        api_key: str = "",
        secret: str = "",
        user_fee_override: Optional[float] = None,
    ) -> dict:
        """
        Execute order with pre-flight slippage check and fee collection.
        Target latency: <1 second for market orders.
        """
        start_time = time.monotonic()

        exchange = await self.get_exchange(exchange_id, api_key, secret)

        # Pre-flight: fetch orderbook for slippage estimate
        try:
            orderbook = await asyncio.wait_for(
                exchange.fetch_order_book(symbol, limit=20),
                timeout=0.5
            )
            ticker = await asyncio.wait_for(
                exchange.fetch_ticker(symbol),
                timeout=0.5
            )
            current_price = ticker["last"]

            order_size_usd = amount * current_price
            slippage_info = self.slippage_estimator.estimate(
                order_size_usd,
                orderbook["bids"],
                orderbook["asks"],
                side
            )

            # Reject if slippage > 2%
            if slippage_info["estimated_slippage"] > 0.02:
                return {
                    "status": "rejected",
                    "reason": "slippage_too_high",
                    "slippage": slippage_info,
                }

        except asyncio.TimeoutError:
            logger.warning(f"Orderbook fetch timeout for {symbol}, proceeding without slippage check")
            current_price = None
            slippage_info = {}

        # Execute the order
        try:
            if order_type == "market":
                order = await asyncio.wait_for(
                    exchange.create_market_order(symbol, side, amount),
                    timeout=3.0
                )
            elif order_type == "limit" and price:
                order = await asyncio.wait_for(
                    exchange.create_limit_order(symbol, side, amount, price),
                    timeout=3.0
                )
            else:
                raise ValueError(f"Invalid order_type={order_type}")

        except asyncio.TimeoutError:
            return {"status": "failed", "reason": "order_timeout"}
        except ccxt.InsufficientFunds as e:
            return {"status": "failed", "reason": "insufficient_funds", "detail": str(e)}
        except ccxt.InvalidOrder as e:
            return {"status": "failed", "reason": "invalid_order", "detail": str(e)}

        latency_ms = (time.monotonic() - start_time) * 1000

        # Calculate and record platform fee
        fill_price = order.get("average") or order.get("price") or current_price or 0
        fee_info = self.fee_collector.calculate_fee(amount, fill_price, user_fee_override)

        return {
            "status": "success",
            "order_id": order.get("id"),
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "fill_price": fill_price,
            "exchange": exchange_id,
            "latency_ms": round(latency_ms, 2),
            "slippage": slippage_info,
            "platform_fee": fee_info,
            "raw_order": order,
        }

    async def get_balances(self, exchange_id: str, api_key: str, secret: str) -> dict:
        exchange = await self.get_exchange(exchange_id, api_key, secret)
        balance = await exchange.fetch_balance()
        return {k: v for k, v in balance["total"].items() if v > 0}


# ─── Arbitrage Scanner ────────────────────────────────────────────────────────

class ArbitrageScanner:
    """
    Scans for price discrepancies between CEX exchanges.
    """

    def __init__(self):
        self.manager = ExchangeManager()

    async def scan(
        self,
        symbol: str,
        exchanges: list[str],
        min_profit_pct: float = 0.003,
    ) -> list[dict]:
        """Find arbitrage opportunities across exchanges."""

        async def fetch_price(ex_id: str):
            try:
                ex = await self.manager.get_exchange(ex_id)
                ticker = await asyncio.wait_for(ex.fetch_ticker(symbol), timeout=2.0)
                return {"exchange": ex_id, "bid": ticker["bid"], "ask": ticker["ask"]}
            except Exception as e:
                return {"exchange": ex_id, "error": str(e)}

        prices = await asyncio.gather(*[fetch_price(ex) for ex in exchanges])
        prices = [p for p in prices if "error" not in p]

        opportunities = []
        for i, buy_ex in enumerate(prices):
            for j, sell_ex in enumerate(prices):
                if i == j:
                    continue
                if not buy_ex.get("ask") or not sell_ex.get("bid"):
                    continue
                profit_pct = (sell_ex["bid"] - buy_ex["ask"]) / buy_ex["ask"]
                if profit_pct > min_profit_pct:
                    opportunities.append({
                        "symbol": symbol,
                        "buy_exchange": buy_ex["exchange"],
                        "sell_exchange": sell_ex["exchange"],
                        "buy_price": buy_ex["ask"],
                        "sell_price": sell_ex["bid"],
                        "profit_pct": round(profit_pct * 100, 4),
                        "estimated_profit_pct_after_fees": round((profit_pct - 0.002) * 100, 4),
                    })

        return sorted(opportunities, key=lambda x: x["profit_pct"], reverse=True)


# Global instances
exchange_manager = ExchangeManager()
arb_scanner = ArbitrageScanner()
