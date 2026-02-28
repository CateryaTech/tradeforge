"""
TradeForge Payment API Routes
- POST /subscribe/stripe - Create Stripe checkout
- POST /subscribe/crypto - Create crypto payment
- POST /webhook/stripe - Stripe webhook handler
- POST /webhook/nowpayments - NOWPayments IPN handler
- GET /currencies - Available crypto currencies
"""

import logging
import uuid
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from app.services.payments.payment_service import (
    StripeService, NOWPaymentsService, FeeTracker, SUBSCRIPTION_PRICES
)
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

stripe_service = StripeService() if settings.STRIPE_SECRET_KEY else None
nowpayments_service = NOWPaymentsService() if settings.NOWPAYMENTS_API_KEY else None
fee_tracker = FeeTracker()


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class StripeSubscribeRequest(BaseModel):
    tier: str  # basic, premium, institutional
    success_url: str = "https://app.tradeforge.io/dashboard?payment=success"
    cancel_url: str = "https://app.tradeforge.io/pricing"


class CryptoSubscribeRequest(BaseModel):
    tier: str
    pay_currency: str = "ETH"


class RevenueProjectionRequest(BaseModel):
    active_free: int = 100
    active_basic: int = 50
    active_premium: int = 20
    active_institutional: int = 5
    avg_trade_volume_usd: float = 500
    avg_trades_per_user_day: float = 5
    avg_defi_yield_monthly_usd: float = 50


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/pricing")
async def get_pricing():
    """Get subscription tier pricing."""
    return {
        "tiers": SUBSCRIPTION_PRICES,
        "fees": {
            "trade_fee": f"{settings.TRADE_FEE_PERCENT * 100}% per trade",
            "defi_yield_share": f"{settings.APY_SHARE_PERCENT * 100}% of earned yield",
            "conversion_fee": f"{settings.CONVERSION_FEE_PERCENT * 100}% fiat-crypto conversion",
        }
    }


@router.post("/subscribe/stripe")
async def subscribe_stripe(
    request: StripeSubscribeRequest,
    x_user_id: str = Header(...),
    x_user_email: str = Header(...),
):
    """Create a Stripe checkout session for subscription."""
    if not stripe_service:
        raise HTTPException(503, "Stripe payments not configured")

    if request.tier not in SUBSCRIPTION_PRICES:
        raise HTTPException(400, f"Invalid tier. Choose: {list(SUBSCRIPTION_PRICES.keys())}")

    try:
        session = await stripe_service.create_checkout_session(
            user_id=x_user_id,
            user_email=x_user_email,
            tier=request.tier,
            success_url=request.success_url,
            cancel_url=request.cancel_url,
        )
        return session
    except Exception as e:
        logger.error(f"Stripe checkout error: {e}")
        raise HTTPException(500, "Failed to create checkout session")


@router.post("/subscribe/crypto")
async def subscribe_crypto(
    request: CryptoSubscribeRequest,
    x_user_id: str = Header(...),
):
    """Create a crypto payment for subscription."""
    if not nowpayments_service:
        raise HTTPException(503, "Crypto payments not configured")

    if request.tier not in SUBSCRIPTION_PRICES:
        raise HTTPException(400, f"Invalid tier")

    amount_usd = SUBSCRIPTION_PRICES[request.tier]["usd"]
    order_id = f"{x_user_id}:{request.tier}:{uuid.uuid4().hex[:8]}"

    try:
        payment = await nowpayments_service.create_payment(
            amount_usd=amount_usd,
            pay_currency=request.pay_currency,
            user_id=x_user_id,
            tier=request.tier,
            order_id=order_id,
            callback_url="https://api.tradeforge.io/api/v1/payments/webhook/nowpayments",
            success_url="https://app.tradeforge.io/dashboard?payment=success",
        )
        return payment
    except Exception as e:
        logger.error(f"NOWPayments error: {e}")
        raise HTTPException(500, "Failed to create crypto payment")


@router.get("/subscribe/crypto/currencies")
async def get_crypto_currencies():
    """List all supported crypto payment currencies."""
    if not nowpayments_service:
        raise HTTPException(503, "Crypto payments not configured")
    currencies = await nowpayments_service.get_available_currencies()
    return {"currencies": currencies, "count": len(currencies)}


@router.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events."""
    if not stripe_service:
        raise HTTPException(503, "Stripe not configured")

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if not sig_header:
        raise HTTPException(400, "Missing stripe-signature header")

    try:
        event = stripe_service.verify_webhook(payload, sig_header)
    except Exception as e:
        logger.error(f"Stripe webhook verification failed: {e}")
        raise HTTPException(400, "Invalid webhook signature")

    action = stripe_service.process_webhook_event(event)

    if action:
        logger.info(f"Stripe webhook action: {action}")
        # TODO: Update user subscription in database
        # await user_service.update_subscription(action["user_id"], action["tier"])

    return {"received": True}


@router.post("/webhook/nowpayments")
async def nowpayments_webhook(
    request: Request,
    x_nowpayments_sig: Optional[str] = Header(None),
):
    """Handle NOWPayments IPN callbacks."""
    if not nowpayments_service:
        raise HTTPException(503, "NOWPayments not configured")

    payload = await request.body()

    if x_nowpayments_sig:
        if not nowpayments_service.verify_ipn(payload, x_nowpayments_sig):
            raise HTTPException(400, "Invalid IPN signature")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON payload")

    action = nowpayments_service.process_ipn(data)

    if action:
        logger.info(f"NOWPayments IPN action: {action}")
        # TODO: Update user subscription in database

    return {"received": True}


@router.post("/project-revenue")
async def project_revenue(request: RevenueProjectionRequest):
    """Calculate revenue projections based on user base."""
    projection = fee_tracker.project_monthly_revenue(
        active_free=request.active_free,
        active_basic=request.active_basic,
        active_premium=request.active_premium,
        active_institutional=request.active_institutional,
        avg_trade_volume_usd=request.avg_trade_volume_usd,
        avg_trades_per_user_day=request.avg_trades_per_user_day,
        avg_defi_yield_monthly_usd=request.avg_defi_yield_monthly_usd,
    )
    return projection
