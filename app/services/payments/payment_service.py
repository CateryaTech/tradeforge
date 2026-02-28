"""
TradeForge Payment Service
- Stripe for fiat subscriptions (USD/EUR)
- NOWPayments for 350+ cryptocurrencies
- Fee collection and revenue tracking
- Subscription tier management
"""

import logging
import hmac
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Subscription pricing
SUBSCRIPTION_PRICES = {
    "basic": {"usd": 29.0, "features": "100 trades/day, 3 strategies, backtesting"},
    "premium": {"usd": 99.0, "features": "Unlimited trades, AI agents, DeFi automation"},
    "institutional": {"usd": 499.0, "features": "White-label, dedicated support, custom integrations"},
}


# ─── Stripe Service ───────────────────────────────────────────────────────────

class StripeService:
    """Handles fiat subscriptions via Stripe."""

    def __init__(self):
        # Import stripe lazily
        import stripe
        stripe.api_key = settings.STRIPE_SECRET_KEY
        self.stripe = stripe

    async def create_checkout_session(
        self,
        user_id: str,
        user_email: str,
        tier: str,
        success_url: str,
        cancel_url: str,
    ) -> dict:
        """Create a Stripe Checkout Session for subscription."""
        price_map = {
            "basic": settings.STRIPE_PRICE_BASIC,
            "premium": settings.STRIPE_PRICE_PREMIUM,
            "institutional": settings.STRIPE_PRICE_INSTITUTIONAL,
        }
        price_id = price_map.get(tier)
        if not price_id:
            raise ValueError(f"Unknown tier: {tier}")

        session = self.stripe.checkout.Session.create(
            customer_email=user_email,
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=cancel_url,
            metadata={"user_id": user_id, "tier": tier},
            subscription_data={
                "metadata": {"user_id": user_id, "tier": tier}
            }
        )
        return {
            "checkout_url": session.url,
            "session_id": session.id,
        }

    async def create_payment_intent(
        self,
        amount_usd: float,
        user_id: str,
        description: str = "TradeForge one-time payment",
    ) -> dict:
        """Create a one-time payment intent."""
        intent = self.stripe.PaymentIntent.create(
            amount=int(amount_usd * 100),  # cents
            currency="usd",
            metadata={"user_id": user_id},
            description=description,
        )
        return {
            "client_secret": intent.client_secret,
            "payment_intent_id": intent.id,
        }

    def verify_webhook(self, payload: bytes, sig_header: str) -> dict:
        """Verify and parse a Stripe webhook event."""
        event = self.stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
        return event

    def process_webhook_event(self, event: dict) -> Optional[dict]:
        """
        Process Stripe webhook events for subscription lifecycle.
        Returns action dict or None.
        """
        event_type = event["type"]
        data = event["data"]["object"]

        if event_type == "checkout.session.completed":
            metadata = data.get("metadata", {})
            return {
                "action": "activate_subscription",
                "user_id": metadata.get("user_id"),
                "tier": metadata.get("tier"),
                "stripe_subscription_id": data.get("subscription"),
                "expires_at": datetime.utcnow() + timedelta(days=30),
            }

        elif event_type == "customer.subscription.deleted":
            metadata = data.get("metadata", {})
            return {
                "action": "cancel_subscription",
                "user_id": metadata.get("user_id"),
                "tier": "free",
            }

        elif event_type == "invoice.payment_failed":
            metadata = data.get("subscription_details", {}).get("metadata", {})
            return {
                "action": "payment_failed",
                "user_id": metadata.get("user_id"),
            }

        return None

    async def create_invoice(
        self,
        customer_email: str,
        amount_usd: float,
        description: str,
    ) -> dict:
        """Create and finalize a Stripe invoice."""
        customer = self.stripe.Customer.create(email=customer_email)
        invoice_item = self.stripe.InvoiceItem.create(
            customer=customer.id,
            amount=int(amount_usd * 100),
            currency="usd",
            description=description,
        )
        invoice = self.stripe.Invoice.create(
            customer=customer.id,
            auto_advance=True,
        )
        finalized = self.stripe.Invoice.finalize_invoice(invoice.id)
        return {
            "invoice_id": finalized.id,
            "invoice_url": finalized.hosted_invoice_url,
            "pdf_url": finalized.invoice_pdf,
            "amount_due": finalized.amount_due / 100,
        }


# ─── NOWPayments Service (Crypto) ─────────────────────────────────────────────

class NOWPaymentsService:
    """
    Crypto payments supporting 350+ coins via NOWPayments API.
    """

    BASE_URL = "https://api.nowpayments.io/v1"

    def __init__(self):
        self.api_key = settings.NOWPAYMENTS_API_KEY
        self.ipn_secret = settings.NOWPAYMENTS_IPN_SECRET

    async def create_payment(
        self,
        amount_usd: float,
        pay_currency: str,
        user_id: str,
        tier: str,
        order_id: str,
        callback_url: str,
        success_url: str,
    ) -> dict:
        """
        Create a crypto payment. User can pay with any of 350+ coins.
        The amount is auto-converted from USD.
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/invoice",
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "price_amount": amount_usd,
                    "price_currency": "usd",
                    "pay_currency": pay_currency.lower(),
                    "order_id": order_id,
                    "order_description": f"TradeForge {tier} subscription",
                    "ipn_callback_url": callback_url,
                    "success_url": success_url,
                    "is_fee_paid_by_user": False,
                },
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()

        return {
            "payment_id": data.get("id"),
            "payment_url": data.get("invoice_url"),
            "pay_address": data.get("pay_address"),
            "pay_amount": data.get("pay_amount"),
            "pay_currency": pay_currency,
            "order_id": order_id,
            "status": data.get("payment_status"),
        }

    async def get_payment_status(self, payment_id: str) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/payment/{payment_id}",
                headers={"x-api-key": self.api_key},
                timeout=10.0,
            )
            response.raise_for_status()
        return response.json()

    async def get_available_currencies(self) -> list[str]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/currencies",
                headers={"x-api-key": self.api_key},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
        return data.get("currencies", [])

    def verify_ipn(self, payload: bytes, sig_header: str) -> bool:
        """Verify NOWPayments IPN callback signature."""
        expected = hmac.new(
            self.ipn_secret.encode(),
            payload,
            hashlib.sha512,
        ).hexdigest()
        return hmac.compare_digest(expected, sig_header)

    def process_ipn(self, data: dict) -> Optional[dict]:
        """Process IPN callback from NOWPayments."""
        status = data.get("payment_status")
        order_id = data.get("order_id", "")

        if status == "confirmed" or status == "finished":
            # Parse user_id and tier from order_id (format: {user_id}:{tier}:{uuid})
            parts = order_id.split(":")
            user_id = parts[0] if len(parts) > 0 else None
            tier = parts[1] if len(parts) > 1 else None

            return {
                "action": "activate_subscription",
                "user_id": user_id,
                "tier": tier,
                "payment_id": data.get("payment_id"),
                "tx_hash": data.get("outcome_amount"),
                "expires_at": datetime.utcnow() + timedelta(days=30),
            }

        elif status == "failed" or status == "expired":
            return {"action": "payment_failed", "payment_id": data.get("payment_id")}

        return None


# ─── Conversion Fee Tracker ───────────────────────────────────────────────────

class FeeTracker:
    """
    Tracks all platform fee revenue.
    Used for revenue reporting and affiliate payouts.
    """

    def calculate_trade_fee(self, amount: float, price: float, fee_pct: float = None) -> float:
        pct = fee_pct or settings.TRADE_FEE_PERCENT
        return round(amount * price * pct, 6)

    def calculate_conversion_fee(self, amount_usd: float) -> float:
        return round(amount_usd * settings.CONVERSION_FEE_PERCENT, 4)

    def calculate_apy_share(self, yield_earned_usd: float) -> float:
        return round(yield_earned_usd * settings.APY_SHARE_PERCENT, 4)

    def project_monthly_revenue(
        self,
        active_free: int = 0,
        active_basic: int = 0,
        active_premium: int = 0,
        active_institutional: int = 0,
        avg_trade_volume_usd: float = 500,
        avg_trades_per_user_day: float = 5,
        avg_defi_yield_monthly_usd: float = 50,
    ) -> dict:
        """Revenue projections based on user base."""
        days = 30
        subscription_revenue = (
            active_basic * SUBSCRIPTION_PRICES["basic"]["usd"] +
            active_premium * SUBSCRIPTION_PRICES["premium"]["usd"] +
            active_institutional * SUBSCRIPTION_PRICES["institutional"]["usd"]
        )

        total_users = active_free + active_basic + active_premium + active_institutional
        trade_fee_revenue = (
            total_users * avg_trades_per_user_day * days *
            avg_trade_volume_usd * settings.TRADE_FEE_PERCENT
        )

        premium_users = active_premium + active_institutional
        defi_yield_revenue = premium_users * avg_defi_yield_monthly_usd * settings.APY_SHARE_PERCENT

        total_mrr = subscription_revenue + trade_fee_revenue + defi_yield_revenue

        return {
            "subscription_mrr": round(subscription_revenue, 2),
            "trade_fee_mrr": round(trade_fee_revenue, 2),
            "defi_yield_mrr": round(defi_yield_revenue, 2),
            "total_mrr": round(total_mrr, 2),
            "projected_arr": round(total_mrr * 12, 2),
            "users": {"free": active_free, "basic": active_basic, "premium": active_premium, "institutional": active_institutional},
        }
