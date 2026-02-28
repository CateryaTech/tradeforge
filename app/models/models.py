"""
TradeForge Database Models
"""

import uuid
from datetime import datetime
from enum import Enum
from sqlalchemy import (
    Column, String, Float, Boolean, DateTime, Integer,
    ForeignKey, Text, JSON, Enum as SAEnum
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, DeclarativeBase
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class SubscriptionTier(str, Enum):
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    INSTITUTIONAL = "institutional"


class TradeStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class PaymentStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    REFUNDED = "refunded"


class PaymentMethod(str, Enum):
    STRIPE = "stripe"
    CRYPTO = "crypto"
    NOWPAYMENTS = "nowpayments"


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    subscription_tier = Column(SAEnum(SubscriptionTier), default=SubscriptionTier.FREE)
    subscription_expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    kyc_status = Column(String(50), default="pending")
    kyc_applicant_id = Column(String(255))  # Sumsub applicant ID
    api_key = Column(String(64), unique=True, index=True)
    api_key_created_at = Column(DateTime)
    trade_fee_override = Column(Float)  # Custom fee for enterprise
    affiliate_code = Column(String(32), unique=True)
    referred_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    white_label_config = Column(JSON)  # For institutional white-label
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    strategies = relationship("Strategy", back_populates="user")
    trades = relationship("Trade", back_populates="user")
    payments = relationship("Payment", back_populates="user")
    portfolios = relationship("Portfolio", back_populates="user")


class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    strategy_type = Column(String(50))  # trend, arbitrage, market_making, yield_farming
    config = Column(JSON)  # Strategy parameters
    is_active = Column(Boolean, default=False)
    is_public = Column(Boolean, default=False)  # Marketplace
    performance_fee = Column(Float, default=0.10)  # 10% for public strategies
    total_return = Column(Float, default=0.0)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    total_trades = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="strategies")
    trades = relationship("Trade", back_populates="strategy")
    backtests = relationship("Backtest", back_populates="strategy")


class Trade(Base):
    __tablename__ = "trades"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("strategies.id"))
    exchange = Column(String(50))  # binance, uniswap, aave
    symbol = Column(String(50))
    trade_type = Column(String(20))  # buy, sell, swap, lp_add, lp_remove
    amount = Column(Float)
    price = Column(Float)
    fee = Column(Float)  # Platform fee charged
    fee_currency = Column(String(20))
    gas_used = Column(Float)  # For on-chain trades
    tx_hash = Column(String(100))
    status = Column(SAEnum(TradeStatus), default=TradeStatus.PENDING)
    slippage = Column(Float)
    pnl = Column(Float)
    chain = Column(String(50), default="ethereum")
    raw_response = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    closed_at = Column(DateTime)

    user = relationship("User", back_populates="trades")
    strategy = relationship("Strategy", back_populates="trades")


class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), default="Default")
    total_value_usd = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    allocations = Column(JSON)  # {asset: percentage}
    rebalance_threshold = Column(Float, default=0.05)  # 5% drift triggers rebalance
    last_rebalanced_at = Column(DateTime)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="portfolios")


class Backtest(Base):
    __tablename__ = "backtests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("strategies.id"))
    symbol = Column(String(50))
    timeframe = Column(String(20))
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    initial_capital = Column(Float)
    final_capital = Column(Float)
    total_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    total_trades = Column(Integer)
    regime = Column(String(20))  # bull, bear, sideways
    results = Column(JSON)
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)

    strategy = relationship("Strategy", back_populates="backtests")


class Payment(Base):
    __tablename__ = "payments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    method = Column(SAEnum(PaymentMethod))
    amount_usd = Column(Float)
    amount_crypto = Column(Float)
    crypto_currency = Column(String(20))
    subscription_tier = Column(SAEnum(SubscriptionTier))
    status = Column(SAEnum(PaymentStatus), default=PaymentStatus.PENDING)
    stripe_payment_intent = Column(String(255))
    stripe_subscription_id = Column(String(255))
    nowpayments_payment_id = Column(String(255))
    tx_hash = Column(String(100))
    invoice_url = Column(String(500))
    created_at = Column(DateTime, default=func.now())
    confirmed_at = Column(DateTime)

    user = relationship("User", back_populates="payments")


class DeFiPosition(Base):
    __tablename__ = "defi_positions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    protocol = Column(String(50))  # uniswap_v3, aave_v3, compound
    chain = Column(String(50))
    position_type = Column(String(50))  # lp, lending, borrowing, staking
    token0 = Column(String(50))
    token1 = Column(String(50))
    amount0 = Column(Float)
    amount1 = Column(Float)
    liquidity = Column(Float)
    current_apy = Column(Float)
    fees_earned = Column(Float)
    token_id = Column(Integer)  # Uniswap V3 NFT token ID
    price_lower = Column(Float)
    price_upper = Column(Float)
    is_in_range = Column(Boolean, default=True)
    tx_hash = Column(String(100))
    chain_address = Column(String(100))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
