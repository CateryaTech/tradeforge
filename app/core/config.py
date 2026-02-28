"""
TradeForge Configuration - All settings via environment variables
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_NAME: str = "TradeForge AaaS"
    APP_ENV: str = "production"
    SECRET_KEY: str = "CHANGE_THIS_IN_PRODUCTION_USE_STRONG_SECRET"
    DEBUG: bool = False
    ALLOWED_ORIGINS: List[str] = ["https://app.tradeforge.io", "http://localhost:3000"]
    ALLOWED_HOSTS: List[str] = ["*"]

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://tradeforge:password@postgres:5432/tradeforge"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 40

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"

    # JWT
    JWT_SECRET_KEY: str = "CHANGE_THIS_JWT_SECRET"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # Subscription Tiers
    FREE_TIER_DAILY_TRADES: int = 10
    BASIC_TIER_DAILY_TRADES: int = 100
    PREMIUM_TIER_DAILY_TRADES: int = -1  # unlimited

    # Stripe (Fiat Payments)
    STRIPE_SECRET_KEY: str = ""
    STRIPE_WEBHOOK_SECRET: str = ""
    STRIPE_PRICE_BASIC: str = ""   # $29/mo
    STRIPE_PRICE_PREMIUM: str = "" # $99/mo
    STRIPE_PRICE_INSTITUTIONAL: str = "" # $499/mo

    # NOWPayments (Crypto Payments)
    NOWPAYMENTS_API_KEY: str = ""
    NOWPAYMENTS_IPN_SECRET: str = ""

    # AI/LLM Keys (set via env, NOT hardcoded)
    GROQ_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    TOGETHER_API_KEY: str = ""
    FIREWORKS_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://ollama:11434"

    # CEX APIs
    BINANCE_API_KEY: str = ""
    BINANCE_SECRET: str = ""
    COINBASE_API_KEY: str = ""
    COINBASE_SECRET: str = ""

    # Blockchain
    ETH_RPC_URL: str = "https://mainnet.infura.io/v3/YOUR_INFURA_KEY"
    ARB_RPC_URL: str = "https://arb1.arbitrum.io/rpc"
    BASE_RPC_URL: str = "https://mainnet.base.org"
    OP_RPC_URL: str = "https://mainnet.optimism.io"

    # KYC/AML
    SUMSUB_APP_TOKEN: str = ""
    SUMSUB_SECRET_KEY: str = ""

    # Monitoring
    PROMETHEUS_ENABLED: bool = True

    # Fee Settings
    TRADE_FEE_PERCENT: float = 0.002   # 0.2% per trade
    APY_SHARE_PERCENT: float = 0.10    # 10% of yield earned
    CONVERSION_FEE_PERCENT: float = 0.002  # 0.2% fiat-crypto conversion

    # Notifications
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASS: str = ""
    TELEGRAM_BOT_TOKEN: str = ""
    DISCORD_WEBHOOK_URL: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
