"""
TradeForge AaaS - Algorithmic Trading as a Service
Author: Ary HH @ CATERYA Tech
Contact: aryhharyanto@proton.me | cateryatech@proton.me
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time

from app.core.config import settings
from app.core.database import init_db
from app.core.redis_client import init_redis
from app.api.routes import (
    auth, users, strategies, backtesting,
    live_trading, defi, payments, notifications, analytics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting TradeForge AaaS...")
    await init_db()
    await init_redis()
    logger.info("TradeForge started successfully.")
    yield
    # Shutdown
    logger.info("Shutting down TradeForge...")


app = FastAPI(
    title="TradeForge AaaS",
    description="Multi-tenant Algorithmic Trading as a Service Platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Register routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(strategies.router, prefix="/api/v1/strategies", tags=["Strategies"])
app.include_router(backtesting.router, prefix="/api/v1/backtest", tags=["Backtesting"])
app.include_router(live_trading.router, prefix="/api/v1/trading", tags=["Live Trading"])
app.include_router(defi.router, prefix="/api/v1/defi", tags=["DeFi"])
app.include_router(payments.router, prefix="/api/v1/payments", tags=["Payments"])
app.include_router(notifications.router, prefix="/api/v1/notifications", tags=["Notifications"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0", "platform": "TradeForge AaaS"}
