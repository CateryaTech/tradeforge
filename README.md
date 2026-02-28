# TradeForge AaaS v1.0

**Multi-Tenant Algorithmic Trading as a Service Platform**

> Author: Ary HH @ CATERYA Tech  
> Contact: aryhharyanto@proton.me | cateryatech@proton.me  
> License: MIT  
> Status: ✅ Production-Ready v1.0

[![CI/CD](https://github.com/aryhhary/TradeForge-AaaS/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/aryhhary/TradeForge-AaaS/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What Is TradeForge?

TradeForge is a **production-grade algorithmic trading platform** that combines:

- **CEX trading** via CCXT (Binance, Coinbase, Kraken, 100+ exchanges)
- **DeFi automation** via Uniswap V3, Aave V3 across Ethereum, Arbitrum, Base, Optimism
- **AI-driven strategies** using LangChain + Groq/OpenRouter LLMs
- **Parallel backtesting** with vectorbt + multiprocessing
- **Hybrid payments** — Stripe (fiat) + NOWPayments (350+ crypto coins)
- **Multi-tenant SaaS** with subscription tiers and fee-sharing revenue model

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Nginx (SSL Termination)            │
├──────────────────────┬──────────────────────────────┤
│   FastAPI (API)      │   Streamlit (Dashboard)       │
├──────────────────────┴──────────────────────────────┤
│              Redis (Cache + Queue)                    │
├──────────────────────┬──────────────────────────────┤
│   PostgreSQL (DB)    │   Celery (Background Tasks)  │
├──────────────────────┴──────────────────────────────┤
│         Prometheus + Grafana (Monitoring)            │
└─────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/aryhhary/TradeForge-AaaS.git
cd TradeForge-AaaS
cp .env.example .env
# Edit .env with your API keys
```

### 2. Generate secrets

```bash
openssl rand -hex 32  # Use for SECRET_KEY
openssl rand -hex 32  # Use for JWT_SECRET_KEY
```

### 3. Launch with Docker Compose

```bash
docker compose up -d
```

### 4. Apply database migrations

```bash
docker compose exec api alembic upgrade head
```

### 5. Access services

| Service | URL |
|---|---|
| API Docs | http://localhost:8000/api/docs |
| Dashboard | http://localhost:8501 |
| Grafana | http://localhost:3001 |
| Prometheus | http://localhost:9090 |

---

## Subscription Tiers

| Tier | Price | Trades/Day | Features |
|------|-------|------------|----------|
| **Free** | $0 | 10 | Basic backtesting, 1 strategy |
| **Basic** | $29/mo | 100 | 3 strategies, live trading, CEX |
| **Premium** | $99/mo | Unlimited | AI agents, DeFi automation, all chains |
| **Institutional** | $499/mo | Unlimited | White-label, dedicated support, custom |

---

## Revenue Model

TradeForge generates revenue through 3 streams:

1. **Subscriptions** — $29–$499/month per user
2. **Trade fees** — 0.2% per executed trade
3. **DeFi yield share** — 10% of APY earned via automated yield farming

### Revenue Projection Example

| User Base | MRR |
|---|---|
| 100 free + 50 basic + 20 premium + 5 institutional | ~$6,500 |
| 500 free + 200 basic + 80 premium + 20 institutional | ~$26,000 |
| 2,000 free + 1,000 basic + 400 premium + 100 institutional | ~$130,000 |

Use the `/api/v1/payments/project-revenue` endpoint for custom projections.

---

## Key Features

### Backtesting Engine

```python
from app.services.trading.backtesting import BacktestEngine

engine = BacktestEngine()

# Single backtest
result = await engine.run_backtest({
    "symbol": "BTC-USD",
    "strategy_type": "sma_crossover",
    "config": {"fast_period": 20, "slow_period": 50},
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "initial_capital": 10000,
})

# Parameter sweep (parallel, all CPU cores)
best_params = await engine.parameter_sweep(
    symbol="BTC-USD",
    strategy_type="sma_crossover",
    param_grid={"fast_period": [10, 20, 30], "slow_period": [50, 100, 200]},
    start_date="2023-01-01",
    end_date="2024-01-01",
)
```

### Live Trading with Slippage Check

```python
from app.services.trading.live_trading import exchange_manager

result = await exchange_manager.execute_order(
    exchange_id="binance",
    symbol="BTC/USDT",
    side="buy",
    amount=0.01,
    order_type="market",
    api_key="USER_BINANCE_KEY",
    secret="USER_BINANCE_SECRET",
)
# Returns: {status, fill_price, latency_ms, slippage, platform_fee}
```

### Arbitrage Scanner

```python
from app.services.trading.live_trading import arb_scanner

opportunities = await arb_scanner.scan(
    symbol="BTC/USDT",
    exchanges=["binance", "kraken", "coinbase"],
    min_profit_pct=0.003,
)
```

### DeFi Yield Optimization

```python
from app.services.defi.defi_service import YieldOptimizer

optimizer = YieldOptimizer()
best = await optimizer.find_best_yield("USDC", chains=["ethereum", "arbitrum", "base"])
# Returns: [{protocol, chain, apy, type, risk}, ...] sorted by APY
```

### AI Strategy Generation

```python
from app.services.ai.ai_agents import strategy_agent

strategy = await strategy_agent.generate_strategy(
    market_data_summary={"symbol": "ETH", "price": 3200, "regime": "bull"},
    user_preferences={"risk_tolerance": "medium", "capital": 5000},
)
```

### Risk Management

```python
from app.services.trading.backtesting import RiskSimulator
import numpy as np

sim = RiskSimulator()
returns = np.array(your_historical_returns)

var_result = sim.monte_carlo_var(returns, confidence=0.95, simulations=10000)
# {var_95: -0.032, cvar_95: -0.048, worst_case: ..., best_case: ...}
```

---

## Payment Integration

### Accept Fiat (Stripe)

```bash
POST /api/v1/payments/subscribe/stripe
Headers: x-user-id: uuid, x-user-email: email@example.com
Body: {"tier": "premium", "success_url": "...", "cancel_url": "..."}
```

### Accept Crypto (350+ coins via NOWPayments)

```bash
POST /api/v1/payments/subscribe/crypto
Headers: x-user-id: uuid
Body: {"tier": "premium", "pay_currency": "ETH"}
# Returns: {payment_url, pay_address, pay_amount}
```

---

## API Endpoints Summary

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/auth/*` | POST | Register, login, refresh |
| `/api/v1/strategies/` | GET/POST | CRUD strategies |
| `/api/v1/backtest/` | POST | Run backtest |
| `/api/v1/backtest/sweep` | POST | Parameter sweep |
| `/api/v1/trading/order` | POST | Execute live order |
| `/api/v1/trading/arbitrage` | GET | Scan arbitrage |
| `/api/v1/defi/positions` | GET/POST | Manage DeFi positions |
| `/api/v1/defi/yield` | GET | Find best yield |
| `/api/v1/payments/pricing` | GET | Tier pricing |
| `/api/v1/payments/subscribe/stripe` | POST | Stripe checkout |
| `/api/v1/payments/subscribe/crypto` | POST | Crypto payment |
| `/api/v1/payments/project-revenue` | POST | Revenue projection |
| `/api/v1/analytics/` | GET | Portfolio analytics |

---

## Security

- JWT authentication with refresh tokens
- API key authentication for programmatic access
- All secrets via environment variables (never hardcoded)
- Rate limiting via Redis
- Input validation via Pydantic
- Non-root Docker containers
- Stripe webhook signature verification
- NOWPayments IPN signature verification
- Bandit security scanning in CI

See [SECURITY.md](SECURITY.md) for vulnerability reporting.

---

## Supported Chains

| Chain | DeFi Protocols |
|---|---|
| Ethereum | Uniswap V3, Aave V3 |
| Arbitrum | Uniswap V3, Aave V3 |
| Base | Uniswap V3, Aave V3 |
| Optimism | Uniswap V3, Aave V3 |

---

## Running Tests

```bash
pip install pytest pytest-asyncio pytest-cov
pytest tests/ -v --asyncio-mode=auto --cov=app
```

---

## Deployment (Production)

Full deployment guide in `infrastructure/` folder.

For Kubernetes: `kubectl apply -f infrastructure/k8s/`

For AWS/GCP with auto-scaling: see `infrastructure/terraform/`

---

## Compliance

- **KYC/AML**: Sumsub integration for identity verification
- **EU MiCA**: Configurable for EU regulatory requirements
- **Data Privacy**: User data isolated per tenant
- **Audit Logs**: All trades and payments logged with full audit trail

---

## Disclaimer

TradeForge is a software platform, not a financial advisor. Algorithmic trading involves substantial risk of loss. Past performance does not guarantee future results. Users are responsible for compliance with local laws and regulations. Use at your own risk.

---

## Contributing

Pull requests welcome. Please run tests and security scan before submitting.

```bash
pytest tests/ && bandit -r app/
```
