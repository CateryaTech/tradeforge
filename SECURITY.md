# Security Policy — TradeForge AaaS

## Reporting Vulnerabilities

Email: cateryatech@proton.me with subject "SECURITY: TradeForge"
Response time: 48 hours. Please do not open public GitHub issues for vulnerabilities.

## Security Measures

### Authentication
- JWT tokens (HS256), 24h expiry
- Bcrypt password hashing (12 rounds)
- OAuth2 Bearer scheme

### Infrastructure
- Non-root Docker containers
- All secrets via environment variables (never hardcoded)
- TLS required in production (terminate at nginx/load balancer)

### Payments
- Stripe webhook signature verification (HMAC-SHA256)
- NOWPayments IPN signature verification (HMAC-SHA512)

### Blockchain
- Gas estimation with 30% safety buffer
- Chainalysis AML screening on all on-chain transactions
- Hot wallet private keys: use AWS KMS or HashiCorp Vault in production

### CI/CD
- Bandit static security analysis on every push
- TruffleHog secret scanning on every push
- Dependency updates via Dependabot

## Known Limitations

- Rate limiting is basic (Redis-backed production rate limiting recommended)
- The PRIVATE_KEY env var is a hot wallet — use HSM for production mainnet operations
- Streamlit UI (if used) should be behind authentication proxy

## Responsible Disclosure

We follow coordinated vulnerability disclosure. Security researchers who responsibly disclose vulnerabilities will be credited.
