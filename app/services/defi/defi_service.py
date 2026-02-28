"""
TradeForge DeFi Service
- Uniswap V3 LP management with auto-rebalance
- Aave V3 lending/borrowing
- Multi-chain: Ethereum, Arbitrum, Base, Optimism
- Gas optimization
"""

import logging
import asyncio
from typing import Optional
from web3 import Web3
from web3.middleware import geth_poa_middleware
import json

from app.core.config import settings

logger = logging.getLogger(__name__)

# ─── Contract ABIs (minimal) ─────────────────────────────────────────────────

UNISWAP_V3_POOL_ABI = json.loads('''[
  {"inputs":[],"name":"slot0","outputs":[
    {"name":"sqrtPriceX96","type":"uint160"},
    {"name":"tick","type":"int24"},
    {"name":"observationIndex","type":"uint16"},
    {"name":"observationCardinality","type":"uint16"},
    {"name":"observationCardinalityNext","type":"uint16"},
    {"name":"feeProtocol","type":"uint8"},
    {"name":"unlocked","type":"bool"}
  ],"stateMutability":"view","type":"function"},
  {"inputs":[],"name":"liquidity","outputs":[{"name":"","type":"uint128"}],"stateMutability":"view","type":"function"},
  {"inputs":[],"name":"fee","outputs":[{"name":"","type":"uint24"}],"stateMutability":"view","type":"function"},
  {"inputs":[],"name":"token0","outputs":[{"name":"","type":"address"}],"stateMutability":"view","type":"function"},
  {"inputs":[],"name":"token1","outputs":[{"name":"","type":"address"}],"stateMutability":"view","type":"function"}
]''')

AAVE_POOL_ABI = json.loads('''[
  {"inputs":[{"name":"asset","type":"address"},{"name":"amount","type":"uint256"},{"name":"onBehalfOf","type":"address"},{"name":"referralCode","type":"uint16"}],"name":"supply","outputs":[],"stateMutability":"nonpayable","type":"function"},
  {"inputs":[{"name":"asset","type":"address"},{"name":"amount","type":"uint256"},{"name":"to","type":"address"}],"name":"withdraw","outputs":[{"name":"","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},
  {"inputs":[{"name":"asset","type":"address"}],"name":"getReserveData","outputs":[
    {"components":[{"name":"configuration","type":"uint256"},{"name":"liquidityIndex","type":"uint128"},{"name":"currentLiquidityRate","type":"uint128"},{"name":"variableBorrowIndex","type":"uint128"},{"name":"currentVariableBorrowRate","type":"uint128"},{"name":"currentStableBorrowRate","type":"uint128"},{"name":"lastUpdateTimestamp","type":"uint40"},{"name":"id","type":"uint16"},{"name":"aTokenAddress","type":"address"},{"name":"stableDebtTokenAddress","type":"address"},{"name":"variableDebtTokenAddress","type":"address"},{"name":"interestRateStrategyAddress","type":"address"},{"name":"accruedToTreasury","type":"uint128"},{"name":"unbacked","type":"uint128"},{"name":"isolationModeTotalDebt","type":"uint128"}],"name":"","type":"tuple"}],"stateMutability":"view","type":"function"}
]''')

# ─── Chain RPC Configuration ─────────────────────────────────────────────────

CHAIN_CONFIG = {
    "ethereum": {
        "rpc": settings.ETH_RPC_URL,
        "chain_id": 1,
        "poa": False,
        "aave_pool": "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2",
        "uniswap_factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "uniswap_nft_manager": "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
    },
    "arbitrum": {
        "rpc": settings.ARB_RPC_URL,
        "chain_id": 42161,
        "poa": False,
        "aave_pool": "0x794a61358D6845594F94dc1DB02A252b5b4814aD",
        "uniswap_factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "uniswap_nft_manager": "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
    },
    "base": {
        "rpc": settings.BASE_RPC_URL,
        "chain_id": 8453,
        "poa": False,
        "aave_pool": "0xA238Dd80C259a72e81d7e4664a9801593F98d1c5",
        "uniswap_factory": "0x33128a8fC17869897dcE68Ed026d694621f6FDfD",
        "uniswap_nft_manager": "0x03a520b32C04BF3bEEf7BEb72E919cf822Ed34f1",
    },
    "optimism": {
        "rpc": settings.OP_RPC_URL,
        "chain_id": 10,
        "poa": False,
        "aave_pool": "0x794a61358D6845594F94dc1DB02A252b5b4814aD",
        "uniswap_factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "uniswap_nft_manager": "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
    },
}


# ─── Web3 Connection Manager ──────────────────────────────────────────────────

class Web3Manager:
    def __init__(self):
        self._connections: dict[str, Web3] = {}

    def get_web3(self, chain: str) -> Web3:
        if chain not in self._connections:
            config = CHAIN_CONFIG.get(chain)
            if not config:
                raise ValueError(f"Unknown chain: {chain}")
            w3 = Web3(Web3.HTTPProvider(config["rpc"]))
            if config.get("poa"):
                w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            self._connections[chain] = w3
        return self._connections[chain]


web3_manager = Web3Manager()


# ─── Gas Optimizer ────────────────────────────────────────────────────────────

class GasOptimizer:
    """
    Smart gas estimation to avoid failed transactions and overpaying.
    """

    async def get_optimal_gas_price(self, chain: str) -> dict:
        w3 = web3_manager.get_web3(chain)

        try:
            # EIP-1559 gas pricing
            latest = w3.eth.get_block("latest")
            base_fee = latest.get("baseFeePerGas", 0)

            # Priority fee: use 10th percentile of recent blocks
            fee_history = w3.eth.fee_history(10, "latest", [10, 50, 90])
            priority_fees = [r[0] for r in fee_history.get("reward", [[0]]) if r]
            median_priority = sorted(priority_fees)[len(priority_fees) // 2] if priority_fees else 2_000_000_000

            max_fee = base_fee * 2 + median_priority

            return {
                "type": "eip1559",
                "base_fee_gwei": round(base_fee / 1e9, 2),
                "max_priority_fee_gwei": round(median_priority / 1e9, 2),
                "max_fee_per_gas_gwei": round(max_fee / 1e9, 2),
                "max_priority_fee_per_gas": median_priority,
                "max_fee_per_gas": max_fee,
            }
        except Exception as e:
            logger.warning(f"EIP-1559 failed for {chain}, falling back to legacy: {e}")
            gas_price = w3.eth.gas_price
            return {
                "type": "legacy",
                "gas_price_gwei": round(gas_price / 1e9, 2),
                "gas_price": gas_price,
            }

    async def estimate_gas_with_buffer(
        self,
        chain: str,
        transaction: dict,
        buffer_pct: float = 0.2,
    ) -> int:
        """Estimate gas with a safety buffer to prevent failed tx."""
        w3 = web3_manager.get_web3(chain)
        try:
            gas_estimate = w3.eth.estimate_gas(transaction)
            return int(gas_estimate * (1 + buffer_pct))
        except Exception as e:
            logger.error(f"Gas estimation failed: {e}")
            return 300_000  # Safe fallback


# ─── Uniswap V3 Manager ───────────────────────────────────────────────────────

class UniswapV3Manager:
    """
    Manages Uniswap V3 LP positions with auto-rebalancing.
    """

    def __init__(self):
        self.gas_optimizer = GasOptimizer()

    def get_pool_price(self, chain: str, pool_address: str) -> float:
        w3 = web3_manager.get_web3(chain)
        pool = w3.eth.contract(
            address=Web3.to_checksum_address(pool_address),
            abi=UNISWAP_V3_POOL_ABI
        )
        slot0 = pool.functions.slot0().call()
        sqrt_price_x96 = slot0[0]
        price = (sqrt_price_x96 / (2 ** 96)) ** 2
        return price

    def get_pool_apy_estimate(
        self,
        volume_24h_usd: float,
        tvl_usd: float,
        fee_tier: int,
    ) -> float:
        """
        Estimate LP APY from volume, TVL, and fee tier.
        fee_tier: 500 = 0.05%, 3000 = 0.3%, 10000 = 1%
        """
        if tvl_usd <= 0:
            return 0.0
        daily_fees = volume_24h_usd * (fee_tier / 1_000_000)
        daily_yield = daily_fees / tvl_usd
        return daily_yield * 365 * 100  # annualized %

    async def should_rebalance(
        self,
        chain: str,
        pool_address: str,
        price_lower: float,
        price_upper: float,
        rebalance_threshold: float = 0.85,
    ) -> dict:
        """
        Check if LP position needs rebalancing.
        Triggers when price is within threshold% of range boundary.
        """
        current_price = self.get_pool_price(chain, pool_address)
        range_width = price_upper - price_lower
        distance_to_lower = (current_price - price_lower) / range_width
        distance_to_upper = (price_upper - current_price) / range_width

        needs_rebalance = (
            distance_to_lower < (1 - rebalance_threshold) or
            distance_to_upper < (1 - rebalance_threshold) or
            current_price < price_lower or
            current_price > price_upper
        )

        return {
            "current_price": current_price,
            "price_lower": price_lower,
            "price_upper": price_upper,
            "is_in_range": price_lower <= current_price <= price_upper,
            "distance_to_lower_pct": round(distance_to_lower * 100, 2),
            "distance_to_upper_pct": round(distance_to_upper * 100, 2),
            "needs_rebalance": needs_rebalance,
        }


# ─── Aave V3 Manager ─────────────────────────────────────────────────────────

class AaveV3Manager:
    """
    Manages Aave V3 supply/borrow/withdraw operations.
    """

    def get_supply_apy(self, chain: str, asset_address: str) -> float:
        """Get current supply APY for an asset on Aave V3."""
        w3 = web3_manager.get_web3(chain)
        aave_address = CHAIN_CONFIG[chain]["aave_pool"]
        aave = w3.eth.contract(
            address=Web3.to_checksum_address(aave_address),
            abi=AAVE_POOL_ABI
        )
        reserve_data = aave.functions.getReserveData(
            Web3.to_checksum_address(asset_address)
        ).call()
        # currentLiquidityRate is in Ray (1e27), convert to APY
        liquidity_rate_ray = reserve_data[2]
        supply_apy = (liquidity_rate_ray / 1e27) * 100
        return round(supply_apy, 4)

    async def build_supply_tx(
        self,
        chain: str,
        asset_address: str,
        amount_wei: int,
        user_address: str,
    ) -> dict:
        """Build a supply transaction for Aave V3."""
        gas_optimizer = GasOptimizer()
        w3 = web3_manager.get_web3(chain)
        aave_address = CHAIN_CONFIG[chain]["aave_pool"]
        aave = w3.eth.contract(
            address=Web3.to_checksum_address(aave_address),
            abi=AAVE_POOL_ABI
        )

        tx_data = aave.functions.supply(
            Web3.to_checksum_address(asset_address),
            amount_wei,
            Web3.to_checksum_address(user_address),
            0  # referral code
        ).build_transaction({
            "from": Web3.to_checksum_address(user_address),
            "nonce": w3.eth.get_transaction_count(Web3.to_checksum_address(user_address)),
        })

        gas_params = await gas_optimizer.get_optimal_gas_price(chain)
        gas_limit = await gas_optimizer.estimate_gas_with_buffer(chain, tx_data)

        tx_data.update({"gas": gas_limit, **{k: v for k, v in gas_params.items() if k in ["max_fee_per_gas", "max_priority_fee_per_gas", "gas_price"]}})
        return tx_data


# ─── Yield Optimizer ─────────────────────────────────────────────────────────

class YieldOptimizer:
    """
    Scans multiple protocols/chains and suggests highest APY opportunities.
    """

    def __init__(self):
        self.aave = AaveV3Manager()
        self.uniswap = UniswapV3Manager()

    async def find_best_yield(
        self,
        asset: str,
        chains: list[str] = None,
    ) -> list[dict]:
        """
        Compare yield opportunities across chains and protocols.
        Returns sorted list by APY.
        """
        if chains is None:
            chains = ["ethereum", "arbitrum", "base"]

        # Known USDC addresses per chain
        usdc_addresses = {
            "ethereum": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            "arbitrum": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
            "base": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        }

        opportunities = []

        for chain in chains:
            try:
                if asset.upper() == "USDC" and chain in usdc_addresses:
                    apy = self.aave.get_supply_apy(chain, usdc_addresses[chain])
                    opportunities.append({
                        "protocol": "aave_v3",
                        "chain": chain,
                        "asset": asset,
                        "apy": apy,
                        "type": "lending",
                        "risk": "low",
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch APY for {chain}: {e}")

        return sorted(opportunities, key=lambda x: x["apy"], reverse=True)
