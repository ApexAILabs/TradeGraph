"""Agent that creates allocation plans across stocks, ETFs, and crypto for various horizons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

from .base_agent import BaseAgent


@dataclass
class AllocationSuggestion:
    asset_class: str
    weight: float
    rationale: str
    sample_assets: List[Dict[str, str]]


HORIZON_LABELS = {
    "1w": "1-Week",
    "1m": "1-Month",
    "1y": "1-Year",
}


STRATEGY_LIBRARY: Dict[str, Dict[str, Dict[str, AllocationSuggestion]]] = {}


def _build_strategy_library() -> Dict[str, Dict[str, Dict[str, AllocationSuggestion]]]:
    asset_pool = {
        "stocks": [
            {"symbol": "AAPL", "thesis": "Cash-rich mega-cap"},
            {"symbol": "MSFT", "thesis": "Enterprise AI exposure"},
            {"symbol": "NVDA", "thesis": "GPU leadership"},
            {"symbol": "AMZN", "thesis": "Cloud + retail"},
        ],
        "etfs": [
            {"symbol": "VOO", "thesis": "S&P 500 core"},
            {"symbol": "QQQ", "thesis": "Large-cap growth"},
            {"symbol": "ARKK", "thesis": "High beta innovation"},
            {"symbol": "TLT", "thesis": "Long-duration bonds"},
        ],
        "crypto": [
            {"symbol": "BTC", "thesis": "Digital gold"},
            {"symbol": "ETH", "thesis": "Smart contracts"},
            {"symbol": "SOL", "thesis": "High throughput L1"},
        ],
    }

    def suggestion(asset_class: str, weight: float, rationale: str) -> AllocationSuggestion:
        return AllocationSuggestion(
            asset_class=asset_class,
            weight=weight,
            rationale=rationale,
            sample_assets=asset_pool[asset_class][:2],
        )

    strategies = {
        "growth": {
            "1w": {
                "stocks": suggestion(
                    "stocks", 0.35, "Stay liquid in quality mega caps while watching catalysts."
                ),
                "etfs": suggestion(
                    "etfs", 0.15, "Use QQQ/ARKK for beta exposure without security selection."
                ),
                "crypto": suggestion(
                    "crypto", 0.50, "Lean into BTC/ETH momentum for tactical upside."
                ),
            },
            "1m": {
                "stocks": suggestion(
                    "stocks", 0.45, "Compound AI and cloud tailwinds via mega caps."
                ),
                "etfs": suggestion(
                    "etfs", 0.20, "Blend sector ETFs for diversification."
                ),
                "crypto": suggestion(
                    "crypto", 0.35, "Maintain crypto beta for asymmetric upside."
                ),
            },
            "1y": {
                "stocks": suggestion(
                    "stocks", 0.5, "Core equity growth allocation with reinvestment."
                ),
                "etfs": suggestion(
                    "etfs", 0.25, "Add thematic ETFs to capture innovation baskets."
                ),
                "crypto": suggestion(
                    "crypto", 0.25, "Long-term conviction in BTC/ETH network effects."
                ),
            },
        },
        "balanced": {
            "1w": {
                "stocks": suggestion(
                    "stocks", 0.3, "Blend defensives with growth to dampen volatility."
                ),
                "etfs": suggestion(
                    "etfs", 0.4, "VOO/TLT core to keep drawdowns manageable."
                ),
                "crypto": suggestion(
                    "crypto", 0.3, "Measured crypto sleeve for opportunistic moves."
                ),
            },
            "1m": {
                "stocks": suggestion(
                    "stocks", 0.4, "Add cyclicals selectively as macro visibility improves."
                ),
                "etfs": suggestion(
                    "etfs", 0.4, "Core passive ETFs to anchor risk."
                ),
                "crypto": suggestion(
                    "crypto", 0.2, "Keep crypto beta but size for volatility."
                ),
            },
            "1y": {
                "stocks": suggestion(
                    "stocks", 0.45, "Dividend growers + quality compounders."
                ),
                "etfs": suggestion(
                    "etfs", 0.4, "Broad equity and bond ETFs for balance."
                ),
                "crypto": suggestion(
                    "crypto", 0.15, "Smaller crypto sleeve for optionality."
                ),
            },
        },
        "defensive": {
            "1w": {
                "stocks": suggestion(
                    "stocks", 0.25, "Prefer healthcare and staples for stability."
                ),
                "etfs": suggestion(
                    "etfs", 0.6, "High-quality bond and minimum-vol ETFs."
                ),
                "crypto": suggestion(
                    "crypto", 0.15, "Tiny crypto exposure to stay engaged."
                ),
            },
            "1m": {
                "stocks": suggestion(
                    "stocks", 0.3, "Income-oriented equities."
                ),
                "etfs": suggestion(
                    "etfs", 0.55, "Blend IG bonds with broad equity ETFs."
                ),
                "crypto": suggestion(
                    "crypto", 0.15, "Cap risk but allow for upside."
                ),
            },
            "1y": {
                "stocks": suggestion(
                    "stocks", 0.35, "Quality and value tilt."
                ),
                "etfs": suggestion(
                    "etfs", 0.5, "VOO/TLT core plus dividend ETFs."
                ),
                "crypto": suggestion(
                    "crypto", 0.15, "Long-dated call option sized exposure."
                ),
            },
        },
        "income": {
            "1w": {
                "stocks": suggestion(
                    "stocks", 0.35, "Dividend aristocrats for short-term distributions."
                ),
                "etfs": suggestion(
                    "etfs", 0.55, "Covered-call and bond ETFs for yield."
                ),
                "crypto": suggestion(
                    "crypto", 0.10, "Stablecoin yield or staking."
                ),
            },
            "1m": {
                "stocks": suggestion(
                    "stocks", 0.4, "REITs + utilities blend."
                ),
                "etfs": suggestion(
                    "etfs", 0.5, "Bond ladders and dividend ETFs."
                ),
                "crypto": suggestion(
                    "crypto", 0.1, "Select staking strategies."
                ),
            },
            "1y": {
                "stocks": suggestion(
                    "stocks", 0.45, "Global dividend growth."
                ),
                "etfs": suggestion(
                    "etfs", 0.45, "Income ETFs and bond funds."
                ),
                "crypto": suggestion(
                    "crypto", 0.1, "Yield-focused crypto vehicles."
                ),
            },
        },
    }
    return strategies


STRATEGY_LIBRARY = _build_strategy_library()


class MultiAssetAllocationAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="MultiAssetAllocationAgent",
            description="Builds allocations across stocks, ETFs, and crypto for multiple horizons",
            **kwargs,
        )

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        budget = float(input_data.get("budget", 0))
        if budget <= 0:
            raise ValueError("Budget must be greater than zero.")
        strategies = input_data.get("strategies") or ["balanced"]
        normalized = self._normalize_strategies(strategies)
        logger.info(
            "Running multi-asset allocation for budget %.2f with strategies %s",
            budget,
            normalized,
        )

        plans = [self._build_plan(strategy, budget) for strategy in normalized]
        advisory_notes = [
            "Allocations are illustrative; rebalance as macro drivers evolve.",
            "Size crypto sleeves according to volatility tolerance and access to custody.",
            "ETFs offer quick diversification for both beta and fixed-income exposures.",
        ]

        return {
            "budget": budget,
            "strategies": plans,
            "notes": advisory_notes,
        }

    def _normalize_strategies(self, strategies: List[str]) -> List[str]:
        valid = []
        for strategy in strategies:
            key = strategy.lower().strip()
            if key in STRATEGY_LIBRARY:
                valid.append(key)
        if not valid:
            valid = ["balanced"]
        return valid

    def _build_plan(self, strategy: str, budget: float) -> Dict[str, Any]:
        template = STRATEGY_LIBRARY[strategy]
        horizons = {}
        for horizon, allocations in template.items():
            horizons[horizon] = self._build_horizon_allocations(
                horizon, allocations, budget
            )
        return {
            "strategy": strategy,
            "description": self._describe_strategy(strategy),
            "horizons": horizons,
        }

    def _describe_strategy(self, strategy: str) -> str:
        descriptions = {
            "growth": "Aggressive mix leaning into innovation, AI, and crypto beta.",
            "balanced": "Even-handed mix balancing upside with drawdown control.",
            "defensive": "Capital preservation first with equity-light tilts.",
            "income": "Yield-focused mix emphasizing distributions and defensives.",
        }
        return descriptions.get(strategy, strategy)

    def _build_horizon_allocations(
        self,
        horizon_key: str,
        allocations: Dict[str, AllocationSuggestion],
        budget: float,
    ) -> Dict[str, Any]:
        total_weight = sum(item.weight for item in allocations.values())
        results = []
        cumulative = 0.0
        for asset_class, suggestion in allocations.items():
            weight = suggestion.weight / total_weight
            amount = round(budget * weight, 2)
            cumulative += amount
            results.append(
                {
                    "asset_class": asset_class,
                    "weight": round(weight, 3),
                    "amount": amount,
                    "rationale": suggestion.rationale,
                    "sample_assets": suggestion.sample_assets,
                }
            )
        drift = round(budget - cumulative, 2)
        if abs(drift) >= 0.01 and results:
            results[0]["amount"] = round(results[0]["amount"] + drift, 2)

        return {
            "label": HORIZON_LABELS.get(horizon_key, horizon_key),
            "allocations": results,
            "risk_focus": self._risk_focus(horizon_key),
        }

    def _risk_focus(self, horizon: str) -> str:
        focus_map = {
            "1w": "Liquidity & catalyst trading",
            "1m": "Trend capture with guardrails",
            "1y": "Compounding and thematic positioning",
        }
        return focus_map.get(horizon, "Balanced risk")


__all__ = ["MultiAssetAllocationAgent"]
