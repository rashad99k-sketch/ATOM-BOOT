"""Multi-market universe construction for BingX TradFi + crypto.

The connected BingX venue exposes crypto plus TradFi perpetual/CFD-style
markets. This module classifies instruments without inventing symbols and
builds a balanced radar universe so crypto cannot crowd out stocks, indices,
gold, or oil.
"""
from __future__ import annotations
import os
from collections import defaultdict
from typing import Dict, Iterable, List

ASSET_CLASSES = ("CRYPTO", "STOCK", "INDEX", "GOLD", "OIL")

STOCKS = set(x.strip().upper() for x in os.getenv("STOCK_SYMBOL_HINTS", "".join([
    "AAPL,AMZN,GOOGL,MSFT,NVDA,META,TSLA,NFLX,AMD,INTC,AVGO,ORCL,CRM,ADBE,",
    "JPM,BAC,GS,MS,MU,PLTR,MSTR,COIN,HOOD,SOFI,UBER,SHOP,DELL,MRVL,SNDK,",
    "CRCL,SMCI,ARM,TSM,QCOM,CSCO,IBM,LLY,NVO,COST,WMT,DIS,NKE,BA,CAT"
])).split(",") if x.strip())

INDEX_HINTS = ("US500", "USSTECH", "USTECH", "US30", "DJI", "SPX", "SP500",
               "NASDAQ", "NDX", "DAX", "DE40", "GER40", "FTSE", "UK100", "CAC",
               "FRA40", "NIKKEI", "JP225", "HSI", "HK50", "AUS200", "EU50", "STOXX")
GOLD_HINTS = ("XAU", "GOLD")
OIL_HINTS = ("OIL", "WTI", "BRENT", "CRUDE")


def _text(symbol: str, market: dict) -> str:
    info = market.get("info") or {}
    values = [symbol, market.get("base", ""), market.get("quote", ""),
              market.get("type", ""), market.get("name", "")]
    if isinstance(info, dict):
        for k in ("name", "displayName", "symbol", "assetType", "category", "type", "contractType"):
            values.append(str(info.get(k, "")))
    return " ".join(map(str, values)).upper()


def classify(symbol: str, market: dict) -> str:
    text = _text(symbol, market)
    base = str(market.get("base", symbol)).upper().split("/")[0].replace("-USDT", "")
    if any(h in text for h in GOLD_HINTS):
        return "GOLD"
    if any(h in text for h in OIL_HINTS):
        return "OIL"
    if any(h in text for h in INDEX_HINTS):
        return "INDEX"
    if base in STOCKS or any(k in text for k in ("STOCK", "EQUITY", "SHARE")):
        return "STOCK"
    return "CRYPTO"


def _configured(asset: str) -> List[str]:
    raw = os.getenv(f"DEEP_SCAN_{asset}_SYMBOLS", "")
    return [x.strip() for x in raw.split(",") if x.strip()]


def build_balanced(markets: Dict[str, dict], radar_limit: int = 240) -> List[dict]:
    buckets = defaultdict(list)
    for symbol, market in markets.items():
        if not market or market.get("active") is False:
            continue
        mtype = str(market.get("type", "")).lower()
        if mtype not in {"swap", "future"}:
            continue
        asset = classify(symbol, market)
        buckets[asset].append({"symbol": symbol, "asset_class": asset,
                               "source": "venue_discovery", "market": market})

    # Add only configured symbols actually present on the venue.
    for asset in ASSET_CLASSES:
        present = {r["symbol"] for r in buckets[asset]}
        for symbol in _configured(asset):
            if symbol in markets and symbol not in present:
                buckets[asset].append({"symbol": symbol, "asset_class": asset,
                                       "source": "configured", "market": markets[symbol]})

    # By default every connected instrument is eligible. Optional per-class
    # quotas remain available as a deliberate rate-limit control.
    quotas = {}
    for asset in ASSET_CLASSES:
        quotas[asset] = max(
            1,
            int(os.getenv(f"DEEP_SCAN_{asset}_QUOTA", str(len(buckets[asset]))))
        )

    # Round-robin across classes: no single asset class can consume the whole radar.
    out = []
    seen = set()
    for round_no in range(max(quotas.values())):
        for asset in ASSET_CLASSES:
            bucket = buckets[asset]
            if round_no >= min(len(bucket), quotas[asset]):
                continue
            row = bucket[round_no]
            if row["symbol"] in seen:
                continue
            seen.add(row["symbol"])
            out.append(row)
            if len(out) >= radar_limit:
                return out
    return out
