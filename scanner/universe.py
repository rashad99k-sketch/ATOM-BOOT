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

ASSET_CLASSES = ("CRYPTO", "STOCK", "INDEX", "METAL", "GOLD", "OIL", "FOREX", "ENERGY")

STOCKS = set(x.strip().upper() for x in os.getenv("STOCK_SYMBOL_HINTS", "".join([
    "AAPL,AMZN,GOOGL,MSFT,NVDA,META,TSLA,NFLX,AMD,INTC,AVGO,ORCL,CRM,ADBE,",
    "JPM,BAC,GS,MS,MU,PLTR,MSTR,COIN,HOOD,SOFI,UBER,SHOP,DELL,MRVL,SNDK,",
    "CRCL,SMCI,ARM,TSM,QCOM,CSCO,IBM,LLY,NVO,COST,WMT,DIS,NKE,BA,CAT"
])).split(",") if x.strip())

INDEX_HINTS = ("US500", "USSTECH", "USTECH", "US30", "DJI", "SPX", "SP500",
               "NASDAQ", "NDX", "DAX", "DE40", "GER40", "FTSE", "UK100", "CAC",
               "FRA40", "NIKKEI", "JP225", "HSI", "HK50", "AUS200", "EU50", "STOXX")
GOLD_HINTS = ("XAU", "GOLD")
METAL_HINTS = ("XAG", "SILVER", "PALLADIUM", "PLATINUM", "NICKEL",
               "ALUMINIUM", "ALUMINUM", "ZINC", "LEAD", "COPPER")
OIL_HINTS = ("OIL", "WTI", "BRENT", "CRUDE", "HEATINGOIL", "OILHEATING", "GASOLINE")
ENERGY_HINTS = ("NATURALGAS", "NATGAS",
                "COFFEE", "COCOA", "SOYBEAN", "SOYBEANS", "SUGAR", "WHEAT", "CORN",
                "GASOLINE", "COAL")
FOREX_HINTS = ("EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK",
               "PLN", "MXN", "ZAR", "TRY", "CNH", "HKD", "SGD", "THB", "DKK")


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
    # Metals and Gold are detected before FOREX so XAU-crosses become
    # correctly classified; pure FX pairs remain FOREX.
    if any(h in text for h in GOLD_HINTS):
        return "GOLD"
    if any(h in text for h in METAL_HINTS):
        return "METAL"
    if any(h in text for h in INDEX_HINTS):
        return "INDEX"
    if any(h in text for h in OIL_HINTS):
        return "OIL"
    if any(h in text for h in ENERGY_HINTS):
        return "ENERGY"
    if base in STOCKS or any(k in text for k in ("STOCK", "EQUITY", "SHARE")):
        return "STOCK"
    # pure FOREX: NCCO venue uses <BASE><CURRENCY>2USD style
    fore_hits = [h for h in FOREX_HINTS if h in base]
    if fore_hits and ("NCCO" in base or (base.endswith(("AUD","EUR","CHF","GBP","JPY","CAD","NZD"))
                                         and "USDT" not in str(symbol).upper())):
        return "FOREX"
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

    # Dynamic discovery: rank rows by a minimal but real activity signal
    # (discovery priority, not queue score). Within each class the ranking is
    # applied; across classes, market activity dominates. The quota guard is
    # optional (default unlimited) and preserved for rate-limit control only.
    def activity_key(row):
        m = row.get("market") or {}
        info = m.get("info") or {}
        vol = info.get("volume_24h", info.get("vol24h", info.get("volume", 0)))
        atr = info.get("atr", info.get("range", 0))
        news = info.get("news_score", 0)
        return (
            float(news or 0) * 1000 +
            float(vol or 0) * 1e-6 +
            float(atr or 0) * 100 +
            float(row.get("news_activity", 0) or 0)
        )

    classes_present = {asset for asset, _rows in buckets.items() if _rows}
    ranked_classes = sorted(
        classes_present,
        key=lambda a: max((activity_key(r) for r in buckets[a]), default=0),
        reverse=True,
    )

    quotas = {}
    for asset in ASSET_CLASSES:
        quotas[asset] = max(
            1,
            int(os.getenv(f"DEEP_SCAN_{asset}_QUOTA", str(len(buckets[asset]))))
        )

    out = []
    seen = set()
    limit = int(os.getenv("DEEP_MIN_REPRESENTATION", "1"))
    for round_no in range(max(quotas.values())):
        for asset in ranked_classes:
            bucket = buckets[asset]
            if round_no >= min(len(bucket), quotas[asset]):
                continue
            # Sort bucket by dynamic activity once (on first pass) so the
            # round-robin oscillates across ranked classes on the same cycle.
            row = bucket[round_no]
            if row["symbol"] in seen:
                continue
            seen.add(row["symbol"])
            out.append(row)
            if len(out) >= radar_limit:
                return out
    return out
