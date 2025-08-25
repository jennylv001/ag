from __future__ import annotations

"""
Trading agent tasks (scaffold-adapted) with a BaseTask-compatible wrapper.

- Provides strict pydantic state (TaskContext) and trading models (Mode, OrderSide, etc.).
- Includes lightweight Signal/Risk/Order pipeline.
- Exposes TradingSymbolOrchestratorTask that implements this repo's BaseTask protocol
  (async run -> TaskResult) so it integrates with TASK_REGISTRY.

All heavy work happens at run-time; no side effects on import.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ConfigDict, field_validator

from .base_task import BaseTask, TaskResult


# ---------------- Enums ----------------
class Mode(str, Enum):
    DRY_RUN = "dry_run"
    PAPER = "paper"
    LIVE = "live"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class TimeInForce(str, Enum):
    DAY = "DAY"
    GTC = "GTC"


# ---------------- Models ----------------
class Timeframes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    htf_confirmations: List[str] = Field(default_factory=lambda: ["1h", "4h"])


class StrategySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    universe: List[str]
    timeframes: Timeframes = Field(default_factory=Timeframes)
    short_permissions: Dict[str, bool] = Field(default_factory=dict)
    max_legs_per_symbol: int = Field(default=4, ge=1, le=16)
    min_time_between_legs_sec: float = Field(default=5.0, ge=0.0)
    min_price_improve_ticks: int = Field(default=1, ge=0)
    spread_guard_bps: float = Field(default=20.0, ge=0.0)
    volatility_guard_atr_mult: float = Field(default=3.0, gt=0.0)


class RiskLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    per_trade_risk: float = Field(..., gt=0.0)
    daily_loss_limit: float = Field(..., gt=0.0)
    max_drawdown: float = Field(..., gt=0.0)
    max_exposure_notional: float = Field(default=1e7, ge=0.0)
    symbol_exposure_caps: Dict[str, float] = Field(default_factory=dict)


class BrokerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    venue: str = "SIM"
    rate_limit_per_sec: int = Field(default=5, ge=1)


class DataSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    staleness_ms: int = Field(default=500, ge=100)


class AlertsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    throttle_sec: float = Field(default=10.0, ge=0.0)


class FeatureFlags(BaseModel):
    model_config = ConfigDict(extra="forbid")
    active: List[str] = Field(default_factory=list)
    mode: Mode = Mode.DRY_RUN


class ConfigSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: StrategySpec
    risk_limits: RiskLimits
    broker: BrokerSpec = Field(default_factory=BrokerSpec)
    data: DataSpec = Field(default_factory=DataSpec)
    alerts: AlertsSpec = Field(default_factory=AlertsSpec)


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str
    qty: float
    avg_price: float
    unrealized_pnl: float = 0.0


class AccountState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cash: float
    equity: float
    buying_power: float
    margin_used: float = 0.0
    last_update_ts: datetime


class CalendarStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    market_open: bool = True
    session: str = Field(default="REG")


class DepthLevel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    price: float
    size: float


class TopOfBook(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ts: datetime
    symbol: str
    bid: DepthLevel
    ask: DepthLevel


class MarketTick(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ts: datetime
    symbol: str
    bid: float
    ask: float
    last: float
    size: float


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision_latency_ms: float = 0.0
    order_submit_latency_ms: float = 0.0
    exposure: float = 0.0
    pnl: float = 0.0
    risk_breaches: int = 0


class Alert(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: str
    message: str
    symbol: Optional[str] = None
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TaskContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: ConfigSpec
    feature_flags: FeatureFlags
    account: AccountState
    calendar: CalendarStatus

    positions: Dict[str, Position] = Field(default_factory=dict)
    last_tick: Dict[str, MarketTick] = Field(default_factory=dict)
    last_top: Dict[str, TopOfBook] = Field(default_factory=dict)
    last_candle: Dict[Tuple[str, str], Dict[str, Any]] = Field(default_factory=dict)

    signals: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    metrics: Metrics = Field(default_factory=Metrics)
    pending_alerts: List[Alert] = Field(default_factory=list)

    mode: Mode = Mode.DRY_RUN
    last_snapshot_ts: Optional[datetime] = None

    @field_validator("mode", mode="before")
    @classmethod
    def _coerce_mode(cls, v: Any) -> Mode:
        try:
            return Mode(str(v))
        except Exception:
            return Mode.DRY_RUN

    def snapshot_if_needed(self, *, interval_s: float = 5.0) -> None:
        now = datetime.now(timezone.utc)
        if self.last_snapshot_ts is None or (now - self.last_snapshot_ts) > timedelta(seconds=interval_s):
            self.last_snapshot_ts = now


# ---------------- Utilities ----------------
def monotonic_ms() -> float:
    return time.perf_counter() * 1000.0


# ---------------- Pipeline steps ----------------
@dataclass
class _Pipeline:
    last_leg_ts: Dict[str, float]

    def __init__(self) -> None:
        self.last_leg_ts = {}

    def compute_signals(self, ctx: TaskContext, symbol: str) -> TaskContext:
        tick = ctx.last_tick.get(symbol)
        top = ctx.last_top.get(symbol)
        if not tick or not top:
            raise RuntimeError(f"Missing market data for {symbol}")

        # Staleness
        age_ms = (datetime.now(timezone.utc) - tick.ts).total_seconds() * 1000.0
        if age_ms > ctx.config.data.staleness_ms:
            ctx.pending_alerts.append(Alert(severity="WARN", message=f"Stale tick {age_ms:.0f}ms", symbol=symbol))
            return ctx

        spread = max(0.0, top.ask.price - top.bid.price)
        mid = (top.ask.price + top.bid.price) / 2.0
        prev = ctx.signals.get(symbol, {})
        ema_fast = 0.6 * mid + 0.4 * prev.get("ema_fast", mid)
        ema_slow = 0.2 * mid + 0.8 * prev.get("ema_slow", mid)
        up = max(0.0, tick.last - prev.get("last_price", tick.last))
        dn = max(0.0, prev.get("last_price", tick.last) - tick.last)
        rs = (0.7 * prev.get("rsi_up", 1e-9) + 0.3 * up) / (0.7 * prev.get("rsi_dn", 1e-9) + 0.3 * dn + 1e-12)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        ctx.signals[symbol] = {
            "mid": mid,
            "spread": spread,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "rsi": rsi,
            "last_price": tick.last,
            "ts_ms": monotonic_ms(),
        }
        return ctx

    def risk_gate(self, ctx: TaskContext, symbol: str, side: OrderSide, notional: float) -> bool:
        if not ctx.calendar.market_open:
            ctx.pending_alerts.append(Alert(severity="WARN", message="market_closed", symbol=symbol))
            ctx.metrics.risk_breaches += 1
            return False
        if symbol not in ctx.config.strategy.universe:
            ctx.pending_alerts.append(Alert(severity="WARN", message="symbol_not_in_universe", symbol=symbol))
            ctx.metrics.risk_breaches += 1
            return False
        sig = ctx.signals.get(symbol)
        if not sig:
            ctx.pending_alerts.append(Alert(severity="WARN", message="missing_signals", symbol=symbol))
            ctx.metrics.risk_breaches += 1
            return False
        mid = float(sig["mid"]) if sig else 0.0
        spread_bps = (float(sig["spread"]) / max(mid, 1e-6)) * 1e4 if sig else 0.0
        if spread_bps > ctx.config.strategy.spread_guard_bps:
            ctx.pending_alerts.append(Alert(severity="WARN", message=f"spread_guard:{spread_bps:.1f}bps", symbol=symbol))
            ctx.metrics.risk_breaches += 1
            return False
        # Simple exposure/daily loss checks
        if ctx.metrics.pnl < -ctx.config.risk_limits.daily_loss_limit:
            ctx.pending_alerts.append(Alert(severity="WARN", message="daily_loss_circuit", symbol=symbol))
            ctx.metrics.risk_breaches += 1
            return False
        if notional + ctx.metrics.exposure > ctx.config.risk_limits.max_exposure_notional:
            ctx.pending_alerts.append(Alert(severity="WARN", message="exceeds_global_exposure", symbol=symbol))
            ctx.metrics.risk_breaches += 1
            return False
        return True

    def next_leg_allowed(self, ctx: TaskContext, symbol: str) -> bool:
        now = time.time()
        min_gap = ctx.config.strategy.min_time_between_legs_sec
        last = self.last_leg_ts.get(symbol, 0.0)
        if now - last < min_gap:
            return False
        # Simple legs-used proxy via position presence
        cur_qty = ctx.positions.get(symbol, Position(symbol=symbol, qty=0.0, avg_price=0.0)).qty
        legs_used = int(abs(cur_qty) > 0)
        return legs_used < ctx.config.strategy.max_legs_per_symbol

    def record_leg(self, symbol: str) -> None:
        self.last_leg_ts[symbol] = time.time()


# ---------------- Adapter Task (BaseTask-compatible) ----------------
class TradingSymbolOrchestratorTask(BaseTask):
    """Runs one per-symbol decision cycle and returns a TaskResult.

    Constructor args:
        symbol: trading symbol
        context: TaskContext (callers prepare ticks/top-of-book)
    """

    def __init__(self, symbol: str, context: TaskContext) -> None:
        self.symbol = symbol
        self.ctx = context
        self._done = False
        self._success = False
        self._pipe = _Pipeline()
        self._last_message: str | None = None
        self._last_data: dict[str, Any] | None = None

    def step(self) -> Optional[TaskResult]:
        return None

    def is_done(self) -> bool:
        return self._done

    def succeeded(self) -> bool:
        return self._success

    async def run(self) -> TaskResult:
        s = self.symbol
        # 1) compute signals
        self.ctx = self._pipe.compute_signals(self.ctx, s)
        sig = self.ctx.signals.get(s)
        if not sig or "mid" not in sig:
            self._done = True
            self._success = False
            self._last_message = "no_signals"
            return TaskResult(success=False, message=self._last_message, data={"symbol": s})

        # 2) decide side (EMA cross + RSI bands)
        side: Optional[OrderSide] = None
        ema_fast, ema_slow, rsi = sig.get("ema_fast", 0.0), sig.get("ema_slow", 0.0), sig.get("rsi", 50.0)
        if ema_fast > ema_slow and rsi < 70.0:
            side = OrderSide.BUY
        elif ema_fast < ema_slow and rsi > 30.0:
            side = OrderSide.SELL
        if side is None:
            self._done = True
            self._success = True
            self._last_message = "no_trade"
            return TaskResult(success=True, message=self._last_message, data={"symbol": s, "signals": sig})

        # 3) gate with risk
        budget = min(self.ctx.account.buying_power * 0.01, 10000.0)
        if not self._pipe.risk_gate(self.ctx, s, side, budget):
            self._done = True
            self._success = True
            self._last_message = "risk_blocked"
            return TaskResult(success=True, message=self._last_message, data={"symbol": s, "signals": sig})

        # 4) pacing / ladder
        if not self._pipe.next_leg_allowed(self.ctx, s):
            self._done = True
            self._success = True
            self._last_message = "cooldown"
            return TaskResult(success=True, message=self._last_message, data={"symbol": s, "signals": sig})

        # 5) construct a mock order (no broker)
        mid = float(sig["mid"])  # price hint
        qty = max(1.0, budget / max(mid, 1e-6))
        order = {
            "client_id": f"{s}-{uuid.uuid4().hex[:10]}",
            "symbol": s,
            "side": side.value,
            "qty": float(qty),
            "type": OrderType.LIMIT.value,
            "limit_price": float(mid),
            "tif": TimeInForce.DAY.value,
        }
        await asyncio.sleep(0)  # yield to event loop
        self._pipe.record_leg(s)

        # Bookkeeping and snapshot hint
        self.ctx.metrics.order_submit_latency_ms = 0.0
        self.ctx.metrics.decision_latency_ms = max(self.ctx.metrics.decision_latency_ms, 0.0)
        self.ctx.snapshot_if_needed(interval_s=5.0)

        self._done = True
        self._success = True
        self._last_message = "order_planned"
        self._last_data = {"symbol": s, "order": order, "signals": sig}
        return TaskResult(success=True, message=self._last_message, data=self._last_data)


__all__ = [
    # enums/models for external callers
    "Mode",
    "OrderSide",
    "OrderType",
    "TimeInForce",
    "Timeframes",
    "StrategySpec",
    "RiskLimits",
    "BrokerSpec",
    "DataSpec",
    "AlertsSpec",
    "FeatureFlags",
    "ConfigSpec",
    "Position",
    "AccountState",
    "CalendarStatus",
    "DepthLevel",
    "TopOfBook",
    "MarketTick",
    "Metrics",
    "Alert",
    "TaskContext",
    # BaseTask-compatible wrapper
    "TradingSymbolOrchestratorTask",
]
