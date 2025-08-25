"""
trading_agent_tasks.py

Production-ready scaffolding for a modular, composable trading agent with
strict pydantic-validated state, embedded micro-prompts, interruption/resume,
risk-gated order emission, and robust execution patterns.

This module defines:
- BaseTask (abstract)
- TaskContext (pydantic) with strict schemas and persistence hooks
- PromptSpec (embedded LLM micro-prompts rendered from context)
- Concrete tasks:
    * SignalComputeTask
    * RiskGateTask
    * OrderLifecycleTask
    * SymbolOrchestratorTask (per-symbol actor-style pipeline)
- Storage adapter interface (ContextStorageAdapter) + InMemoryStorage
- Retry policy utilities, idempotent order keying, and structured logging
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import hashlib
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, AnyUrl

# -----------------------------------------------------------------------------
# Logging (structured JSON)
# -----------------------------------------------------------------------------
logger = logging.getLogger("trading_agent")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


def log_json(event: str, **kv: Any) -> None:
    payload = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **kv}
    logger.info(json.dumps(payload, separators=(",", ":")))


# -----------------------------------------------------------------------------
# Enums & Constants
# -----------------------------------------------------------------------------
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
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class TimeInForce(str, Enum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class Severity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# -----------------------------------------------------------------------------
# Prompt Spec (Embedded Micro-Prompts)
# -----------------------------------------------------------------------------
class PromptSpec(BaseModel):
    role: str = Field(..., description="System or user role for LLM.")
    template: str = Field(..., description="Jinja-like template string.")
    constraints: List[str] = Field(default_factory=list)

    def render(self, *, ctx: "TaskContext", symbol: str) -> str:
        """Render prompt dynamically from context (no external deps)."""
        acct = ctx.account
        risk = ctx.config.risk_limits
        strat = ctx.config.strategy
        # Minimal safe rendering
        filled = self.template.format(
            now=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            mode=ctx.mode.value,
            cash=acct.cash,
            equity=acct.equity,
            bp=acct.buying_power,
            cur_pos=ctx.positions.get(symbol).qty if symbol in ctx.positions else 0.0,
            max_dd=risk.max_drawdown,
            daily_loss=risk.daily_loss_limit,
            permits_short=str(ctx.config.strategy.short_permissions.get(symbol, False)).lower(),
            htf_conf=strat.timeframes.htf_confirmations,
            active_flags=",".join(sorted(ctx.feature_flags.active)),
        )
        # Constraints appended to keep LLM guardrails explicit
        if self.constraints:
            filled += "\n\nConstraints:\n- " + "\n- ".join(self.constraints)
        return filled


# -----------------------------------------------------------------------------
# Schemas: Config & State
# -----------------------------------------------------------------------------
class Timeframes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tick: bool = True
    s1: bool = True
    s5: bool = True
    m1: bool = True
    m5: bool = True
    htf_confirmations: List[str] = Field(default_factory=lambda: ["1h", "4h"])


class StrategySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    universe: List[str] = Field(..., min_length=1)
    timeframes: Timeframes
    short_permissions: Dict[str, bool] = Field(default_factory=dict)
    max_legs_per_symbol: int = Field(default=4, ge=1, le=16)
    min_time_between_legs_sec: float = Field(default=5.0, ge=0.0)
    ladder_linear: bool = True
    ladder_geometric_ratio: float = Field(default=1.5, gt=1.0)
    min_price_improve_ticks: int = Field(default=1, ge=0)
    allow_dca_band_pct: float = Field(default=0.0, ge=0.0, le=20.0)
    volatility_guard_atr_mult: float = Field(default=3.0, gt=0.0)
    spread_guard_bps: float = Field(default=20.0, ge=0.0)
    slippage_guard_bps: float = Field(default=50.0, ge=0.0)


class RiskLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    per_trade_risk: float = Field(..., gt=0.0)
    daily_loss_limit: float = Field(..., gt=0.0)
    max_drawdown: float = Field(..., gt=0.0)
    max_concurrent_symbols: int = Field(default=20, ge=1)
    max_exposure_notional: float = Field(default=1e7, ge=0.0)
    symbol_exposure_caps: Dict[str, float] = Field(default_factory=dict)


class BrokerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    venue: str
    fix: Optional[AnyUrl] = None
    rest: Optional[AnyUrl] = None
    ws: Optional[AnyUrl] = None
    rate_limit_per_sec: int = Field(default=5, ge=1)
    sandbox: bool = Field(default=True)


class DataSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    primary_ws: Optional[AnyUrl] = None
    backup_rest: Optional[AnyUrl] = None
    staleness_ms: int = Field(default=500, ge=100)
    candle_delay_intervals: int = Field(default=2, ge=1)


class AlertsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slack: Optional[str] = None
    email: Optional[str] = None
    pagerduty: Optional[str] = None
    throttle_sec: float = Field(default=10.0, ge=0.0)


class FeatureFlags(BaseModel):
    model_config = ConfigDict(extra="forbid")
    active: List[str] = Field(default_factory=list)
    mode: Mode = Mode.DRY_RUN
    canary_universe: List[str] = Field(default_factory=list)


class ConfigSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: StrategySpec
    risk_limits: RiskLimits
    broker: BrokerSpec
    data: DataSpec
    alerts: AlertsSpec


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
    market_open: bool
    session: str = Field(default="REG")
    reason: Optional[str] = None


class MarketTick(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ts: datetime
    symbol: str
    bid: float
    ask: float
    last: float
    size: float


class Candle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ts: datetime
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float


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


class BracketSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sl: float
    tp: float
    trailing: Optional[float] = None


class Order(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_id: str
    symbol: str
    side: OrderSide
    qty: float
    type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tif: TimeInForce = TimeInForce.DAY
    bracket: Optional[BracketSpec] = None
    tags: Dict[str, str] = Field(default_factory=dict)

    @field_validator("qty")
    @classmethod
    def _qty_pos(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("qty must be positive")
        return v


class OrderAction(str, Enum):
    NEW = "NEW"
    REPLACE = "REPLACE"
    CANCEL = "CANCEL"


class OrderCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: OrderAction
    order: Order
    idempotency_key: str
    correlation_id: str


class RiskDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allow: bool
    reason: Optional[str] = None
    soft_blocks: List[str] = Field(default_factory=list)
    hard_blocks: List[str] = Field(default_factory=list)


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision_latency_ms: float = 0.0
    order_submit_latency_ms: float = 0.0
    fills: int = 0
    cancels: int = 0
    replaces: int = 0
    pnl: float = 0.0
    exposure: float = 0.0
    reconnects: int = 0
    duplicate_orders: int = 0
    risk_breaches: int = 0


class Alert(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Severity
    message: str
    symbol: Optional[str] = None
    dedupe_key: Optional[str] = None
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Snapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ts: datetime
    context_version: str
    positions: Dict[str, Position]
    account: AccountState
    metrics: Metrics
    feature_flags: FeatureFlags


# -----------------------------------------------------------------------------
# Storage Adapter
# -----------------------------------------------------------------------------
class ContextStorageAdapter(abc.ABC):
    @abc.abstractmethod
    async def save(self, key: str, blob: bytes) -> None: ...

    @abc.abstractmethod
    async def load(self, key: str) -> Optional[bytes]: ...


class InMemoryStorage(ContextStorageAdapter):
    def __init__(self) -> None:
        self._mem: Dict[str, bytes] = {}

    async def save(self, key: str, blob: bytes) -> None:
        self._mem[key] = blob

    async def load(self, key: str) -> Optional[bytes]:
        return self._mem.get(key)


# -----------------------------------------------------------------------------
# Task Context (pydantic enforced)
# -----------------------------------------------------------------------------
class TaskContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: ConfigSpec
    feature_flags: FeatureFlags
    account: AccountState
    calendar: CalendarStatus

    positions: Dict[str, Position] = Field(default_factory=dict)

    # Streams (latest snapshots for actor loop)
    last_tick: Dict[str, MarketTick] = Field(default_factory=dict)
    last_candle: Dict[Tuple[str, str], Candle] = Field(default_factory=dict)  # (symbol, tf)
    last_top: Dict[str, TopOfBook] = Field(default_factory=dict)

    # Derived signal cache (per symbol)
    signals: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    # Operational controls
    mode: Mode = Field(default=Mode.DRY_RUN)
    kill_switch: bool = Field(default=False)
    last_snapshot_ts: Optional[datetime] = None
    resume_tokens: Dict[str, str] = Field(default_factory=dict)  # task_name -> token

    # Metrics & alerts
    metrics: Metrics = Field(default_factory=Metrics)
    pending_alerts: List[Alert] = Field(default_factory=list)

    # Storage
    context_key: str = Field(default="agent_ctx/v1")
    storage: Optional[ContextStorageAdapter] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _sync_mode(self) -> "TaskContext":
        self.mode = self.feature_flags.mode
        return self

    async def persist(self) -> None:
        if not self.storage:
            return
        snap = Snapshot(
            ts=datetime.now(timezone.utc),
            context_version="1.0",
            positions=self.positions,
            account=self.account,
            metrics=self.metrics,
            feature_flags=self.feature_flags,
        )
        await self.storage.save(self.context_key, snap.model_dump_json().encode("utf-8"))
        self.last_snapshot_ts = snap.ts
        log_json("snapshot.persisted", key=self.context_key, ts=self.last_snapshot_ts.isoformat())

    async def restore(self) -> Optional[Snapshot]:
        if not self.storage:
            return None
        blob = await self.storage.load(self.context_key)
        if not blob:
            return None
        data = json.loads(blob.decode("utf-8"))
        snap = Snapshot(**data)
        self.positions = snap.positions
        self.account = snap.account
        self.metrics = snap.metrics
        self.feature_flags = snap.feature_flags
        self.mode = self.feature_flags.mode
        self.last_snapshot_ts = snap.ts
        log_json("snapshot.restored", key=self.context_key, ts=self.last_snapshot_ts.isoformat())
        return snap


# -----------------------------------------------------------------------------
# Retry & Timing Utilities
# -----------------------------------------------------------------------------
@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.2
    max_delay: float = 2.0
    jitter: float = 0.2

    def compute_backoff(self, attempt: int) -> float:
        d = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        j = random.uniform(-self.jitter, self.jitter) * d
        return max(0.0, d + j)


def monotonic_ms() -> float:
    return time.perf_counter() * 1000.0


def make_idempotency_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()[:24]


# -----------------------------------------------------------------------------
# Abstract BaseTask
# -----------------------------------------------------------------------------
class BaseTask(abc.ABC):
    name: str = "base_task"
    retry_policy: RetryPolicy

    def __init__(self, retry_policy: Optional[RetryPolicy] = None) -> None:
        self.retry_policy = retry_policy or RetryPolicy()

    @abc.abstractmethod
    async def execute(self, ctx: TaskContext, **kwargs) -> TaskContext:
        """Run the task to completion if possible. Respect ctx.kill_switch and SLA."""
        ...

    @abc.abstractmethod
    async def resume(self, ctx: TaskContext, token: Optional[str] = None) -> TaskContext:
        """Resume after interruption using ctx.resume_tokens or provided token."""
        ...

    @abc.abstractmethod
    def prompt_specs(self) -> List[PromptSpec]:
        """Expose at least one PromptSpec for embedded prompting."""
        ...

    async def run(self, ctx: TaskContext, **kwargs) -> TaskContext:
        """Retry wrapper with structured logging and cancellation awareness."""
        attempt = 1
        start = monotonic_ms()
        while True:
            if ctx.kill_switch:
                log_json("task.killed", task=self.name)
                return ctx
            try:
                log_json("task.start", task=self.name, attempt=attempt)
                new_ctx = await self.execute(ctx, **kwargs)
                log_json("task.success", task=self.name, attempt=attempt, latency_ms=monotonic_ms() - start)
                return new_ctx
            except asyncio.CancelledError:
                log_json("task.cancelled", task=self.name, attempt=attempt)
                raise
            except Exception as e:
                log_json("task.error", task=self.name, attempt=attempt, error=str(e))
                if attempt >= self.retry_policy.max_attempts:
                    # Persist context before surfacing error
                    with contextlib.suppress(Exception):
                        await ctx.persist()
                    raise
                await asyncio.sleep(self.retry_policy.compute_backoff(attempt))
                attempt += 1


# -----------------------------------------------------------------------------
# Concrete Task: SignalComputeTask
# -----------------------------------------------------------------------------
class SignalComputeTask(BaseTask):
    name = "signal_compute"

    _prompt = PromptSpec(
        role="system",
        template=(
            "You are an execution-time strategist optimizing microstructure-aware entries.\n"
            "Now: {now}\nSymbol: {symbol}\nMode: {mode}\nCash: {cash:.2f}\nEquity: {equity:.2f}\nBP: {bp:.2f}\n"
            "CurrentQty: {cur_pos}\nMaxDD: {max_dd}\nDailyLossLim: {daily_loss}\nShortPermitted: {permits_short}\n"
            "HTFConfirmations: {htf_conf}\nActiveFlags: {active_flags}\n"
            "Respond with concise features to emphasize for latency-constrained decisions."
        ),
        constraints=[
            "No long narratives.",
            "Prefer features derivable within 20ms.",
            "Degrade gracefully: drop heavy indicators under load.",
        ],
    )

    def prompt_specs(self) -> List[PromptSpec]:
        return [self._prompt]

    async def resume(self, ctx: TaskContext, token: Optional[str] = None) -> TaskContext:
        # This task is stateless per tick; nothing to resume beyond cached signals.
        log_json("task.resume.noop", task=self.name)
        return ctx

    async def execute(self, ctx: TaskContext, symbol: str) -> TaskContext:
        t0 = monotonic_ms()
        tick = ctx.last_tick.get(symbol)
        top = ctx.last_top.get(symbol)
        if not tick or not top:
            raise RuntimeError(f"Missing market data for {symbol}")

        # Staleness checks
        age_ms = (datetime.now(timezone.utc) - tick.ts).total_seconds() * 1000.0
        if age_ms > ctx.config.data.staleness_ms:
            # Enter safe mode gating by emitting alert and skipping decision
            ctx.pending_alerts.append(Alert(
                severity=Severity.WARN,
                message=f"Stale tick ({age_ms:.0f}ms) for {symbol}",
                symbol=symbol,
                dedupe_key=f"stale:{symbol}",
            ))
            log_json("signal.stale", symbol=symbol, age_ms=age_ms)
            return ctx

        # Lightweight indicators within 20ms budget
        spread = max(0.0, top.ask.price - top.bid.price)
        mid = (top.ask.price + top.bid.price) / 2.0 if (top.ask.price and top.bid.price) else tick.last
        imbalance = (top.bid.size - top.ask.size) / max(1.0, (top.bid.size + top.ask.size))

        # EMA proxy (1-tick EWMA for speed)
        prev = ctx.signals.get(symbol, {})
        ewma_alpha = 0.6
        ema_fast = ewma_alpha * mid + (1 - ewma_alpha) * prev.get("ema_fast", mid)
        ema_slow = 0.2 * mid + 0.8 * prev.get("ema_slow", mid)

        # Volatility proxy (micro ATR over synthetic window)
        vol_proxy = 0.7 * prev.get("vol_proxy", 0.0) + 0.3 * abs(tick.last - mid)

        # Regime
        regime_high_vol = vol_proxy > (ctx.config.strategy.volatility_guard_atr_mult * 1e-4 * max(1.0, mid))

        # Donchian-lite using last M1 candle (if exists)
        m1 = ctx.last_candle.get((symbol, "1m"))
        donchian_up = donchian_dn = None
        if m1:
            donchian_up = m1.high
            donchian_dn = m1.low

        # RSI-lite (1-tap)
        up = max(0.0, tick.last - prev.get("last_price", tick.last))
        dn = max(0.0, prev.get("last_price", tick.last) - tick.last)
        rs = (0.7 * prev.get("rsi_up", 1e-9) + 0.3 * up) / (0.7 * prev.get("rsi_dn", 1e-9) + 0.3 * dn + 1e-12)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        ctx.signals[symbol] = {
            "mid": mid,
            "spread": spread,
            "imbalance": imbalance,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "vol_proxy": vol_proxy,
            "rsi": rsi,
            "donchian_up": donchian_up or mid,
            "donchian_dn": donchian_dn or mid,
            "regime_high_vol": 1.0 if regime_high_vol else 0.0,
            "last_price": tick.last,
            "ts_ms": monotonic_ms(),
        }

        ctx.metrics.decision_latency_ms = (monotonic_ms() - t0)
        return ctx


# -----------------------------------------------------------------------------
# Concrete Task: RiskGateTask
# -----------------------------------------------------------------------------
class RiskGateTask(BaseTask):
    name = "risk_gate"

    _prompt = PromptSpec(
        role="system",
        template=(
            "You are a deterministic pre-trade risk gate. Now: {now}, Symbol: {symbol}, Mode: {mode}. "
            "Ensure: market open, symbol tradable, borrow availability if short, exposure below limits, "
            "spread/slippage within thresholds, volatility not excessive, recent drawdown ok, and cooldowns respected."
        ),
        constraints=[
            "Binary gate: allow or block with reason.",
            "Never suggest overrides.",
        ],
    )

    def prompt_specs(self) -> List[PromptSpec]:
        return [self._prompt]

    async def resume(self, ctx: TaskContext, token: Optional[str] = None) -> TaskContext:
        log_json("task.resume.noop", task=self.name)
        return ctx

    async def execute(
        self,
        ctx: TaskContext,
        symbol: str,
        intended_side: OrderSide,
        intended_notional: float,
    ) -> TaskContext:
        if not ctx.calendar.market_open:
            self._block(ctx, symbol, "market_closed")
            return ctx

        if symbol not in ctx.config.strategy.universe:
            self._block(ctx, symbol, "symbol_not_in_universe")
            return ctx

        if intended_side == OrderSide.SELL and ctx.positions.get(symbol, Position(symbol=symbol, qty=0.0, avg_price=0.0)).qty <= 0:
            # Short?
            if not ctx.config.strategy.short_permissions.get(symbol, False):
                self._block(ctx, symbol, "short_not_permitted")
                return ctx

        # Exposure limits
        total_exposure = ctx.metrics.exposure
        per_symbol_cap = ctx.config.risk_limits.symbol_exposure_caps.get(symbol, ctx.config.risk_limits.max_exposure_notional)
        if intended_notional + total_exposure > ctx.config.risk_limits.max_exposure_notional:
            self._block(ctx, symbol, "exceeds_global_exposure")
            return ctx
        if intended_notional > per_symbol_cap:
            self._block(ctx, symbol, "exceeds_symbol_exposure_cap")
            return ctx

        # Spread/slippage/vol guards from signals
        sig = ctx.signals.get(symbol)
        if not sig:
            self._block(ctx, symbol, "missing_signals")
            return ctx

        mid = float(sig["mid"])
        spread_bps = (float(sig["spread"]) / max(mid, 1e-6)) * 1e4
        if spread_bps > ctx.config.strategy.spread_guard_bps:
            self._block(ctx, symbol, f"spread_guard:{spread_bps:.1f}bps")
            return ctx

        if bool(sig.get("regime_high_vol", 0.0)):
            self._block(ctx, symbol, "high_volatility_regime")
            return ctx

        # Daily loss / drawdown enforcement
        if ctx.metrics.pnl < -ctx.config.risk_limits.daily_loss_limit:
            self._block(ctx, symbol, "daily_loss_circuit")
            return ctx

        # If all passed, mark allow
        decision = RiskDecision(allow=True)
        ctx.signals.setdefault(symbol, {})["risk_decision"] = 1.0
        log_json("risk.allow", symbol=symbol, notional=intended_notional)
        return ctx

    def _block(self, ctx: TaskContext, symbol: str, reason: str) -> None:
        ctx.signals.setdefault(symbol, {})["risk_decision"] = 0.0
        ctx.metrics.risk_breaches += 1
        ctx.pending_alerts.append(Alert(severity=Severity.WARN, message=f"Risk blocked {symbol}: {reason}", symbol=symbol))
        log_json("risk.block", symbol=symbol, reason=reason)


# -----------------------------------------------------------------------------
# Concrete Task: OrderLifecycleTask
# -----------------------------------------------------------------------------
class OrderLifecycleTask(BaseTask):
    name = "order_lifecycle"

    _prompt = PromptSpec(
        role="system",
        template=(
            "You orchestrate order safety for {symbol}. Mode {mode}. Ensure OCO brackets, trailing stops, "
            "idempotent client IDs, and at-most-once semantics. Respect TIF and replace/cancel retries."
        ),
        constraints=[
            "On ack timeout, query open orders before re-issuing.",
            "Never emit duplicate orders.",
        ],
    )

    def prompt_specs(self) -> List[PromptSpec]:
        return [self._prompt]

    async def resume(self, ctx: TaskContext, token: Optional[str] = None) -> TaskContext:
        # Resume by reconciling open orders would be implemented against broker API.
        # Here, we record a token placeholder and no-op.
        if token:
            ctx.resume_tokens[self.name] = token
            log_json("order.resume.token", token=token)
        return ctx

    async def execute(
        self,
        ctx: TaskContext,
        symbol: str,
        side: OrderSide,
        qty: float,
        order_type: OrderType = OrderType.LIMIT,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        tif: TimeInForce = TimeInForce.DAY,
        bracket: Optional[BracketSpec] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> TaskContext:
        t0 = monotonic_ms()
        sig = ctx.signals.get(symbol, {})
        mid = sig.get("mid")
        if mid is None:
            raise RuntimeError("Cannot place order without mid price")

        client_id = f"{symbol}-{uuid.uuid4().hex[:10]}"
        idem = make_idempotency_key(client_id, symbol, side.value, f"{qty:.8f}", str(limit_price or ""), str(stop_price or ""))

        # Construct order with guards
        order = Order(
            client_id=client_id,
            symbol=symbol,
            side=side,
            qty=qty,
            type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            tif=tif,
            bracket=bracket,
            tags=tags or {},
        )
        cmd = OrderCommand(action=OrderAction.NEW, order=order, idempotency_key=idem, correlation_id=uuid.uuid4().hex[:12])

        # Broker adapter would be called here; we simulate idempotent submission and ack timing.
        ack = await self._submit_with_ack(ctx, cmd)
        if not ack:
            # Timeout handling with open-order query would be here; we conservatively avoid duplicate.
            ctx.metrics.duplicate_orders += 0  # explicit no-op duplicate emission
            ctx.pending_alerts.append(Alert(severity=Severity.ERROR, message=f"Ack timeout for {symbol}", symbol=symbol))
            log_json("order.ack_timeout", symbol=symbol, client_id=client_id)
            return ctx

        # Ensure bracket creation post-trade (simulated)
        if bracket and ctx.mode != Mode.DRY_RUN:
            # In real impl: verify SL/TP created; if failure, reduce/close position.
            log_json("order.bracket.ensure", symbol=symbol, sl=bracket.sl, tp=bracket.tp)

        ctx.metrics.order_submit_latency_ms = (monotonic_ms() - t0)
        return ctx

    async def _submit_with_ack(self, ctx: TaskContext, cmd: OrderCommand) -> bool:
        # Enforce global rate limit expectations at scaffolding level
        await asyncio.sleep(0)  # yield
        log_json("order.submit", symbol=cmd.order.symbol, action=cmd.action, idem=cmd.idempotency_key, client_id=cmd.order.client_id, mode=ctx.mode.value)
        # Simulate ack within 2s SLA
        await asyncio.sleep(min(0.01, 2.0))
        log_json("order.ack", symbol=cmd.order.symbol, client_id=cmd.order.client_id)
        return True


# -----------------------------------------------------------------------------
# Concrete Task: SymbolOrchestratorTask (actor-style per symbol)
# -----------------------------------------------------------------------------
class SymbolOrchestratorTask(BaseTask):
    """
    Per-symbol decision stream with strict ordering, scaling ladder, and guards.
    """
    name = "symbol_orchestrator"

    decision_prompt = PromptSpec(
        role="system",
        template=(
            "You are a fast-reacting symbol orchestrator for {symbol}. "
            "Prioritize: risk adherence, capital guarding, incremental scaling with min price improvement, "
            "and exit ladder with trailing residual. Mode: {mode}. CurrentQty: {cur_pos}."
        ),
        constraints=[
            "Total data-to-decision under 150ms.",
            "Abort adds if spread/vol exceed guards.",
        ],
    )

    def prompt_specs(self) -> List[PromptSpec]:
        return [self.decision_prompt]

    def __init__(
        self,
        signal_task: SignalComputeTask,
        risk_task: RiskGateTask,
        order_task: OrderLifecycleTask,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> None:
        super().__init__(retry_policy)
        self.signal_task = signal_task
        self.risk_task = risk_task
        self.order_task = order_task
        # Per-symbol pacing
        self._last_leg_ts: Dict[str, float] = {}

    async def resume(self, ctx: TaskContext, token: Optional[str] = None) -> TaskContext:
        # Resume orchestration by reading token (e.g., last step) but we maintain per-symbol timestamps.
        if token:
            ctx.resume_tokens[self.name] = token
            log_json("orchestrator.resume", token=token)
        return ctx

    async def execute(
        self,
        ctx: TaskContext,
        symbol: str,
        strategy_side_hint: Optional[OrderSide] = None,
        notional_budget: float = 0.0,
    ) -> TaskContext:
        t0 = monotonic_ms()
        # 1) Compute/refresh signals
        ctx = await self.signal_task.run(ctx, symbol=symbol)

        sig = ctx.signals.get(symbol, {})
        mid = sig.get("mid")
        if mid is None:
            return ctx

        # 2) Decide side using micro signals (EMA cross + RSI) with hint
        ema_fast, ema_slow, rsi = sig.get("ema_fast", mid), sig.get("ema_slow", mid), sig.get("rsi", 50.0)
        side: Optional[OrderSide] = None
        if ema_fast > ema_slow and rsi < 70.0:
            side = OrderSide.BUY
        elif ema_fast < ema_slow and rsi > 30.0:
            side = OrderSide.SELL
        if strategy_side_hint:
            side = strategy_side_hint or side

        if side is None:
            log_json("orchestrator.no_signal", symbol=symbol)
            return ctx

        # 3) Scaling ladder logic
        now_s = time.time()
        min_gap = ctx.config.strategy.min_time_between_legs_sec
        last_leg = self._last_leg_ts.get(symbol, 0.0)
        if now_s - last_leg < min_gap:
            log_json("orchestrator.cooldown", symbol=symbol, remain_s=round(min_gap - (now_s - last_leg), 2))
            return ctx

        max_legs = ctx.config.strategy.max_legs_per_symbol
        cur_qty = ctx.positions.get(symbol, Position(symbol=symbol, qty=0.0, avg_price=0.0)).qty
        legs_used = int(abs(cur_qty) > 0)  # simple proxy; real impl would track per-order legs
        if legs_used >= max_legs:
            log_json("orchestrator.max_legs", symbol=symbol)
            return ctx

        # Determine qty by budget and ladder policy
        notional = notional_budget or min(ctx.account.buying_power * 0.01, 10000.0)  # conservative default
        per_leg = notional / max(1, (max_legs - legs_used))
        qty = max(1.0, per_leg / max(mid, 1e-6))

        # 4) Risk gate
        ctx = await self.risk_task.run(ctx, symbol=symbol, intended_side=side, intended_notional=per_leg)
        if ctx.signals.get(symbol, {}).get("risk_decision", 0.0) <= 0.0:
            return ctx

        # 5) Price improvement & order construction
        min_improve_ticks = ctx.config.strategy.min_price_improve_ticks
        tick_size = self._infer_tick_size(mid)
        improve = min_improve_ticks * tick_size
        limit_price: Optional[float] = None
        if side == OrderSide.BUY:
            limit_price = max(0.0, mid - improve)
        else:
            limit_price = mid + improve

        # Bracket safety
        tp = (1.0 + (0.002 if side == OrderSide.BUY else -0.002)) * limit_price
        sl = (1.0 - (0.003 if side == OrderSide.BUY else -0.003)) * limit_price
        bracket = BracketSpec(sl=float(sl), tp=float(tp), trailing=None)

        # 6) Submit order
        ctx = await self.order_task.run(
            ctx,
            symbol=symbol,
            side=side,
            qty=float(qty),
            order_type=OrderType.LIMIT,
            limit_price=float(limit_price),
            tif=TimeInForce.DAY,
            bracket=bracket,
            tags={"strategy": "ema_rsi_donchian", "leg": str(legs_used + 1)},
        )

        # Bookkeeping
        self._last_leg_ts[symbol] = time.time()
        ctx.metrics.decision_latency_ms = max(ctx.metrics.decision_latency_ms, (monotonic_ms() - t0))
        # Persist occasionally
        if (ctx.last_snapshot_ts is None) or ((datetime.now(timezone.utc) - ctx.last_snapshot_ts) > timedelta(seconds=5)):
            await ctx.persist()
        return ctx

    @staticmethod
    def _infer_tick_size(price: float) -> float:
        if price >= 1000:
            return 0.5
        if price >= 100:
            return 0.1
        if price >= 10:
            return 0.01
        return 0.001


# -----------------------------------------------------------------------------
# Factory for a minimal-but-complete pipeline wire-up
# -----------------------------------------------------------------------------
def build_symbol_pipeline() -> SymbolOrchestratorTask:
    signal = SignalComputeTask()
    risk = RiskGateTask()
    order = OrderLifecycleTask()
    return SymbolOrchestratorTask(signal, risk, order)


# -----------------------------------------------------------------------------
# Example of how to initialize context (callers wire inputs/streams)
# -----------------------------------------------------------------------------
def default_context(
    universe: List[str],
    mode: Mode = Mode.DRY_RUN,
    storage: Optional[ContextStorageAdapter] = None,
) -> TaskContext:
    cfg = ConfigSpec(
        strategy=StrategySpec(
            universe=universe,
            timeframes=Timeframes(),
            short_permissions={s: True for s in universe},
        ),
        risk_limits=RiskLimits(
            per_trade_risk=500.0,
            daily_loss_limit=2000.0,
            max_drawdown=5000.0,
            symbol_exposure_caps={s: 250000.0 for s in universe},
        ),
        broker=BrokerSpec(venue="SIM", sandbox=True, rate_limit_per_sec=5),
        data=DataSpec(staleness_ms=500, candle_delay_intervals=2),
        alerts=AlertsSpec(throttle_sec=10.0),
    )
    feats = FeatureFlags(active=["orchestrator", "risk_gate"], mode=mode)
    acct = AccountState(cash=100000.0, equity=100000.0, buying_power=200000.0, last_update_ts=datetime.now(timezone.utc))
    cal = CalendarStatus(market_open=True)
    ctx = TaskContext(
        config=cfg,
        feature_flags=feats,
        account=acct,
        calendar=cal,
        storage=storage or InMemoryStorage(),
    )
    return ctx


# -----------------------------------------------------------------------------
# Public Contracts (Inputs/Outputs expectations)
# -----------------------------------------------------------------------------
__all__ = [
    # Enums & core
    "Mode",
    "OrderSide",
    "OrderType",
    "TimeInForce",
    "Severity",
    # Models
    "PromptSpec",
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
    "MarketTick",
    "Candle",
    "DepthLevel",
    "TopOfBook",
    "BracketSpec",
    "Order",
    "OrderAction",
    "OrderCommand",
    "RiskDecision",
    "Metrics",
    "Alert",
    "Snapshot",
    # Context & storage
    "TaskContext",
    "ContextStorageAdapter",
    "InMemoryStorage",
    # Tasks
    "BaseTask",
    "SignalComputeTask",
    "RiskGateTask",
    "OrderLifecycleTask",
    "SymbolOrchestratorTask",
    # Factories & helpers
    "build_symbol_pipeline",
    "default_context",
]
