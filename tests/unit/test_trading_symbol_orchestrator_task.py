from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from browser_use.agent.tasks.trading_agent_task import (
    TradingSymbolOrchestratorTask,
    TaskContext,
    ConfigSpec,
    StrategySpec,
    Timeframes,
    RiskLimits,
    BrokerSpec,
    DataSpec,
    AlertsSpec,
    FeatureFlags,
    AccountState,
    CalendarStatus,
    MarketTick,
    DepthLevel,
    TopOfBook,
    Position,
    Mode,
)


def make_ctx(symbol: str = "AAPL") -> TaskContext:
    cfg = ConfigSpec(
        strategy=StrategySpec(
            universe=[symbol],
            timeframes=Timeframes(),
            short_permissions={symbol: True},
        ),
        risk_limits=RiskLimits(
            per_trade_risk=100.0,
            daily_loss_limit=1000.0,
            max_drawdown=2000.0,
            symbol_exposure_caps={symbol: 100000.0},
        ),
        broker=BrokerSpec(),
        data=DataSpec(staleness_ms=500),
        alerts=AlertsSpec(),
    )
    feats = FeatureFlags(active=["trading"], mode=Mode.DRY_RUN)
    acct = AccountState(cash=100000.0, equity=100000.0, buying_power=200000.0, last_update_ts=datetime.now(timezone.utc))
    cal = CalendarStatus(market_open=True)
    ctx = TaskContext(config=cdf(cfg), feature_flags=feats, account=acct, calendar=cal)
    ts = datetime.now(timezone.utc)
    ctx.last_tick[symbol] = MarketTick(ts=ts, symbol=symbol, bid=100.0, ask=100.1, last=100.05, size=100)
    ctx.last_top[symbol] = TopOfBook(
        ts=ts,
        symbol=symbol,
        bid=DepthLevel(price=100.0, size=200),
        ask=DepthLevel(price=100.1, size=180),
    )
    ctx.positions[symbol] = Position(symbol=symbol, qty=0.0, avg_price=0.0)
    return ctx


def cdf(x):
    # identity helper to avoid pydantic re-parse by pytest snapshot; keeps object types
    return x


async def _run_task(task: TradingSymbolOrchestratorTask):
    return await task.run()


def test_trading_symbol_orchestrator_runs_and_returns_result():
    symbol = "AAPL"
    ctx = make_ctx(symbol)
    task = TradingSymbolOrchestratorTask(symbol=symbol, context=ctx)
    res = asyncio.get_event_loop().run_until_complete(_run_task(task))
    assert res.success is True
    assert res.message in {"no_trade", "risk_blocked", "cooldown", "order_planned"}
    assert isinstance(res.data, dict)

