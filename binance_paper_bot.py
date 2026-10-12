#!/usr/bin/env python3
"""
Binance Algorithmic Paper Trading Bot (1h Timeframe)
Single-file paper trading execution engine matching strategy in binance_algo_backtest.py:
- Timeframe: 1h
- Indicators: 20 EMA, 50 EMA, 200 EMA, 14 ATR
- Signals: Dual LONG/SHORT Trend Regime Condition (Condition-based, No Crossover required)
- Live Tracking: Session PnL, Unrealized PnL, Current Position, SL & TP targets.
"""

import sys
import time
import os
import datetime
import requests
import numpy as np
import pandas as pd

# =========================================================
# CONFIGURATION DEFAULT VALUES
# =========================================================

SYMBOL = "SOLUSDT"
INTERVAL = "15m"
INITIAL_CAPITAL = 3000.0  # Session starting balance in INR (₹) or USD ($)
RISK_PER_TRADE = 0.01  # 1% risk per trade
EMA_FAST = 10
EMA_SLOW = 30
EMA_TREND = 200
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
TP_RISK_REWARD_RATIO = 2.0  # Take Profit at 2x Risk (2:1 R:R)
SLIPPAGE_PERCENT = 0.0005  # 0.05% estimated slippage
TRADING_FEE_PERCENT = 0.001  # 0.1% Binance spot fee
POLL_INTERVAL_SECONDS = 10  # Seconds between market polling updates
CSV_LOG_FILE = "binance_paper_trades.csv"
STATE_FILE = "binance_paper_state.json"

# =========================================================
# BINANCE PUBLIC API DATA FETCHERS
# =========================================================


def fetch_recent_klines(symbol=SYMBOL, interval=INTERVAL, limit=300):
    """
    Fetches the most recent OHLCV kline candles from Binance Public REST API.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}

    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(
                data,
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )
            df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            return df[["datetime", "open", "high", "low", "close", "volume"]].copy()
        else:
            print(f"[API Error] Binance HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"[Network Error] Failed to fetch klines: {e}")
        return None


def fetch_live_price(symbol=SYMBOL):
    """
    Fetches immediate real-time ticker price from Binance Public REST API.
    """
    base_url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": symbol.upper()}
    try:
        resp = requests.get(base_url, params=params, timeout=5)
        if resp.status_code == 200:
            return float(resp.json()["price"])
    except Exception:
        pass
    return None


# =========================================================
# INDICATORS & STRATEGY SIGNAL
# =========================================================


def calculate_indicators(df):
    """
    Calculates 20 EMA, 50 EMA, 200 EMA, and 14 ATR on OHLCV DataFrame.
    """
    data = df.copy()
    data["ema_fast"] = data["close"].ewm(span=EMA_FAST, adjust=False).mean()
    data["ema_slow"] = data["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    data["ema_trend"] = data["close"].ewm(span=EMA_TREND, adjust=False).mean()

    prev_close = data["close"].shift(1)
    tr1 = data["high"] - data["low"]
    tr2 = (data["high"] - prev_close).abs()
    tr3 = (data["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data["atr"] = tr.rolling(window=ATR_PERIOD).mean()

    return data


def get_strategy_signal(df):
    """
    Evaluates strategy condition based on completed candles:
    - BUY (LONG): 20 EMA > 50 EMA AND Close > 200 EMA
    - SELL (SHORT): 20 EMA < 50 EMA AND Close < 200 EMA
    - HOLD: Neither condition met
    """
    if df is None or len(df) < 1:
        return "HOLD"

    latest = df.iloc[-1]

    close = latest["close"]
    ema_fast = latest["ema_fast"]
    ema_slow = latest["ema_slow"]
    ema_trend = latest["ema_trend"]

    if pd.isna(ema_trend) or pd.isna(ema_fast) or pd.isna(ema_slow):
        return "HOLD"

    # LONG condition: EMA 20 > EMA 50 and Close > EMA 200
    if (ema_fast > ema_slow) and (close > ema_trend):
        return "BUY"

    # SHORT condition: EMA 20 < EMA 50 and Close < EMA 200
    if (ema_fast < ema_slow) and (close < ema_trend):
        return "SELL"

    return "HOLD"


# =========================================================
# POSITION SIZING & RISK MANAGEMENT
# =========================================================


def calculate_position_size(equity, entry_price, initial_stop):
    """
    Calculates position size strictly risking 1% of total equity.
    """
    risk_per_unit = abs(entry_price - initial_stop)
    if risk_per_unit <= 1e-8:
        risk_per_unit = entry_price * 0.02

    max_risk_amount = equity * RISK_PER_TRADE
    qty_by_risk = max_risk_amount / risk_per_unit

    max_qty_by_capital = (equity * (1.0 - TRADING_FEE_PERCENT)) / entry_price
    return max(0.0, min(qty_by_risk, max_qty_by_capital))


# =========================================================
# PAPER TRADING ENGINE & SESSION MANAGEMENT
# =========================================================


class PaperTradingSession:
    def __init__(self, symbol=SYMBOL, starting_balance=INITIAL_CAPITAL):
        self.symbol = symbol.upper()
        self.starting_balance = float(starting_balance)
        self.cash = float(starting_balance)

        # Position State
        self.position = "FLAT"  # "FLAT", "LONG", "SHORT"
        self.qty = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.entry_fee = 0.0

        # Performance Tracking
        self.session_realized_pnl = 0.0
        self.closed_trades = []

        # CSV initialization
        if not os.path.exists(CSV_LOG_FILE):
            pd.DataFrame(
                columns=[
                    "session_start",
                    "entry_time",
                    "exit_time",
                    "symbol",
                    "side",
                    "entry_price",
                    "exit_price",
                    "quantity",
                    "stop_loss",
                    "take_profit",
                    "exit_reason",
                    "fees",
                    "slippage",
                    "realized_pnl",
                    "return_pct",
                ]
            ).to_csv(CSV_LOG_FILE, index=False)

    def get_unrealized_pnl(self, current_price):
        """
        Calculates active trade unrealized PnL in currency and percentage.
        """
        if self.position == "FLAT" or self.qty <= 0:
            return 0.0, 0.0

        estimated_exit_fee = self.qty * current_price * TRADING_FEE_PERCENT

        if self.position == "LONG":
            gross_pnl = self.qty * (current_price - self.entry_price)
        else:  # SHORT
            gross_pnl = self.qty * (self.entry_price - current_price)

        net_unrealized = gross_pnl - self.entry_fee - estimated_exit_fee
        pos_cost = self.qty * self.entry_price + self.entry_fee
        ret_pct = (net_unrealized / pos_cost) * 100.0 if pos_cost > 0 else 0.0
        return net_unrealized, ret_pct

    def open_position(self, side, price, atr, timestamp):
        """
        Executes a paper BUY (LONG) or SELL (SHORT) position.
        """
        current_equity = self.starting_balance + self.session_realized_pnl

        if side == "LONG":
            entry_price = price * (1.0 + SLIPPAGE_PERCENT)
            sl = entry_price - (atr * ATR_STOP_MULTIPLIER)
            risk_dist = entry_price - sl
            min_tp_dist = entry_price * 0.008  # Minimum 0.8% price move target
            raw_tp_dist = risk_dist * TP_RISK_REWARD_RATIO
            tp_dist = max(raw_tp_dist, min_tp_dist)
            tp = entry_price + tp_dist
        else:  # SHORT
            entry_price = price * (1.0 - SLIPPAGE_PERCENT)
            sl = entry_price + (atr * ATR_STOP_MULTIPLIER)
            risk_dist = sl - entry_price
            min_tp_dist = entry_price * 0.008  # Minimum 0.8% price move target
            raw_tp_dist = risk_dist * TP_RISK_REWARD_RATIO
            tp_dist = max(raw_tp_dist, min_tp_dist)
            tp = entry_price - tp_dist

        qty = calculate_position_size(current_equity, entry_price, sl)
        if qty <= 0:
            print(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Insufficient equity to open position."
            )
            return False

        entry_fee = qty * entry_price * TRADING_FEE_PERCENT

        self.position = side
        self.qty = qty
        self.entry_price = entry_price
        self.entry_time = timestamp
        self.stop_loss = sl
        self.take_profit = tp
        self.entry_fee = entry_fee

        print(
            f"\n[PAPER TRADE OPENED] {side} {qty:.4f} {self.symbol} @ ₹{entry_price:,.4f}"
        )
        print(
            f" ├─ Stop Loss   : ₹{sl:,.4f} ({((sl-entry_price)/entry_price)*100:+.4f}%)"
        )
        print(
            f" ├─ Take Profit : ₹{tp:,.4f} ({((tp-entry_price)/entry_price)*100:+.4f}%)"
        )
        print(f" └─ Estimated Fee: ₹{entry_fee:,.4f}\n")
        return True

    def close_position(self, exit_price, exit_reason, timestamp):
        """
        Closes current active paper trade and updates session statistics.
        """
        if self.position == "FLAT":
            return

        if self.position == "LONG":
            actual_exit_price = exit_price * (1.0 - SLIPPAGE_PERCENT)
            gross_pnl = self.qty * (actual_exit_price - self.entry_price)
        else:  # SHORT
            actual_exit_price = exit_price * (1.0 + SLIPPAGE_PERCENT)
            gross_pnl = self.qty * (self.entry_price - actual_exit_price)

        exit_fee = self.qty * actual_exit_price * TRADING_FEE_PERCENT
        total_fees = self.entry_fee + exit_fee
        slippage_cost = self.qty * abs(exit_price - actual_exit_price)
        net_pnl = gross_pnl - exit_fee

        pos_cost = self.qty * self.entry_price + self.entry_fee
        ret_pct = (net_pnl / pos_cost) * 100.0 if pos_cost > 0 else 0.0

        self.session_realized_pnl += net_pnl

        trade_record = {
            "session_start": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "entry_time": str(self.entry_time),
            "exit_time": str(timestamp),
            "symbol": self.symbol,
            "side": self.position,
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(actual_exit_price, 4),
            "quantity": round(self.qty, 4),
            "stop_loss": round(self.stop_loss, 4),
            "take_profit": round(self.take_profit, 4),
            "exit_reason": exit_reason,
            "fees": round(total_fees, 4),
            "slippage": round(slippage_cost, 4),
            "realized_pnl": round(net_pnl, 4),
            "return_pct": round(ret_pct, 4),
        }

        self.closed_trades.append(trade_record)

        # Log to CSV
        try:
            pd.DataFrame([trade_record]).to_csv(
                CSV_LOG_FILE, mode="a", header=False, index=False
            )
        except Exception as e:
            print(f"Error writing trade to CSV: {e}")

        print(
            f"\n[PAPER TRADE CLOSED] {self.position} {self.symbol} @ ₹{actual_exit_price:,.4f}"
        )
        print(f" ├─ Reason       : {exit_reason}")
        print(f" ├─ Realized PnL : ₹{net_pnl:+,.4f} ({ret_pct:+.4f}%)")
        print(f" └─ Session PnL  : ₹{self.session_realized_pnl:+,.4f}\n")

        # Reset position
        self.position = "FLAT"
        self.qty = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.entry_fee = 0.0


# =========================================================
# DASHBOARD DISPLAY & LIVE MONITORING
# =========================================================


def print_dashboard(session, current_price, latest_candle, df):
    """
    Renders live terminal console dashboard showing session status, PnL, active position, TP/SL.
    """
    os.system("clear" if os.name == "posix" else "cls")

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    candle_time = latest_candle["datetime"].strftime("%Y-%m-%d %H:%M UTC")

    ema_fast = latest_candle["ema_fast"]
    ema_slow = latest_candle["ema_slow"]
    ema_trend = latest_candle["ema_trend"]
    atr = latest_candle["atr"]

    unrealized_pnl, unrealized_pct = session.get_unrealized_pnl(current_price)
    total_session_pnl = session.session_realized_pnl + unrealized_pnl
    current_equity = session.starting_balance + total_session_pnl
    session_return_pct = (total_session_pnl / session.starting_balance) * 100.0

    print("=========================================================================")
    print(
        f"           BINANCE {INTERVAL.upper()} ALGORITHMIC PAPER TRADING BOT                     "
    )
    print("=========================================================================")
    print(f" System Time   : {now_str} | Polling Binance API")
    print(f" Symbol        : {session.symbol} | Timeframe: {INTERVAL}")
    print(f" Live Price    : ₹{current_price:,.4f} USDT")
    print(f" Latest Candle : {candle_time} | Close: ₹{latest_candle['close']:,.4f}")
    print("-------------------------------------------------------------------------")
    print(f" TECHNICAL INDICATORS ({INTERVAL.upper()}):")
    print(f" ├─ EMA 20 (Fast)  : ₹{ema_fast:,.4f}")
    print(
        f" ├─ EMA 50 (Slow)  : ₹{ema_slow:,.4f}  [{'BULLISH (Fast > Slow)' if ema_fast > ema_slow else 'BEARISH (Fast < Slow)'}]"
    )
    print(
        f" ├─ EMA 200 (Trend): ₹{ema_trend:,.4f} [{'ABOVE MACRO TREND' if current_price > ema_trend else 'BELOW MACRO TREND'}]"
    )
    print(f" └─ ATR 14 (Vol)   : ₹{atr:,.4f}")
    print("-------------------------------------------------------------------------")
    print(" POSITION & RISK MONITOR:")

    if session.position == "FLAT":
        print(" Status        : ⚪ FLAT (No Active Position)")
        print(" Signal        : Monitoring Trend Regime Condition (EMA 20/50 & EMA 200)")
    else:
        side_icon = "🟢 LONG" if session.position == "LONG" else "🔴 SHORT"
        sl_dist = abs(current_price - session.stop_loss)
        tp_dist = abs(current_price - session.take_profit)
        sl_dist_pct = (sl_dist / current_price) * 100.0
        tp_dist_pct = (tp_dist / current_price) * 100.0

        print(
            f" Status        : {side_icon} | Size: {session.qty:.4f} {session.symbol}"
        )
        print(
            f" Entry Price   : ₹{session.entry_price:,.4f} (Opened at {session.entry_time})"
        )
        print(
            f" 🛑 Stop Loss  : ₹{session.stop_loss:,.4f} (Distance: ₹{sl_dist:,.4f} / {sl_dist_pct:.4f}%)"
        )
        print(
            f" 🎯 Take Profit: ₹{session.take_profit:,.4f} (Distance: ₹{tp_dist:,.4f} / {tp_dist_pct:.4f}%)"
        )

        pnl_color_icon = "🟢" if unrealized_pnl >= 0 else "🔴"
        print(
            f" Unrealized PnL: {pnl_color_icon} ₹{unrealized_pnl:+,.4f} ({unrealized_pct:+.4f}%)"
        )

    print("-------------------------------------------------------------------------")
    print(" SESSION PERFORMANCE SUMMARY:")
    print(f" Starting Balance   : ₹{session.starting_balance:,.4f}")
    print(f" Session Realized   : ₹{session.session_realized_pnl:+,.4f}")
    print(f" Session Unrealized : ₹{unrealized_pnl:+,.4f}")
    print(
        f" Total Session PnL  : ₹{total_session_pnl:+,.4f} ({session_return_pct:+.4f}%)"
    )
    print(f" Current Equity     : ₹{current_equity:,.4f}")
    print(f" Trades Executed    : {len(session.closed_trades)}")

    if session.closed_trades:
        print(
            "-------------------------------------------------------------------------"
        )
        print(" SESSION RECENT CLOSED TRADES:")
        for t in session.closed_trades[-3:]:
            print(
                f"  • {t['exit_time']} | {t['side']} | Exit: ₹{t['exit_price']:,.4f} | Reason: {t['exit_reason']} | PnL: ₹{t['realized_pnl']:+,.4f} ({t['return_pct']:+.4f}%)"
            )

    print("=========================================================================")
    print(" Press Ctrl+C to terminate bot cleanly and view summary report.")
    print("=========================================================================")


# =========================================================
# MAIN LOOP & CLI PARSING
# =========================================================


def run_paper_trading_bot(single_run=False):
    print("Initializing Binance Algorithmic Paper Trading Bot (1h Strategy)...")

    symbol = SYMBOL
    starting_balance = INITIAL_CAPITAL

    # Check command line args
    if "--once" in sys.argv or "--single-run" in sys.argv:
        single_run = True

    if not single_run and sys.stdin.isatty():
        try:
            sym_input = input(f"Enter Binance Symbol [{SYMBOL}]: ").strip().upper()
            if sym_input:
                symbol = sym_input
            bal_input = input(
                f"Enter Starting Paper Capital [₹{INITIAL_CAPITAL:.0f}]: "
            ).strip()
            if bal_input:
                starting_balance = float(bal_input)
        except Exception:
            pass

    session = PaperTradingSession(symbol=symbol, starting_balance=starting_balance)

    print(
        f"Starting paper trading session for {session.symbol} (Initial Capital: ₹{session.starting_balance:,.4f})..."
    )
    time.sleep(1.0)

    try:
        while True:
            # 1. Download kline history
            df_raw = fetch_recent_klines(
                symbol=session.symbol, interval=INTERVAL, limit=300
            )
            if df_raw is None or len(df_raw) < 50:
                print("Fetching data from Binance API...")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # 2. Calculate Indicators
            df = calculate_indicators(df_raw)

            # Completed candles (excluding ongoing unclosed candle for signals)
            df_completed = df.iloc[:-1].copy()
            latest_candle = df_completed.iloc[-1]

            # Live real-time price
            live_price = fetch_live_price(session.symbol)
            if live_price is None:
                live_price = df.iloc[-1]["close"]

            current_dt = datetime.datetime.now(datetime.timezone.utc)

            # 3. Check SL / TP for active position against live price
            if session.position == "LONG":
                if live_price <= session.stop_loss:
                    session.close_position(
                        session.stop_loss, "Stop Loss Hit 🛑", current_dt
                    )
                elif live_price >= session.take_profit:
                    session.close_position(
                        session.take_profit, "Take Profit Hit 🎯", current_dt
                    )

            elif session.position == "SHORT":
                if live_price >= session.stop_loss:
                    session.close_position(
                        session.stop_loss, "Stop Loss Hit 🛑", current_dt
                    )
                elif live_price <= session.take_profit:
                    session.close_position(
                        session.take_profit, "Take Profit Hit 🎯", current_dt
                    )

            # 4. Check Strategy Signal on Candle Close / Update
            ema_f = latest_candle["ema_fast"]
            ema_s = latest_candle["ema_slow"]

            # Evaluate EMA Exit condition
            if session.position == "LONG" and ema_f < ema_s:
                session.close_position(
                    live_price, "EMA Trend Exit (Fast < Slow)", current_dt
                )
            elif session.position == "SHORT" and ema_f > ema_s:
                session.close_position(
                    live_price, "EMA Trend Exit (Fast > Slow)", current_dt
                )

            # Check strategy signal strictly based on trend regime condition
            signal = get_strategy_signal(df_completed)
            atr_val = latest_candle["atr"]

            if signal == "BUY" and session.position != "LONG":
                if session.position == "SHORT":
                    session.close_position(
                        live_price, "Strategy Signal Reversal to LONG", current_dt
                    )
                session.open_position("LONG", live_price, atr_val, current_dt)

            elif signal == "SELL" and session.position != "SHORT":
                if session.position == "LONG":
                    session.close_position(
                        live_price, "Strategy Signal Reversal to SHORT", current_dt
                    )
                session.open_position("SHORT", live_price, atr_val, current_dt)

            # 5. Render Console Dashboard
            print_dashboard(session, live_price, latest_candle, df_completed)

            if single_run:
                break

            # Wait for next poll cycle
            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n\n=========================================================")
        print("          PAPER TRADING BOT TERMINATED BY USER           ")
        print("=========================================================")
        unrealized, unreal_pct = session.get_unrealized_pnl(
            fetch_live_price(session.symbol) or 0
        )
        total_pnl = session.session_realized_pnl + unrealized
        final_eq = session.starting_balance + total_pnl
        print(f" Starting Balance   : ₹{session.starting_balance:,.4f}")
        print(f" Session Realized   : ₹{session.session_realized_pnl:+,.4f}")
        print(f" Session Unrealized : ₹{unrealized:+,.4f}")
        print(f" Final Session Equity: ₹{final_eq:,.4f}")
        print(
            f" Total Session PnL  : ₹{total_pnl:+,.4f} ({(total_pnl/session.starting_balance)*100:+.4f}%)"
        )
        print(f" Trade Log Saved To : {CSV_LOG_FILE}")
        print("=========================================================\n")


if __name__ == "__main__":
    run_paper_trading_bot()
