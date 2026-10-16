#!/usr/bin/env python3
"""
Binance Algorithmic Paper Trading Bot - Channel Breakout Strategy (Donchian-style)
Single-file paper trading execution engine based on the architecture of binance_paper_bot.py:
- Timeframe: 5m (Configurable: 1m, 5m, 15m, 1h, etc.)
- Channel Lookback: Highest High & Lowest Low of previous N completed candles (default: 20)
- Volume Filter: Optional volume confirmation (Volume >= 20 SMA Volume)
- Entry Triggers:
    * LONG: Close breaks ABOVE previous N-candle Upper Channel
    * SHORT: Close breaks BELOW previous N-candle Lower Channel (if ALLOW_SHORT is True)
- Risk Management: 1% equity risk per trade, ATR-based Stop Loss (2x ATR 14), 2:1 Take Profit (2R target)
- Exit Rules: Stop-Loss hit, Take-Profit hit (2R), or Channel Reversal (Opposite Breakout)
- Accounting: Full round-trip fees (0.1% entry + 0.1% exit), realistic slippage (0.05%), position sizing
- Persistence: Independent CSV log (channel_breakout_trades.csv) & state (channel_breakout_state.json)
- Zero Interference: Operates 100% independently from binance_paper_bot.py
"""

import sys
import os
import json
import datetime
import time
import requests
import numpy as np
import pandas as pd

# =========================================================
# STRATEGY & ENGINE CONFIGURATION
# =========================================================

SYMBOL = "SOLUSDT"
INTERVAL = "1m"  # Binance intervals: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d
INITIAL_CAPITAL = 3000.0  # Session starting balance in USDT
RISK_PER_TRADE = 0.01  # 1% equity risk per trade

# Channel Breakout Parameters
CHANNEL_LOOKBACK = 20  # Number of completed candles to calculate Upper/Lower channel bounds
USE_VOLUME_FILTER = True  # Require volume >= 20 SMA volume on breakout candle
VOLUME_SMA_PERIOD = 20  # Lookback for average volume calculation

# Risk & Exit Parameters
ALLOW_SHORT = True  # True for Futures/Derivatives simulation, False for Spot (LONG-only)
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0  # Stop Loss = Entry +/- (2.0 * ATR)
TP_RISK_REWARD_RATIO = 2.0  # Take Profit = Entry +/- (2.0 * Risk Distance) [2:1 R:R]
COOLDOWN_CANDLES = 0  # Minimum completed candles to wait after trade exit before next entry

# Market Friction & Simulation Settings
SLIPPAGE_PERCENT = 0.0005  # 0.05% simulated slippage on entry/exit
TRADING_FEE_PERCENT = 0.001  # 0.1% Binance maker/taker fee per side (0.2% round-trip)
POLL_INTERVAL_SECONDS = 10  # Seconds between polling loops

# Dedicated Independent File Outputs
CSV_LOG_FILE = "channel_breakout_trades.csv"
STATE_FILE = "channel_breakout_state.json"
SIGNALS_LOG_FILE = "channel_breakout_signals.csv"

BINANCE_API_BASE = "https://api.binance.com"


# =========================================================
# MARKET DATA FETCHER & API INTEGRATION
# =========================================================

def fetch_recent_klines(symbol, interval, limit=200):
    """
    Downloads historical candlestick (kline) data from Binance public REST API.
    Does not require API keys or live execution permissions.
    """
    endpoint = f"{BINANCE_API_BASE}/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit
    }
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        if response.status_code == 200:
            raw = response.json()
            cols = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ]
            df = pd.DataFrame(raw, columns=cols)
            df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = df[c].astype(float)
            return df[["datetime", "open", "high", "low", "close", "volume"]]
        else:
            print(f"[API Error] Binance HTTP {response.status_code}: {response.text[:100]}")
            return None
    except Exception as e:
        print(f"[Network Error] Error fetching klines: {e}")
        return None


def fetch_live_price(symbol):
    """
    Retrieves latest real-time ticker price for precise execution and SL/TP checking.
    """
    endpoint = f"{BINANCE_API_BASE}/api/v3/ticker/price"
    params = {"symbol": symbol.upper()}
    try:
        response = requests.get(endpoint, params=params, timeout=5)
        if response.status_code == 200:
            return float(response.json()["price"])
        return None
    except Exception as e:
        print(f"[Network Error] Error fetching live price: {e}")
        return None


# =========================================================
# TECHNICAL INDICATOR ENGINE (CHANNEL & ATR)
# =========================================================

def calculate_indicators(df):
    """
    Computes Channel Breakout bounds, ATR, and Volume moving average.
    Avoids lookahead bias by shifting channel bounds by 1 so the current candle
    is evaluated strictly against the previous N completed candles.
    """
    df = df.copy()

    # 1. True Range & Average True Range (ATR 14)
    df["prev_close"] = df["close"].shift(1)
    df["tr1"] = df["high"] - df["low"]
    df["tr2"] = (df["high"] - df["prev_close"]).abs()
    df["tr3"] = (df["low"] - df["prev_close"]).abs()
    df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
    df["atr"] = df["tr"].rolling(window=ATR_PERIOD, min_periods=ATR_PERIOD).mean()

    # 2. Channel Bounds (Highest High & Lowest Low of previous N candles)
    # Shifting by 1 ensures we look at previous N completed candles EXCLUDING the test candle
    df["channel_upper"] = df["high"].shift(1).rolling(window=CHANNEL_LOOKBACK).max()
    df["channel_lower"] = df["low"].shift(1).rolling(window=CHANNEL_LOOKBACK).min()
    df["channel_mid"] = (df["channel_upper"] + df["channel_lower"]) / 2.0

    # 3. Volume Filter (SMA of Volume)
    df["volume_sma"] = df["volume"].rolling(window=VOLUME_SMA_PERIOD).mean()

    return df


def get_channel_breakout_signal(df_completed):
    """
    Evaluates Channel Breakout condition on the latest completed candle.
    - BUY (LONG):
        1. Close > Previous N-Candle Upper Channel Bound
        2. (Optional) Volume >= Volume SMA
    - SELL (SHORT):
        1. Close < Previous N-Candle Lower Channel Bound
        2. (Optional) Volume >= Volume SMA
    - HOLD: Price inside channel or requirements not met
    """
    if df_completed is None or len(df_completed) < (CHANNEL_LOOKBACK + 5):
        return "HOLD", "Insufficient history"

    latest = df_completed.iloc[-1]

    close = latest["close"]
    upper = latest["channel_upper"]
    lower = latest["channel_lower"]
    atr = latest["atr"]
    volume = latest["volume"]
    volume_sma = latest["volume_sma"]

    if pd.isna(upper) or pd.isna(lower) or pd.isna(atr) or atr <= 0:
        return "HOLD", "Indicators initializing"

    # Volume Confirmation Filter
    volume_confirmed = True
    if USE_VOLUME_FILTER and not pd.isna(volume_sma):
        volume_confirmed = (volume >= volume_sma)

    # LONG Breakout condition
    if close > upper and volume_confirmed:
        return "BUY", f"Bullish Breakout > Upper Channel ({upper:.4f})"

    # SHORT Breakout condition
    if close < lower and volume_confirmed:
        return "SELL", f"Bearish Breakdown < Lower Channel ({lower:.4f})"

    return "HOLD", "Inside Channel (Consolidation)"


def calculate_position_size(equity, entry_price, stop_loss_price):
    """
    Computes strict risk-based position size matching binance_paper_bot.py:
    Risk per trade = Equity * RISK_PER_TRADE (default 1%)
    Position Size = Risk Amount / Distance to Stop Loss
    """
    risk_amount = equity * RISK_PER_TRADE
    risk_per_unit = abs(entry_price - stop_loss_price)
    if risk_per_unit <= 0 or entry_price <= 0:
        return 0.0

    target_qty = risk_amount / risk_per_unit

    # Safety margin: limit maximum position notional to 3x equity
    max_notional = equity * 3.0
    if (target_qty * entry_price) > max_notional:
        target_qty = max_notional / entry_price

    return target_qty


# =========================================================
# PAPER TRADING SESSION & PORTFOLIO ENGINE
# =========================================================

class ChannelBreakoutSession:
    """
    Manages active position state, risk accounting, order execution simulation,
    state persistence, and trade history recording for Channel Breakout Strategy.
    """
    def __init__(self, symbol, starting_balance=INITIAL_CAPITAL):
        self.symbol = symbol.upper()
        self.starting_balance = float(starting_balance)

        # Active Position State
        self.position = "FLAT"  # 'FLAT', 'LONG', 'SHORT'
        self.qty = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.entry_fee = 0.0

        # Cooldown State
        self.cooldown_candles = 0

        # Performance Tracking
        self.session_realized_pnl = 0.0
        self.closed_trades = []

        # Initialize CSV log file with identical compatible schema
        if not os.path.exists(CSV_LOG_FILE):
            pd.DataFrame(
                columns=[
                    "timestamp",
                    "entry_time",
                    "exit_time",
                    "symbol",
                    "side",
                    "entry_price",
                    "exit_price",
                    "qty",
                    "stop_loss",
                    "take_profit",
                    "exit_reason",
                    "entry_fee",
                    "exit_fee",
                    "realized_pnl",
                    "return_pct"
                ]
            ).to_csv(CSV_LOG_FILE, index=False)

        # Initialize Signals CSV log
        if not os.path.exists(SIGNALS_LOG_FILE):
            pd.DataFrame(
                columns=[
                    "timestamp",
                    "symbol",
                    "close",
                    "channel_upper",
                    "channel_lower",
                    "atr",
                    "volume",
                    "volume_sma",
                    "signal",
                    "reason"
                ]
            ).to_csv(SIGNALS_LOG_FILE, index=False)

        # Restore previous state if available
        self.load_state()

    def save_state(self):
        """
        Persists current session position, realized PnL, and cooldown state to STATE_FILE.
        """
        try:
            state = {
                "strategy": "Channel_Breakout",
                "symbol": self.symbol,
                "interval": INTERVAL,
                "channel_lookback": CHANNEL_LOOKBACK,
                "starting_balance": self.starting_balance,
                "position": self.position,
                "qty": self.qty,
                "entry_price": self.entry_price,
                "entry_time": str(self.entry_time) if self.entry_time else None,
                "stop_loss": self.stop_loss,
                "take_profit": self.take_profit,
                "entry_fee": self.entry_fee,
                "session_realized_pnl": self.session_realized_pnl,
                "cooldown_candles": self.cooldown_candles,
                "trades_count": len(self.closed_trades),
                "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[Warning] Failed to save state to {STATE_FILE}: {e}")

    def load_state(self):
        """
        Restores session, active position, and cooldown state from STATE_FILE.
        """
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            if state.get("symbol") == self.symbol:
                # Prompt to resume or start clean
                if sys.stdin.isatty():
                    ans = input(
                        f"Found saved Channel Breakout session for {self.symbol} "
                        f"(PnL: {state.get('session_realized_pnl', 0):+.2f} USDT). Resume? [Y/n]: "
                    ).strip().lower()
                    if ans == "n":
                        print("[Session Reset] Starting fresh Channel Breakout session.")
                        return

                self.position = state.get("position", "FLAT")
                self.qty = float(state.get("qty", 0.0))
                self.entry_price = float(state.get("entry_price", 0.0))
                self.entry_time = state.get("entry_time")
                self.stop_loss = float(state.get("stop_loss", 0.0))
                self.take_profit = float(state.get("take_profit", 0.0))
                self.entry_fee = float(state.get("entry_fee", 0.0))
                self.session_realized_pnl = float(state.get("session_realized_pnl", 0.0))
                self.cooldown_candles = int(state.get("cooldown_candles", 0))
                print(
                    f"[State Restored] Channel Breakout Session for {self.symbol}: Position={self.position}, "
                    f"Qty={self.qty:.4f}, Cooldown={self.cooldown_candles}c, Realized PnL={self.session_realized_pnl:+.4f} USDT"
                )
        except Exception as e:
            print(f"[Warning] Failed to load state from {STATE_FILE}: {e}")

    def get_unrealized_pnl(self, current_price):
        """
        Calculates active trade unrealized PnL in currency and percentage.
        Deducts entry fee and estimated exit fee.
        """
        if self.position == "FLAT" or self.qty <= 0 or not current_price:
            return 0.0, 0.0

        if self.position == "LONG":
            gross = (current_price - self.entry_price) * self.qty
        else:  # SHORT
            gross = (self.entry_price - current_price) * self.qty

        est_exit_fee = self.qty * current_price * TRADING_FEE_PERCENT
        net_unrealized = gross - self.entry_fee - est_exit_fee

        pos_cost = self.qty * self.entry_price + self.entry_fee
        ret_pct = (net_unrealized / pos_cost) * 100.0 if pos_cost > 0 else 0.0

        return net_unrealized, ret_pct

    def open_position(self, side, price, atr, timestamp):
        """
        Executes a paper BUY (LONG) or SELL (SHORT) breakout position.
        Calculates realistic slippage, fee, ATR Stop Loss, and 2:1 Take Profit.
        """
        current_equity = self.starting_balance + self.session_realized_pnl

        if side == "LONG":
            entry_price = price * (1.0 + SLIPPAGE_PERCENT)
            sl = entry_price - (atr * ATR_STOP_MULTIPLIER)
            risk_dist = entry_price - sl
            tp_dist = risk_dist * TP_RISK_REWARD_RATIO
            tp = entry_price + tp_dist
        else:  # SHORT
            entry_price = price * (1.0 - SLIPPAGE_PERCENT)
            sl = entry_price + (atr * ATR_STOP_MULTIPLIER)
            risk_dist = sl - entry_price
            tp_dist = risk_dist * TP_RISK_REWARD_RATIO
            tp = entry_price - tp_dist

        qty = calculate_position_size(current_equity, entry_price, sl)
        if qty <= 0:
            print("[Position Sizing Alert] Calculated trade size is 0. Trade skipped.")
            return False

        entry_fee = qty * entry_price * TRADING_FEE_PERCENT

        self.position = side
        self.qty = qty
        self.entry_price = entry_price
        self.entry_time = str(timestamp)
        self.stop_loss = sl
        self.take_profit = tp
        self.entry_fee = entry_fee

        print(f"\n[CHANNEL BREAKOUT EXECUTED] {side} {qty:.4f} {self.symbol} @ {entry_price:,.4f} USDT")
        print(f" ├─ Stop Loss   : {sl:,.4f} USDT ({((sl-entry_price)/entry_price)*100:+.4f}%) [2x ATR]")
        print(f" ├─ Take Profit : {tp:,.4f} USDT ({((tp-entry_price)/entry_price)*100:+.4f}%) [2R Target]")
        print(f" └─ Estimated Fee: {entry_fee:,.4f} USDT\n")

        self.save_state()
        return True

    def close_position(self, exit_price, exit_reason, timestamp):
        """
        Closes current active paper trade, updates statistics, and initiates cooldown.
        Deducts both entry and exit fees for full round-trip accuracy.
        """
        if self.position == "FLAT":
            return

        # Slippage on exit
        if self.position == "LONG":
            actual_exit_price = exit_price * (1.0 - SLIPPAGE_PERCENT)
            gross_pnl = (actual_exit_price - self.entry_price) * self.qty
        else:  # SHORT
            actual_exit_price = exit_price * (1.0 + SLIPPAGE_PERCENT)
            gross_pnl = (self.entry_price - actual_exit_price) * self.qty

        exit_fee = self.qty * actual_exit_price * TRADING_FEE_PERCENT
        total_fees = self.entry_fee + exit_fee
        net_pnl = gross_pnl - total_fees

        pos_cost = self.qty * self.entry_price + self.entry_fee
        ret_pct = (net_pnl / pos_cost) * 100.0 if pos_cost > 0 else 0.0

        self.session_realized_pnl += net_pnl

        trade_record = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "entry_time": self.entry_time,
            "exit_time": str(timestamp),
            "symbol": self.symbol,
            "side": self.position,
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(actual_exit_price, 4),
            "qty": round(self.qty, 4),
            "stop_loss": round(self.stop_loss, 4),
            "take_profit": round(self.take_profit, 4),
            "exit_reason": exit_reason,
            "entry_fee": round(self.entry_fee, 4),
            "exit_fee": round(exit_fee, 4),
            "realized_pnl": round(net_pnl, 4),
            "return_pct": round(ret_pct, 4)
        }

        self.closed_trades.append(trade_record)

        # Append trade to dedicated CSV
        try:
            pd.DataFrame([trade_record]).to_csv(
                CSV_LOG_FILE, mode="a", header=False, index=False
            )
        except Exception as e:
            print(f"[CSV Error] Failed writing trade to {CSV_LOG_FILE}: {e}")

        print(f"\n[PAPER TRADE CLOSED] {self.position} {self.symbol} @ {actual_exit_price:,.4f} USDT")
        print(f" ├─ Exit Reason : {exit_reason}")
        print(f" ├─ Realized PnL: {net_pnl:+,.4f} USDT ({ret_pct:+.4f}%)")
        print(f" ├─ Session PnL : {self.session_realized_pnl:+,.4f} USDT")
        print(f" └─ Cooldown    : Initiated ({COOLDOWN_CANDLES} completed candles)\n")

        # Reset position and initiate cooldown
        self.position = "FLAT"
        self.qty = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.entry_fee = 0.0
        self.cooldown_candles = COOLDOWN_CANDLES

        self.save_state()

    def log_signal(self, candle, signal, reason):
        """
        Records signal checks to dedicated signals CSV for post-session analysis.
        """
        try:
            row = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": self.symbol,
                "close": round(candle["close"], 4),
                "channel_upper": round(candle["channel_upper"], 4) if not pd.isna(candle["channel_upper"]) else 0.0,
                "channel_lower": round(candle["channel_lower"], 4) if not pd.isna(candle["channel_lower"]) else 0.0,
                "atr": round(candle["atr"], 4) if not pd.isna(candle["atr"]) else 0.0,
                "volume": round(candle["volume"], 2),
                "volume_sma": round(candle["volume_sma"], 2) if not pd.isna(candle["volume_sma"]) else 0.0,
                "signal": signal,
                "reason": reason
            }
            pd.DataFrame([row]).to_csv(SIGNALS_LOG_FILE, mode="a", header=False, index=False)
        except Exception as e:
            print(f"[CSV Error] Error logging signal: {e}")


# =========================================================
# CONSOLE DASHBOARD RENDERING ENGINE
# =========================================================

def print_dashboard(session, current_price, latest_candle, df):
    """
    Renders live terminal console dashboard showing session status, PnL, active position, TP/SL, and cooldown.
    """
    os.system("clear" if os.name == "posix" else "cls")

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    candle_time = latest_candle["datetime"].strftime("%Y-%m-%d %H:%M UTC")

    upper = latest_candle["channel_upper"]
    lower = latest_candle["channel_lower"]
    mid = latest_candle["channel_mid"]
    atr = latest_candle["atr"]
    vol = latest_candle["volume"]
    vol_sma = latest_candle["volume_sma"]

    unrealized_pnl, unrealized_pct = session.get_unrealized_pnl(current_price)
    total_session_pnl = session.session_realized_pnl + unrealized_pnl
    current_equity = session.starting_balance + total_session_pnl
    session_return_pct = (total_session_pnl / session.starting_balance) * 100.0

    mode_label = "LONG/SHORT (Futures Sim)" if ALLOW_SHORT else "LONG-ONLY (Spot Sim)"

    # Price position relative to channel
    if current_price > upper:
        channel_status = "⚡ BREAKOUT (Above Upper Channel)"
    elif current_price < lower:
        channel_status = "⚡ BREAKDOWN (Below Lower Channel)"
    else:
        channel_status = "INSIDE CHANNEL (Consolidation)"

    vol_status = "CONFIRMED (> Avg)" if (vol >= vol_sma) else "LOW (< Avg)"

    print("=========================================================================")
    print(f"        BINANCE {INTERVAL.upper()} CHANNEL BREAKOUT PAPER TRADING BOT          ")
    print("=========================================================================")
    print(f" System Time   : {now_str} | Polling: {POLL_INTERVAL_SECONDS}s")
    print(f" Symbol        : {session.symbol} | Mode: {mode_label} | Timeframe: {INTERVAL}")
    print(f" Live Price    : {current_price:,.4f} USDT")
    print(f" Latest Candle : {candle_time} | Close: {latest_candle['close']:,.4f} USDT")
    print("-------------------------------------------------------------------------")
    print(f" CHANNEL INDICATORS ({INTERVAL.upper()} - {CHANNEL_LOOKBACK} Periods Lookback):")
    print(f" ├─ Upper Channel   : {upper:,.4f} USDT [Distance: {((current_price - upper)/upper)*100:+.2f}%]")
    print(f" ├─ Lower Channel   : {lower:,.4f} USDT [Distance: {((current_price - lower)/lower)*100:+.2f}%]")
    print(f" ├─ Channel Midline : {mid:,.4f} USDT [{channel_status}]")
    print(f" ├─ ATR 14 (Volat.) : {atr:,.4f} USDT (Stop Buffer: {atr * ATR_STOP_MULTIPLIER:,.4f})")
    print(f" └─ Volume Check    : {vol:,.0f} vs SMA {vol_sma:,.0f} [{vol_status}]")
    print("-------------------------------------------------------------------------")
    print(" POSITION & RISK MONITOR:")

    if session.position == "FLAT":
        print(" Status        : ⚪ FLAT (No Active Position)")
        if session.cooldown_candles > 0:
            print(f" Cooldown      : ⏳ ACTIVE ({session.cooldown_candles}/{COOLDOWN_CANDLES} completed candles remaining)")
        else:
            print(" Cooldown      : ✅ READY (Awaiting Next Channel Breakout)")
        print(f" Target Signal : Breakout above {upper:,.4f} (BUY) or below {lower:,.4f} (SELL)")
    else:
        side_icon = "🟢 LONG" if session.position == "LONG" else "🔴 SHORT"
        sl_dist = abs(current_price - session.stop_loss)
        sl_dist_pct = (sl_dist / current_price) * 100.0
        tp_dist = abs(current_price - session.take_profit)
        tp_dist_pct = (tp_dist / current_price) * 100.0

        print(f" Status        : {side_icon} | Size: {session.qty:.4f} {session.symbol}")
        print(f" Entry Price   : {session.entry_price:,.4f} USDT (Opened at {session.entry_time})")
        print(f" 🛑 Stop Loss  : {session.stop_loss:,.4f} USDT (Distance: {sl_dist:,.4f} USDT / {sl_dist_pct:.4f}%)")
        print(f" 🎯 Take Profit: {session.take_profit:,.4f} USDT (Distance: {tp_dist:,.4f} USDT / {tp_dist_pct:.4f}%) [2R]")
        print(" 🔄 Reversal   : Opposite Channel Breakout Active")

        pnl_color_icon = "🟢" if unrealized_pnl >= 0 else "🔴"
        print(f" Unrealized PnL: {pnl_color_icon} {unrealized_pnl:+,.4f} USDT ({unrealized_pct:+.4f}%)")

    print("-------------------------------------------------------------------------")
    print(" SESSION PERFORMANCE SUMMARY:")
    print(f" Starting Balance   : {session.starting_balance:,.4f} USDT")
    print(f" Session Realized   : {session.session_realized_pnl:+,.4f} USDT")
    print(f" Session Unrealized : {unrealized_pnl:+,.4f} USDT")
    print(f" Total Session PnL  : {total_session_pnl:+,.4f} USDT ({session_return_pct:+.4f}%)")
    print(f" Current Equity     : {current_equity:,.4f} USDT")
    print(f" Trades Executed    : {len(session.closed_trades)}")

    if session.closed_trades:
        print("-------------------------------------------------------------------------")
        print(" SESSION RECENT CLOSED TRADES:")
        for t in session.closed_trades[-3:]:
            print(
                f"  • {t['exit_time']} | {t['side']} | Exit: {t['exit_price']:,.4f} USDT | "
                f"Reason: {t['exit_reason']} | PnL: {t['realized_pnl']:+,.4f} USDT ({t['return_pct']:+.4f}%)"
            )

    print("=========================================================================")
    print(f" Dedicated Logs: {CSV_LOG_FILE} | {STATE_FILE} | {SIGNALS_LOG_FILE}")
    print(" Press Ctrl+C at any time to halt session gracefully.")
    print("=========================================================================")


# =========================================================
# MAIN EXECUTION LOOP & CLI RUNNER
# =========================================================

def run_channel_breakout_bot(single_run=False):
    """
    Main loop managing market data ingestion, candle close detection,
    risk monitoring, and breakout execution.
    """
    print(f"Initializing Binance Channel Breakout Paper Trading Bot ({INTERVAL} Strategy)...")

    symbol = SYMBOL
    starting_balance = INITIAL_CAPITAL

    # Interactive input matching binance_paper_bot.py behavior
    if not single_run and sys.stdin.isatty():
        try:
            sym_input = input(f"Enter Binance Symbol [{SYMBOL}]: ").strip()
            if sym_input:
                symbol = sym_input.replace("\\", "").upper()
            bal_input = input(f"Enter Starting Paper Capital [{INITIAL_CAPITAL:.2f} USDT]: ").strip()
            if bal_input:
                starting_balance = float(bal_input)
        except (KeyboardInterrupt, EOFError):
            print("\nAborted by user.")
            return

    session = ChannelBreakoutSession(symbol=symbol, starting_balance=starting_balance)

    print(f"Starting Channel Breakout session for {session.symbol} (Capital: {session.starting_balance:,.2f} USDT)...")
    time.sleep(1.0)

    last_processed_candle = None

    try:
        while True:
            closed_this_cycle = False

            # 1. Download kline history (ensure sufficient history for channel lookback + ATR)
            df_raw = fetch_recent_klines(
                symbol=session.symbol,
                interval=INTERVAL,
                limit=max(CHANNEL_LOOKBACK + 50, 100)
            )

            if df_raw is None or len(df_raw) < (CHANNEL_LOOKBACK + 15):
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Waiting for sufficient candle data...")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # 2. Compute Indicators
            df = calculate_indicators(df_raw)

            # Completed candles (excluding ongoing unclosed candle for signals)
            df_completed = df.iloc[:-1].copy()
            latest_candle = df_completed.iloc[-1]
            candle_time = latest_candle["datetime"]
            is_new_candle = (last_processed_candle is None or candle_time != last_processed_candle)

            # Live real-time price for SL/TP evaluation
            live_price = fetch_live_price(session.symbol)
            if live_price is None:
                live_price = latest_candle["close"]

            current_dt = datetime.datetime.now(datetime.timezone.utc)

            # 3. Check Stop-Loss and Take-Profit on every poll tick
            if session.position == "LONG":
                if live_price <= session.stop_loss:
                    session.close_position(session.stop_loss, "Stop Loss Hit 🛑", current_dt)
                    closed_this_cycle = True
                elif live_price >= session.take_profit:
                    session.close_position(session.take_profit, "Take Profit Hit 🎯 [2R]", current_dt)
                    closed_this_cycle = True

            elif session.position == "SHORT":
                if live_price >= session.stop_loss:
                    session.close_position(session.stop_loss, "Stop Loss Hit 🛑", current_dt)
                    closed_this_cycle = True
                elif live_price <= session.take_profit:
                    session.close_position(session.take_profit, "Take Profit Hit 🎯 [2R]", current_dt)
                    closed_this_cycle = True

            # 4. Check Strategy Signals & Reversals on newly closed candle
            if is_new_candle:
                last_processed_candle = candle_time

                # Advance cooldown when flat
                if session.position == "FLAT" and session.cooldown_candles > 0:
                    session.cooldown_candles -= 1
                    session.save_state()

                signal, reason = get_channel_breakout_signal(df_completed)
                atr_val = latest_candle["atr"]

                # Log signal state
                session.log_signal(latest_candle, signal, reason)

                # Check Opposite Channel Breakout Exit (Early Reversal)
                if session.position == "LONG" and signal == "SELL":
                    session.close_position(live_price, "Opposite Channel Breakdown Exit", current_dt)
                    closed_this_cycle = True
                elif session.position == "SHORT" and signal == "BUY":
                    session.close_position(live_price, "Opposite Channel Breakout Exit", current_dt)
                    closed_this_cycle = True

                # Check fresh entry on confirmed candle breakout
                if (
                    not closed_this_cycle
                    and session.position == "FLAT"
                    and session.cooldown_candles == 0
                ):
                    if signal == "BUY":
                        session.open_position("LONG", live_price, atr_val, current_dt)
                    elif signal == "SELL" and ALLOW_SHORT:
                        session.open_position("SHORT", live_price, atr_val, current_dt)

            # 5. Render Console Dashboard
            print_dashboard(session, live_price, latest_candle, df_completed)

            if single_run:
                break

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n[Shutdown] Paper trading session terminated gracefully by user.")
        unrealized, _ = session.get_unrealized_pnl(live_price if 'live_price' in locals() and live_price else 0)
        total_pnl = session.session_realized_pnl + unrealized
        final_eq = session.starting_balance + total_pnl
        print("\n================ FINAL SESSION REPORT ================")
        print(f" Starting Balance   : {session.starting_balance:,.4f} USDT")
        print(f" Session Realized   : {session.session_realized_pnl:+,.4f} USDT")
        print(f" Session Unrealized : {unrealized:+,.4f} USDT")
        print(f" Final Session Equity: {final_eq:,.4f} USDT")
        print(f" Total Session PnL  : {total_pnl:+,.4f} USDT ({(total_pnl/session.starting_balance)*100:+.4f}%)")
        print(f" Trade Log Saved To : {CSV_LOG_FILE}")
        print(f" State Saved To     : {STATE_FILE}")
        print(f" Signal Log Saved To: {SIGNALS_LOG_FILE}")
        print("======================================================\n")


if __name__ == "__main__":
    run_channel_breakout_bot()
