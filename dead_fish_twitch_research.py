"""
Dead Fish Twitch / 死鱼抽搐

An experimental A-share mean-reversion research script.

The original hypothesis was simple: under chaotic market conditions, certain
"ignored" short-term oversold stocks might show a brief mean-reversion twitch.
After data repair, funnel diagnosis, ranking validation, tail-risk filtering,
and fixed-horizon trade validation, the current conclusion is intentionally
conservative:

    The original trading hypothesis did not survive empirical validation.
    The research pipeline did.

This file is kept as a reproducible research case, not as a live trading system
and not as financial advice.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import tushare as ts
from scipy.stats import entropy

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = os.getcwd()


def _load_local_token() -> str:
    return os.getenv("TUSHARE_TOKEN", "").strip()


CONFIG = {
    "token": _load_local_token(),
    "start_date": "20230101",
    "end_date": "20260530",
    "index_code": "000001.SH",
    "pool_code": "000300.SH",
    "cache_dir": os.path.join(ROOT, "dead_fish_cache_v412_opt"),
    "use_cache": True,
    "commission": 0.0003,
    "stamp_tax": 0.001,
    "slippage_buy": 0.001,
    "slippage_sell": 0.001,
    "min_3d_decline": -0.01,
    "ma20_buffer": 0.95,
    "min_avg_amount": 50000,
    "max_gap_up": 0.025,
    "max_index_decline": -0.03,
    "min_list_days": 60,
    "hurst_window": 40,
    "max_beta_abs": 0.6,
    "max_vol": 2.5,
    "min_r_squared": 0.03,
    "max_hurst": 0.56,
    "min_vol_ratio": 1.05,
    "min_vol_vol_ratio": 1.0,
    "max_positions": 2,
    "batch1_take_profit": 0.03,
    "batch1_stop_loss": -0.04,
    "batch1_max_hold": 5,
    "batch1_trigger_batch2": -0.03,
    "batch2_take_profit": 0.025,
    "batch2_stop_loss": -0.04,
    "batch2_max_hold": 5,
    "max_batch2_per_stock": 1,
    "unified_stop_loss": -0.05,
    "score_threshold": 1.8,
    "max_position_pct": 0.15,
    "pending_buy_expire_days": 1,
    "limit_up_pct": 0.098,
    "limit_down_pct": -0.098,
    "recent_crash_days": 5,
    "recent_crash_threshold": -0.07,
    "initial_cash": 1_000_000,
    "force_refresh_index_weights": False,
    "force_refresh_daily": False,
    "force_refresh_adj": False,
    "min_daily_rows_warn": 500,
    "request_sleep": 0.08,
    "run_funnel_diagnosis": True,
    "run_rank_validation": True,
    "run_tail_filter_validation": True,
    "run_tail_trade_validation": True,
    "rank_forward_days": 5,
    "rank_top_n": 2,
    "tail_score_quantile": 0.8,
    "tail_max_vol_ratio": 2.0,
    "tail_max_hurst": 0.70,
    "tail_min_ret_3d": -0.04,
    "tail_min_recent_ret": -0.03,
    "tail_hold_days": 5,
    "tail_trade_max_positions": 2,
    "near_miss_top_n": 20,
}

os.makedirs(CONFIG["cache_dir"], exist_ok=True)
if not CONFIG["token"]:
    raise RuntimeError("Missing Tushare token. Set the TUSHARE_TOKEN environment variable before running.")

ts.set_token(CONFIG["token"])
pro = ts.pro_api()


def get_or_cache(filename, fetch_func, force=False):
    path = os.path.join(CONFIG["cache_dir"], filename)
    if not force and CONFIG["use_cache"] and os.path.exists(path):
        age_days = (time.time() - os.path.getmtime(path)) / 86400
        if age_days < 3:
            print(f"   [缓存] {filename}")
            df = pd.read_csv(path, dtype={"trade_date": str, "list_date": str})
            if "trade_date" in df.columns:
                df["trade_date"] = df["trade_date"].astype(str)
            return df
    print(f"   [拉取] {filename}")
    df = fetch_func()
    if CONFIG["use_cache"] and df is not None and len(df) > 0:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def _safe_code_filename(code):
    return code.replace(".", "_")


def fetch_stock_file(code, subdir, fetch_func, force=False, min_rows=1):
    folder = os.path.join(CONFIG["cache_dir"], subdir)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{_safe_code_filename(code)}.csv")
    if not force and os.path.exists(path):
        df = pd.read_csv(path, dtype={"trade_date": str})
        if len(df) >= min_rows:
            if "trade_date" in df.columns:
                df["trade_date"] = df["trade_date"].astype(str)
            return df
    last_error = None
    for attempt in range(3):
        try:
            df = fetch_func(code)
            if df is not None and len(df) > 0:
                df = df.copy()
                if "trade_date" in df.columns:
                    df["trade_date"] = df["trade_date"].astype(str)
                df.to_csv(path, index=False, encoding="utf-8-sig")
                time.sleep(CONFIG["request_sleep"])
                return df
        except Exception as exc:
            last_error = exc
            time.sleep(0.5 + attempt * 0.5)
    if last_error is not None:
        print(f"   警告: {code} 拉取失败: {last_error}")
    return pd.DataFrame()


def fetch_index_weights_monthly():
    weights = []
    success, failed = 0, []
    for ym in pd.period_range("2022-01", "2026-06", freq="M"):
        found = False
        for day in ["01", "03", "05", "08", "10", "15", "20", "25", "28"]:
            trade_date = f"{ym.year}{ym.month:02d}{day}"
            try:
                w = pro.index_weight(index_code=CONFIG["pool_code"], trade_date=trade_date)
                if w is not None and len(w) > 0:
                    w = w.copy()
                    w["trade_date"] = trade_date
                    weights.append(w)
                    success += 1
                    found = True
                    break
            except Exception:
                pass
        if not found:
            failed.append(str(ym))
    print(f"   成分股成功={success}月，失败={len(failed)}月")
    if failed:
        print(f"   失败样例={failed[:8]}")
    return pd.concat(weights, ignore_index=True) if weights else pd.DataFrame()


def build_pool(index_weights):
    index_weights = index_weights.copy()
    index_weights["trade_date"] = index_weights["trade_date"].astype(str)
    rebalance_dates = sorted(index_weights["trade_date"].unique())
    pool_map = index_weights.groupby("trade_date")["con_code"].apply(list).to_dict()

    def get_daily_pool(date_str):
        valid = [d for d in rebalance_dates if d <= date_str]
        return pool_map.get(max(valid), []) if valid else []

    return rebalance_dates, pool_map, get_daily_pool


def hurst(prices, max_lag=20):
    if len(prices) < max_lag:
        return 0.5
    lr = np.diff(np.log(np.maximum(prices, 0.01)))
    if len(lr) < max_lag:
        return 0.5
    eff = min(max_lag, len(lr) // 2)
    rs_values = []
    for lag in range(2, eff + 1):
        n_seg = len(lr) // lag
        if n_seg < 2:
            continue
        rs = []
        for seg in np.array_split(lr[: n_seg * lag], n_seg):
            if len(seg) < 2:
                continue
            adj = seg - np.mean(seg)
            r = np.max(np.cumsum(adj)) - np.min(np.cumsum(adj))
            s = np.std(seg, ddof=1)
            if s > 1e-10:
                rs.append(r / s)
        if rs:
            rs_values.append(np.mean(rs))
    if len(rs_values) < 3:
        return 0.5
    x = np.log(range(2, len(rs_values) + 2))
    y = np.log(rs_values)
    n = len(x)
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x) ** 2)
    return max(0, min(1, slope))


def beta_calc(stock_ret, index_ret):
    if len(stock_ret) < 20 or len(index_ret) < 20:
        return None, None
    n = min(len(stock_ret), len(index_ret))
    s = np.asarray(stock_ret[-n:])
    i = np.asarray(index_ret[-n:])
    cov = np.cov(s, i, ddof=1)
    if cov[1, 1] == 0:
        return 0, 0
    beta = cov[0, 1] / cov[1, 1]
    r2 = (cov[0, 1] ** 2) / (cov[0, 0] * cov[1, 1]) if cov[0, 0] > 0 else 0
    return beta, r2


def prepare_data():
    print("=" * 72)
    print("Dead Fish Twitch - research pipeline")
    print("=" * 72)
    print(
        f"参数: 仓位<{CONFIG['max_position_pct']:.0%} | score>{CONFIG['score_threshold']} | "
        f"vol_ratio>{CONFIG['min_vol_ratio']} | "
        f"vol_vol_ratio>{CONFIG['min_vol_vol_ratio']} | "
        f"Hurst<{CONFIG['max_hurst']} | Beta<{CONFIG['max_beta_abs']} | "
        f"近{CONFIG['recent_crash_days']}日单日跌幅>{CONFIG['recent_crash_threshold']:.0%}"
    )

    print("\n[1] 指数")
    df_index = get_or_cache(
        "index.csv",
        lambda: pro.index_daily(
            ts_code=CONFIG["index_code"], start_date=CONFIG["start_date"], end_date=CONFIG["end_date"]
        ).sort_values("trade_date").reset_index(drop=True),
    )
    df_index["trade_date"] = df_index["trade_date"].astype(str)
    df_index["pct_chg"] = df_index["close"].pct_change()

    print("\n[2] 沪深300股票池")
    index_weights = get_or_cache(
        "index_weights_v412_monthly.csv",
        fetch_index_weights_monthly,
        force=CONFIG["force_refresh_index_weights"],
    )
    if index_weights.empty:
        raise RuntimeError("没有拉到沪深300成分股数据，请检查 Tushare 权限。")
    rebalance_dates, pool_map, get_daily_pool = build_pool(index_weights)
    print(f"   调仓日期数={len(rebalance_dates)} 最早={rebalance_dates[0]} 最晚={rebalance_dates[-1]}")

    print("\n[3] 气象熵 + 市场状态")
    df_index["amplitude"] = (df_index["high"] - df_index["low"]) / df_index["pre_close"]
    ent = []
    for i in range(len(df_index)):
        if i < 3:
            ent.append(np.nan)
            continue
        sample = df_index["amplitude"].iloc[i - 3 : i].values
        hist, _ = np.histogram(sample, bins=8, density=True)
        hist = hist[hist > 0]
        ent.append(entropy(hist, base=np.e) if len(hist) > 1 else 0)
    df_index["weather_entropy"] = ent
    df_index = df_index.dropna(subset=["weather_entropy"]).reset_index(drop=True)
    df_index["entropy_threshold"] = df_index["weather_entropy"].rolling(30, min_periods=10).apply(
        lambda x: np.percentile(x, 65), raw=True
    )
    df_index["chaos_signal"] = df_index["weather_entropy"] >= df_index["entropy_threshold"]
    df_index["market_crash"] = df_index["pct_chg"] <= CONFIG["max_index_decline"]
    df_index["ma60"] = df_index["close"].rolling(60, min_periods=30).mean()
    df_index["ma60_slope"] = df_index["ma60"].pct_change(20)
    df_index["market_regime"] = "sideways"
    df_index.loc[df_index["ma60_slope"] > 0.03, "market_regime"] = "up"
    df_index.loc[df_index["ma60_slope"] < -0.03, "market_regime"] = "down"

    chaos_days = df_index[df_index["chaos_signal"]]["trade_date"].tolist()
    empty_days = sum(1 for d in chaos_days if len(get_daily_pool(d)) == 0)
    print(f"   混沌日={len(chaos_days)} 空池={empty_days} 暴跌日={int(df_index['market_crash'].sum())}")
    if empty_days > len(chaos_days) * 0.5:
        raise RuntimeError("空池天数过多，历史成分股覆盖仍然不完整。")

    print("\n[4] 股票基础信息")
    stock_basic = get_or_cache(
        "stock_basic.csv",
        lambda: pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,list_date"),
    )
    st_codes = set(stock_basic[stock_basic["name"].str.contains("ST|退", na=False)]["ts_code"].tolist())
    list_date_map = dict(zip(stock_basic["ts_code"], stock_basic["list_date"]))
    all_codes = sorted(set(c for codes in pool_map.values() for c in codes if c not in st_codes))
    print(f"   历史股票池={len(all_codes)}只")

    print("\n[5] 日线 + 前复权")

    adj_parts = []
    daily_parts = []
    short_daily = []
    empty_daily = []
    for i, code in enumerate(all_codes, 1):
        adj = fetch_stock_file(
            code,
            "adj_by_stock_v413",
            lambda c: pro.adj_factor(ts_code=c, start_date=CONFIG["start_date"], end_date=CONFIG["end_date"]),
            force=CONFIG["force_refresh_adj"],
        )
        daily = fetch_stock_file(
            code,
            "daily_by_stock_v413",
            lambda c: pro.daily(ts_code=c, start_date=CONFIG["start_date"], end_date=CONFIG["end_date"]),
            force=CONFIG["force_refresh_daily"],
            min_rows=CONFIG["hurst_window"],
        )
        if len(adj):
            adj_parts.append(adj)
        if len(daily):
            daily_parts.append(daily)
            if len(daily) < CONFIG["min_daily_rows_warn"]:
                short_daily.append((code, len(daily)))
        else:
            empty_daily.append(code)
        if i == 1 or i % 50 == 0 or i == len(all_codes):
            print(f"   逐股进度 {i}/{len(all_codes)} 日线累计={sum(len(x) for x in daily_parts)}")

    df_adj = pd.concat(adj_parts, ignore_index=True) if adj_parts else pd.DataFrame()
    df_raw = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    df_raw.to_csv(os.path.join(CONFIG["cache_dir"], "daily_raw_v413_full.csv"), index=False, encoding="utf-8-sig")
    df_adj.to_csv(os.path.join(CONFIG["cache_dir"], "adj_factors_v413_full.csv"), index=False, encoding="utf-8-sig")
    if df_adj.empty or df_raw.empty:
        raise RuntimeError("日线或复权因子为空，无法继续回测。")

    df_raw["trade_date"] = df_raw["trade_date"].astype(str)
    df_adj["trade_date"] = df_adj["trade_date"].astype(str)
    df_adj = df_adj.sort_values(["ts_code", "trade_date"])
    latest_adj = df_adj.groupby("ts_code")["adj_factor"].last().reset_index()
    latest_adj.columns = ["ts_code", "latest_adj_factor"]
    df_raw = pd.merge(df_raw, df_adj[["ts_code", "trade_date", "adj_factor"]], on=["ts_code", "trade_date"], how="left")
    df_raw = pd.merge(df_raw, latest_adj, on="ts_code", how="left")
    for col in ["open", "high", "low", "close", "pre_close"]:
        df_raw[col] = df_raw[col] * df_raw["adj_factor"].fillna(1) / df_raw["latest_adj_factor"].fillna(1)
    df_raw = df_raw.sort_values(["ts_code", "trade_date"])
    df_raw["pct_chg"] = df_raw.groupby("ts_code")["close"].pct_change()
    if "amount" not in df_raw.columns:
        df_raw["amount"] = df_raw["vol"] * df_raw["close"] / 1000
    code_count = df_raw["ts_code"].nunique()
    avg_rows = len(df_raw) / max(code_count, 1)
    print(f"   日线记录={len(df_raw)}条 覆盖股票={code_count}/{len(all_codes)} 平均每股={avg_rows:.0f}条")
    if empty_daily:
        print(f"   空日线股票={len(empty_daily)} 样例={empty_daily[:8]}")
    if short_daily:
        sample = ", ".join(f"{c}:{n}" for c, n in short_daily[:8])
        print(f"   日线偏少股票={len(short_daily)} 样例={sample}")

    stock_map = {
        code: part.sort_values("trade_date").set_index("trade_date", drop=False)
        for code, part in df_raw.groupby("ts_code", sort=False)
    }
    index_by_date = df_index.set_index("trade_date", drop=False)
    regime_map = dict(zip(df_index["trade_date"], df_index["market_regime"]))
    return df_index, index_by_date, stock_map, get_daily_pool, st_codes, list_date_map, regime_map


def get_stock_row(stock_map, code, trade_date):
    df = stock_map.get(code)
    if df is None or trade_date not in df.index:
        return None
    row = df.loc[trade_date]
    return row.iloc[-1] if isinstance(row, pd.DataFrame) else row


FUNNEL_STEPS = [
    ("new_stock", "1.新股/上市不足"),
    ("no_stock_data", "2.无个股数据"),
    ("not_enough_data", "3.数据不足"),
    ("suspended", "4.停牌/最新数据断档"),
    ("limit", "5.涨跌停"),
    ("three_day_decline", "6.3日跌幅不足"),
    ("ma20_break", "7.跌破MA20过深"),
    ("low_amount", "8.成交额不足"),
    ("recent_crash", "9.近5日暴跌"),
    ("merge_not_enough", "10.指数对齐不足"),
    ("beta_vol_r2", "11.Beta/波动/R2"),
    ("hurst", "12.Hurst过高"),
    ("vol_ratio", "13.波动/量比不足"),
    ("score", "14.评分不足"),
    ("pass", "通过"),
]

FUNNEL_LABELS = dict(FUNNEL_STEPS)


def evaluate_dead_fish(code, ref_date, stock_map, index_by_date, list_date_map):
    probe = {"date": ref_date, "stock_code": code}

    def fail(reason):
        return reason, None, probe

    list_date = list_date_map.get(code, "19900101")
    if pd.to_datetime(ref_date) - pd.to_datetime(list_date) < timedelta(days=CONFIG["min_list_days"]):
        return fail("new_stock")
    sdf = stock_map.get(code)
    if sdf is None:
        return fail("no_stock_data")
    data = sdf.loc[:ref_date].tail(CONFIG["hurst_window"] + 10)
    if len(data) < CONFIG["hurst_window"]:
        probe["bars"] = len(data)
        return fail("not_enough_data")
    if (pd.to_datetime(ref_date) - pd.to_datetime(data["trade_date"].iloc[-1])).days > 3:
        return fail("suspended")
    latest_pct = data["pct_chg"].iloc[-1]
    if pd.notna(latest_pct) and (latest_pct >= CONFIG["limit_up_pct"] or latest_pct <= CONFIG["limit_down_pct"]):
        probe["latest_pct"] = latest_pct
        return fail("limit")

    closes = data["close"].values
    ret_3d = closes[-1] / closes[-4] - 1 if len(closes) >= 4 else 0
    probe["ret_3d"] = ret_3d
    if ret_3d > CONFIG["min_3d_decline"]:
        return fail("three_day_decline")
    ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.nan
    probe["ma20_ratio"] = closes[-1] / ma20 if ma20 and not np.isnan(ma20) else np.nan
    if len(closes) >= 20 and closes[-1] < np.mean(closes[-20:]) * CONFIG["ma20_buffer"]:
        return fail("ma20_break")
    avg_amount = data["amount"].tail(5).mean()
    probe["avg_amount"] = avg_amount
    if avg_amount < CONFIG["min_avg_amount"]:
        return fail("low_amount")
    recent_min_ret = data["pct_chg"].tail(CONFIG["recent_crash_days"]).min()
    probe["recent_min_ret"] = recent_min_ret
    if pd.notna(recent_min_ret) and recent_min_ret <= CONFIG["recent_crash_threshold"]:
        return fail("recent_crash")

    idx = index_by_date[["pct_chg"]].reindex(data["trade_date"])
    merged = data[["trade_date", "pct_chg", "vol", "close"]].copy()
    merged["pct_chg_i"] = idx["pct_chg"].values
    merged = merged.rename(columns={"pct_chg": "pct_chg_s"}).dropna(subset=["pct_chg_s", "pct_chg_i"])
    if len(merged) < 20:
        probe["merged_bars"] = len(merged)
        return fail("merge_not_enough")
    beta, r2 = beta_calc(merged["pct_chg_s"].values, merged["pct_chg_i"].values)
    if beta is None:
        return fail("merge_not_enough")
    vol = np.std(merged["pct_chg_s"].values, ddof=1)
    probe.update({"beta": beta, "r2": r2, "vol": vol})
    if abs(beta) >= CONFIG["max_beta_abs"] or vol >= CONFIG["max_vol"] or r2 < CONFIG["min_r_squared"]:
        return fail("beta_vol_r2")

    prices = merged["close"].values[-CONFIG["hurst_window"] :]
    h = hurst(prices, min(20, CONFIG["hurst_window"] // 2))
    probe["hurst"] = h
    if h >= CONFIG["max_hurst"]:
        return fail("hurst")
    vol_short = np.std(np.diff(prices[-5:]), ddof=1) if len(prices) >= 5 else 0
    vol_long = np.std(np.diff(prices[-20:]), ddof=1) if len(prices) >= 20 else 0.01
    vol_ratio = vol_short / max(vol_long, 1e-10)
    avg_vol = merged["vol"].iloc[-6:-1].mean()
    vol_vol_ratio = merged["vol"].iloc[-1] / avg_vol if avg_vol > 0 else 0
    probe.update({"vol_ratio": vol_ratio, "vol_vol_ratio": vol_vol_ratio})
    if vol_ratio <= CONFIG["min_vol_ratio"] or vol_vol_ratio <= CONFIG["min_vol_vol_ratio"]:
        return fail("vol_ratio")

    score = (
        (CONFIG["max_hurst"] - h) * 8
        + min(vol_ratio, 3) * 1.2
        + min(vol_vol_ratio, 3) * 0.8
        - abs(beta) * 2
        - max(0, ret_3d + 0.01) * 20
    )
    probe["score"] = score
    if score < CONFIG["score_threshold"]:
        return fail("score")
    result = {
        "stock_code": code,
        "beta": beta,
        "r2": r2,
        "hurst": h,
        "hurst_tier": "strong" if h < 0.35 else "weak",
        "signal_close": closes[-1],
        "score": score,
        "ret_3d": ret_3d,
        "ma20_ratio": probe.get("ma20_ratio"),
        "vol_ratio": vol_ratio,
        "vol_vol_ratio": vol_vol_ratio,
        "recent_min_ret": recent_min_ret,
    }
    return "pass", result, probe


def find_dead_fish(code, ref_date, stock_map, index_by_date, list_date_map):
    _, result, _ = evaluate_dead_fish(code, ref_date, stock_map, index_by_date, list_date_map)
    return result


def evaluate_rank_candidate(code, ref_date, stock_map, index_by_date, list_date_map):
    reason, result, probe = evaluate_dead_fish(code, ref_date, stock_map, index_by_date, list_date_map)
    if result is not None:
        return result
    safe_reasons = {"hurst", "vol_ratio", "score"}
    if reason not in safe_reasons:
        return None

    sdf = stock_map.get(code)
    if sdf is None:
        return None
    data = sdf.loc[:ref_date].tail(CONFIG["hurst_window"] + 10)
    if len(data) < CONFIG["hurst_window"]:
        return None

    idx = index_by_date[["pct_chg"]].reindex(data["trade_date"])
    merged = data[["trade_date", "pct_chg", "vol", "close"]].copy()
    merged["pct_chg_i"] = idx["pct_chg"].values
    merged = merged.rename(columns={"pct_chg": "pct_chg_s"}).dropna(subset=["pct_chg_s", "pct_chg_i"])
    if len(merged) < 20:
        return None

    beta, r2 = beta_calc(merged["pct_chg_s"].values, merged["pct_chg_i"].values)
    if beta is None:
        return None
    vol = np.std(merged["pct_chg_s"].values, ddof=1)
    if abs(beta) >= CONFIG["max_beta_abs"] or vol >= CONFIG["max_vol"] or r2 < CONFIG["min_r_squared"]:
        return None

    prices = merged["close"].values[-CONFIG["hurst_window"] :]
    h = hurst(prices, min(20, CONFIG["hurst_window"] // 2))
    vol_short = np.std(np.diff(prices[-5:]), ddof=1) if len(prices) >= 5 else 0
    vol_long = np.std(np.diff(prices[-20:]), ddof=1) if len(prices) >= 20 else 0.01
    vol_ratio = vol_short / max(vol_long, 1e-10)
    avg_vol = merged["vol"].iloc[-6:-1].mean()
    vol_vol_ratio = merged["vol"].iloc[-1] / avg_vol if avg_vol > 0 else 0
    closes = data["close"].values
    ret_3d = closes[-1] / closes[-4] - 1 if len(closes) >= 4 else 0
    ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.nan
    recent_min_ret = data["pct_chg"].tail(CONFIG["recent_crash_days"]).min()
    score = (
        (CONFIG["max_hurst"] - h) * 8
        + min(vol_ratio, 3) * 1.2
        + min(vol_vol_ratio, 3) * 0.8
        - abs(beta) * 2
        - max(0, ret_3d + 0.01) * 20
    )
    return {
        "stock_code": code,
        "signal_date": ref_date,
        "signal_close": closes[-1],
        "score": score,
        "beta": beta,
        "r2": r2,
        "hurst": h,
        "ret_3d": ret_3d,
        "ma20_ratio": closes[-1] / ma20 if ma20 and not np.isnan(ma20) else np.nan,
        "vol_ratio": vol_ratio,
        "vol_vol_ratio": vol_vol_ratio,
        "recent_min_ret": recent_min_ret,
        "relaxed_reason": reason,
    }


def forward_return(code, ref_date, stock_map):
    sdf = stock_map.get(code)
    if sdf is None:
        return None
    future = sdf.loc[ref_date:]
    future = future[future["trade_date"] > ref_date]
    if len(future) < CONFIG["rank_forward_days"]:
        return None
    buy_row = future.iloc[0]
    sell_row = future.iloc[CONFIG["rank_forward_days"] - 1]
    if pd.notna(buy_row["pct_chg"]) and buy_row["pct_chg"] >= CONFIG["limit_up_pct"]:
        return None
    if pd.notna(sell_row["pct_chg"]) and sell_row["pct_chg"] <= CONFIG["limit_down_pct"]:
        return None
    buy_price = buy_row["open"] * (1 + CONFIG["slippage_buy"] + CONFIG["commission"])
    sell_price = sell_row["close"] * (1 - CONFIG["slippage_sell"] - CONFIG["stamp_tax"] - CONFIG["commission"])
    return {
        "buy_date": buy_row["trade_date"],
        "sell_date": sell_row["trade_date"],
        "buy_price": buy_price,
        "sell_price": sell_price,
        "forward_ret": sell_price / buy_price - 1,
    }


def forward_return_days(code, ref_date, stock_map, days):
    old_days = CONFIG["rank_forward_days"]
    CONFIG["rank_forward_days"] = days
    try:
        return forward_return(code, ref_date, stock_map)
    finally:
        CONFIG["rank_forward_days"] = old_days


def rank_validation(df_index, index_by_date, stock_map, get_daily_pool, st_codes, list_date_map, regime_map):
    print("\n" + "=" * 72)
    print("4.16 排序验证")
    print("=" * 72)
    rows = []
    scan_rows = df_index[(df_index["chaos_signal"]) & (~df_index["market_crash"])].copy()
    for day_idx, (_, row) in enumerate(scan_rows.iterrows()):
        date = row["trade_date"]
        candidates = []
        for code in get_daily_pool(date):
            if code in st_codes:
                continue
            candidate = evaluate_rank_candidate(code, date, stock_map, index_by_date, list_date_map)
            if candidate:
                candidates.append(candidate)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        for rank, item in enumerate(candidates[: CONFIG["rank_top_n"]], 1):
            fwd = forward_return(item["stock_code"], date, stock_map)
            if fwd is None:
                continue
            rows.append({**item, **fwd, "rank": rank, "market_regime": row["market_regime"]})
        if day_idx % 100 == 0:
            print(f"   排序验证进度 {day_idx}/{len(scan_rows)} 日期={date} 候选={len(candidates)}")

    result = pd.DataFrame(rows)
    if result.empty:
        print("排序验证无有效样本。")
        return result
    print(f"\n排序样本={len(result)} 未来{CONFIG['rank_forward_days']}日")
    for rank in range(1, CONFIG["rank_top_n"] + 1):
        sub = result[result["rank"] == rank]
        if len(sub):
            print(
                f"Top{rank}: 样本={len(sub)} 胜率={(sub['forward_ret'] > 0).mean():.1%} "
                f"平均={sub['forward_ret'].mean():.2%} 中位={sub['forward_ret'].median():.2%} "
                f"最差={sub['forward_ret'].min():.2%}"
            )
    print("\nTop样本收益最差:")
    cols = ["rank", "stock_code", "signal_date", "buy_date", "sell_date", "forward_ret", "score", "hurst", "beta", "r2", "ret_3d", "vol_ratio", "vol_vol_ratio", "relaxed_reason"]
    print(result[cols].sort_values("forward_ret").head(10).to_string(index=False))
    out_path = os.path.join(ROOT, "dead_fish_v416_rank_validation.csv")
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n已导出排序验证: dead_fish_v416_rank_validation.csv")
    return result


def tail_filter_validation(df_index, index_by_date, stock_map, get_daily_pool, st_codes, list_date_map, regime_map):
    print("\n" + "=" * 72)
    print("4.17 尾部过滤验证")
    print("=" * 72)
    base_rows = []
    scan_rows = df_index[(df_index["chaos_signal"]) & (~df_index["market_crash"])].copy()
    for day_idx, (_, row) in enumerate(scan_rows.iterrows()):
        date = row["trade_date"]
        candidates = []
        for code in get_daily_pool(date):
            if code in st_codes:
                continue
            candidate = evaluate_rank_candidate(code, date, stock_map, index_by_date, list_date_map)
            if candidate:
                candidates.append(candidate)
        if not candidates:
            continue
        candidates.sort(key=lambda x: x["score"], reverse=True)
        for item in candidates[: CONFIG["rank_top_n"]]:
            row_out = dict(item)
            for days in [3, 5, 7]:
                fwd = forward_return_days(item["stock_code"], date, stock_map, days)
                row_out[f"fwd_{days}d"] = np.nan if fwd is None else fwd["forward_ret"]
            base_rows.append(row_out)
        if day_idx % 100 == 0:
            print(f"   尾部过滤进度 {day_idx}/{len(scan_rows)} 日期={date} 候选={len(candidates)}")

    result = pd.DataFrame(base_rows)
    if result.empty:
        print("尾部过滤验证无样本。")
        return result
    score_cutoff = result["score"].quantile(CONFIG["tail_score_quantile"])
    high_score = result[result["score"] >= score_cutoff].copy()
    filtered = high_score[
        (high_score["vol_ratio"] < CONFIG["tail_max_vol_ratio"])
        & (high_score["hurst"] < CONFIG["tail_max_hurst"])
        & (high_score["ret_3d"] > CONFIG["tail_min_ret_3d"])
        & (high_score["recent_min_ret"] > CONFIG["tail_min_recent_ret"])
    ].copy()

    print(
        f"\n条件: 全局最高{int((1 - CONFIG['tail_score_quantile']) * 100)}%分数组(score>={score_cutoff:.3f}) | "
        f"vol_ratio<{CONFIG['tail_max_vol_ratio']} | Hurst<{CONFIG['tail_max_hurst']} | "
        f"ret_3d>{CONFIG['tail_min_ret_3d']:.0%} | recent_min_ret>{CONFIG['tail_min_recent_ret']:.0%}"
    )
    print(f"原Top样本={len(high_score)} 过滤后样本={len(filtered)}")
    for days in [3, 5, 7]:
        col = f"fwd_{days}d"
        base_sub = high_score.dropna(subset=[col])
        sub = filtered.dropna(subset=[col])
        if len(base_sub):
            print(
                f"{days}日-过滤前: 样本={len(base_sub)} 胜率={(base_sub[col] > 0).mean():.1%} "
                f"平均={base_sub[col].mean():.2%} 中位={base_sub[col].median():.2%} 最差={base_sub[col].min():.2%}"
            )
        if len(sub):
            print(
                f"{days}日-过滤后: 样本={len(sub)} 胜率={(sub[col] > 0).mean():.1%} "
                f"平均={sub[col].mean():.2%} 中位={sub[col].median():.2%} "
                f"最差={sub[col].min():.2%} 最好={sub[col].max():.2%}"
            )
    print("\n尾部过滤后最差样本:")
    cols = ["stock_code", "signal_date", "score", "hurst", "beta", "r2", "ret_3d", "recent_min_ret", "vol_ratio", "vol_vol_ratio", "fwd_3d", "fwd_5d", "fwd_7d", "relaxed_reason"]
    cols = [c for c in cols if c in result.columns]
    print(filtered.sort_values("fwd_5d")[cols].head(10).to_string(index=False))
    out_path = os.path.join(ROOT, "dead_fish_v417_tail_filter_validation.csv")
    filtered.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n已导出尾部过滤验证: dead_fish_v417_tail_filter_validation.csv")
    return result


def collect_tail_candidates(df_index, index_by_date, stock_map, get_daily_pool, st_codes, list_date_map):
    rows = []
    scan_rows = df_index[(df_index["chaos_signal"]) & (~df_index["market_crash"])].copy()
    for _, row in scan_rows.iterrows():
        date = row["trade_date"]
        candidates = []
        for code in get_daily_pool(date):
            if code in st_codes:
                continue
            candidate = evaluate_rank_candidate(code, date, stock_map, index_by_date, list_date_map)
            if candidate:
                candidates.append(candidate)
        if not candidates:
            continue
        candidates.sort(key=lambda x: x["score"], reverse=True)
        for item in candidates[: CONFIG["rank_top_n"]]:
            rows.append(dict(item))
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    score_cutoff = result["score"].quantile(CONFIG["tail_score_quantile"])
    result = result[result["score"] >= score_cutoff].copy()
    result = result[
        (result["vol_ratio"] < CONFIG["tail_max_vol_ratio"])
        & (result["hurst"] < CONFIG["tail_max_hurst"])
        & (result["ret_3d"] > CONFIG["tail_min_ret_3d"])
        & (result["recent_min_ret"] > CONFIG["tail_min_recent_ret"])
    ].copy()
    result["score_cutoff"] = score_cutoff
    return result.sort_values(["signal_date", "score"], ascending=[True, False]).reset_index(drop=True)


def tail_trade_validation(df_index, index_by_date, stock_map, get_daily_pool, st_codes, list_date_map, regime_map):
    print("\n" + "=" * 72)
    print("4.18 五日尾部过滤交易验证")
    print("=" * 72)
    candidates = collect_tail_candidates(df_index, index_by_date, stock_map, get_daily_pool, st_codes, list_date_map)
    if candidates.empty:
        print("尾部过滤交易验证无候选。")
        return pd.DataFrame(), pd.DataFrame()

    by_date = {d: part.to_dict("records") for d, part in candidates.groupby("signal_date", sort=False)}
    cash = CONFIG["initial_cash"]
    positions, trades, nav = [], [], []
    dates = df_index["trade_date"].tolist()

    for date in dates:
        for pos in list(positions):
            row_s = get_stock_row(stock_map, pos["stock"], date)
            if row_s is None:
                continue
            hold_days = (pd.to_datetime(date) - pd.to_datetime(pos["buy_date"])).days
            if hold_days >= CONFIG["tail_hold_days"]:
                if pd.notna(row_s["pct_chg"]) and row_s["pct_chg"] <= CONFIG["limit_down_pct"]:
                    continue
                sell_price = row_s["open"] * (1 - CONFIG["slippage_sell"] - CONFIG["stamp_tax"] - CONFIG["commission"])
                profit = sell_price / pos["buy_price"] - 1
                cash += sell_price * pos["shares"]
                trades.append({**pos, "sell_date": date, "sell_price": sell_price, "profit": profit, "reason": "tail_5d_exit", "hold_days": hold_days})
                positions.remove(pos)

        held = {p["stock"] for p in positions}
        if len(positions) < CONFIG["tail_trade_max_positions"]:
            for item in by_date.get(date, []):
                if len(positions) >= CONFIG["tail_trade_max_positions"]:
                    break
                code = item["stock_code"]
                if code in held:
                    continue
                sdf = stock_map.get(code)
                if sdf is None:
                    continue
                future = sdf[sdf["trade_date"] > date]
                if len(future) == 0:
                    continue
                row_buy = future.iloc[0]
                buy_date = row_buy["trade_date"]
                if row_buy is None:
                    continue
                if pd.notna(row_buy["pct_chg"]) and row_buy["pct_chg"] >= CONFIG["limit_up_pct"]:
                    continue
                buy_price = row_buy["open"] * (1 + CONFIG["slippage_buy"] + CONFIG["commission"])
                shares = int(cash * CONFIG["max_position_pct"] / buy_price / 100) * 100
                if shares < 100 or buy_price * shares > cash:
                    continue
                cash -= buy_price * shares
                positions.append({
                    "stock": code,
                    "signal_date": date,
                    "buy_date": buy_date,
                    "buy_price": buy_price,
                    "shares": shares,
                    "signal_score": item["score"],
                    "hurst": item["hurst"],
                    "signal_beta": item["beta"],
                    "signal_r2": item["r2"],
                    "signal_ret_3d": item["ret_3d"],
                    "signal_recent_min_ret": item["recent_min_ret"],
                    "signal_vol_ratio": item["vol_ratio"],
                    "signal_vol_vol_ratio": item["vol_vol_ratio"],
                })
                held.add(code)

        position_value = 0
        for pos in positions:
            row_s = get_stock_row(stock_map, pos["stock"], date)
            if row_s is not None:
                position_value += row_s["close"] * pos["shares"]
        nav.append({"date": date, "nav": cash + position_value, "cash": cash, "position_value": position_value, "positions": len(positions)})

    last_date = dates[-1]
    for pos in list(positions):
        row_s = get_stock_row(stock_map, pos["stock"], last_date)
        if row_s is None:
            continue
        sell_price = row_s["close"] * (1 - CONFIG["slippage_sell"] - CONFIG["stamp_tax"] - CONFIG["commission"])
        profit = sell_price / pos["buy_price"] - 1
        cash += sell_price * pos["shares"]
        hold_days = (pd.to_datetime(last_date) - pd.to_datetime(pos["buy_date"])).days
        trades.append({**pos, "sell_date": last_date, "sell_price": sell_price, "profit": profit, "reason": "force_close", "hold_days": hold_days})
    nav.append({"date": f"{last_date}_close", "nav": cash, "cash": cash, "position_value": 0, "positions": 0})

    df_trades = pd.DataFrame(trades)
    df_nav = pd.DataFrame(nav)
    if df_trades.empty:
        print("尾部过滤交易验证无成交。")
        return df_trades, df_nav

    df_nav["peak"] = df_nav["nav"].cummax()
    df_nav["drawdown"] = df_nav["nav"] / df_nav["peak"] - 1
    df_nav["daily_ret"] = df_nav["nav"].pct_change()
    final_nav = df_nav["nav"].iloc[-1]
    total_return = final_nav / CONFIG["initial_cash"] - 1
    sharpe = df_nav["daily_ret"].mean() / df_nav["daily_ret"].std() * np.sqrt(252) if df_nav["daily_ret"].std() > 0 else 0
    print(
        f"候选={len(candidates)} 交易={len(df_trades)} 胜率={(df_trades['profit'] > 0).mean():.1%} "
        f"平均={df_trades['profit'].mean():.2%} 中位={df_trades['profit'].median():.2%}"
    )
    print(f"净值={final_nav:,.0f} 收益={total_return:.2%} 夏普={sharpe:.2f} 最大回撤={df_nav['drawdown'].min():.2%}")
    print("\n尾部交易最差:")
    cols = ["stock", "signal_date", "buy_date", "sell_date", "hold_days", "profit", "signal_score", "hurst", "signal_ret_3d", "signal_vol_ratio", "signal_vol_vol_ratio"]
    print(df_trades[cols].sort_values("profit").head(10).to_string(index=False))
    df_trades.to_csv(os.path.join(ROOT, "dead_fish_v418_tail_trades.csv"), index=False, encoding="utf-8-sig")
    df_nav.to_csv(os.path.join(ROOT, "dead_fish_v418_tail_nav.csv"), index=False, encoding="utf-8-sig")
    print("\n已导出五日交易验证: dead_fish_v418_tail_trades.csv / dead_fish_v418_tail_nav.csv")
    return df_trades, df_nav


def run_backtest(df_index, index_by_date, stock_map, get_daily_pool, st_codes, list_date_map, regime_map):
    print("\n[6] 回测")
    cash = CONFIG["initial_cash"]
    positions, pending, trades, nav = [], [], [], []
    filtered = {"gap_up": 0, "limit": 0, "crash": 0, "buy_expired": 0, "sell_limit_down": 0}
    last_date = df_index["trade_date"].iloc[-1]

    for idx, row in df_index.iterrows():
        date = row["trade_date"]
        regime = row["market_regime"]
        if idx % 100 == 0:
            print(f"   进度 {idx}/{len(df_index)} 日期={date} 持仓={len(positions)} 待成交={len(pending)}")

        for order in [o for o in pending if o["type"] == "buy" and (pd.to_datetime(date) - pd.to_datetime(o["add_date"])).days > CONFIG["pending_buy_expire_days"]]:
            pending.remove(order)
            filtered["buy_expired"] += 1

        for order in list(pending):
            row_s = get_stock_row(stock_map, order["stock"], date)
            if row_s is None:
                continue
            if order["type"] == "sell":
                if pd.notna(row_s["pct_chg"]) and row_s["pct_chg"] <= CONFIG["limit_down_pct"]:
                    filtered["sell_limit_down"] += 1
                    continue
                sell_price = row_s["open"] * (1 - CONFIG["slippage_sell"] - CONFIG["stamp_tax"] - CONFIG["commission"])
                for pos in list(positions):
                    if pos["stock"] == order["stock"] and pos["batch"] == order["batch"]:
                        profit = (sell_price - pos["buy_price"]) / pos["buy_price"]
                        cash += sell_price * pos["shares"]
                        hold_days = (pd.to_datetime(date) - pd.to_datetime(pos["buy_date"])).days
                        trades.append({**pos, "sell_date": date, "sell_price": sell_price, "profit": profit, "reason": order["reason"], "sell_regime": regime, "hold_days": hold_days})
                        positions.remove(pos)
                        break
                pending.remove(order)
            else:
                buy_open = row_s["open"]
                if order["batch"] == 1 and buy_open / order["signal_close"] - 1 > CONFIG["max_gap_up"]:
                    filtered["gap_up"] += 1
                    pending.remove(order)
                    continue
                if pd.notna(row_s["pct_chg"]) and row_s["pct_chg"] >= CONFIG["limit_up_pct"]:
                    filtered["limit"] += 1
                    pending.remove(order)
                    continue
                buy_price = buy_open * (1 + CONFIG["slippage_buy"] + CONFIG["commission"])
                shares = max(100, int(cash * CONFIG["max_position_pct"] / buy_price / 100) * 100) if order["batch"] == 1 else order.get("shares", 100)
                if buy_price * shares <= cash:
                    cash -= buy_price * shares
                    positions.append({
                        "stock": order["stock"], "batch": order["batch"], "buy_date": date,
                        "buy_price": buy_price, "shares": shares, "hurst": order.get("hurst"),
                        "hurst_tier": order.get("hurst_tier"), "signal_regime": order.get("signal_regime", "unknown"),
                        "signal_date": order.get("add_date"), "signal_score": order.get("score"),
                        "signal_beta": order.get("beta"), "signal_r2": order.get("r2"),
                        "signal_ret_3d": order.get("ret_3d"), "signal_ma20_ratio": order.get("ma20_ratio"),
                        "signal_vol_ratio": order.get("vol_ratio"), "signal_vol_vol_ratio": order.get("vol_vol_ratio"),
                        "signal_recent_min_ret": order.get("recent_min_ret"),
                    })
                pending.remove(order)

        for pos in list(positions):
            row_s = get_stock_row(stock_map, pos["stock"], date)
            if row_s is None:
                continue
            close = row_s["close"]
            pnl = (close - pos["buy_price"]) / pos["buy_price"]
            hold_days = (pd.to_datetime(date) - pd.to_datetime(pos["buy_date"])).days
            tp = CONFIG["batch1_take_profit"] if pos["batch"] == 1 else CONFIG["batch2_take_profit"]
            sl = CONFIG["batch1_stop_loss"] if pos["batch"] == 1 else CONFIG["batch2_stop_loss"]
            mh = CONFIG["batch1_max_hold"] if pos["batch"] == 1 else CONFIG["batch2_max_hold"]
            reason = "take_profit" if pnl >= tp else "stop_loss" if pnl <= sl else "max_hold" if hold_days >= mh else None
            if reason is None:
                same = [p for p in positions if p["stock"] == pos["stock"]]
                cost = sum(p["buy_price"] * p["shares"] for p in same)
                value = close * sum(p["shares"] for p in same)
                if cost > 0 and value / cost - 1 <= CONFIG["unified_stop_loss"]:
                    reason = "unified_stop_loss"
                    for sp in same:
                        if not any(o["type"] == "sell" and o["stock"] == sp["stock"] and o["batch"] == sp["batch"] for o in pending):
                            pending.append({"type": "sell", "stock": sp["stock"], "batch": sp["batch"], "reason": reason, "add_date": date})
            if reason and reason != "unified_stop_loss":
                if not any(o["type"] == "sell" and o["stock"] == pos["stock"] and o["batch"] == pos["batch"] for o in pending):
                    pending.append({"type": "sell", "stock": pos["stock"], "batch": pos["batch"], "reason": reason, "add_date": date})

        for pos in positions:
            if pos["batch"] != 1:
                continue
            row_s = get_stock_row(stock_map, pos["stock"], date)
            if row_s is None:
                continue
            pnl = (row_s["close"] - pos["buy_price"]) / pos["buy_price"]
            if pnl <= CONFIG["batch1_trigger_batch2"]:
                b2_count = sum(1 for p in positions if p["stock"] == pos["stock"] and p["batch"] == 2)
                b2_pending = sum(1 for o in pending if o["type"] == "buy" and o["stock"] == pos["stock"] and o["batch"] == 2)
                if b2_count + b2_pending >= CONFIG["max_batch2_per_stock"]:
                    continue
                hist = stock_map[pos["stock"]].loc[:date].tail(20)
                if len(hist) >= 20 and row_s["close"] < hist["close"].mean() * 0.9:
                    continue
                shares = max(100, int(pos["shares"] * 0.5 / 100) * 100)
                pending.append({"type": "buy", "stock": pos["stock"], "batch": 2, "shares": shares, "hurst": pos["hurst"], "hurst_tier": pos["hurst_tier"], "add_date": date, "signal_regime": regime, "score": pos.get("signal_score"), "beta": pos.get("signal_beta"), "r2": pos.get("signal_r2"), "ret_3d": pos.get("signal_ret_3d"), "ma20_ratio": pos.get("signal_ma20_ratio"), "vol_ratio": pos.get("signal_vol_ratio"), "vol_vol_ratio": pos.get("signal_vol_vol_ratio"), "recent_min_ret": pos.get("signal_recent_min_ret")})

        excluded = {p["stock"] for p in positions} | {o["stock"] for o in pending}
        if row["chaos_signal"] and len({p["stock"] for p in positions}) + sum(1 for o in pending if o["type"] == "buy" and o["batch"] == 1) < CONFIG["max_positions"] and not row["market_crash"]:
            candidates = []
            for code in get_daily_pool(date):
                if code in excluded or code in st_codes:
                    continue
                result = find_dead_fish(code, date, stock_map, index_by_date, list_date_map)
                if result:
                    candidates.append(result)
            candidates.sort(key=lambda x: x["score"], reverse=True)
            if candidates:
                best = candidates[0]
                pending.append({"type": "buy", "stock": best["stock_code"], "batch": 1, "signal_close": best["signal_close"], "hurst": best["hurst"], "hurst_tier": best["hurst_tier"], "add_date": date, "signal_regime": regime, "score": best.get("score"), "beta": best.get("beta"), "r2": best.get("r2"), "ret_3d": best.get("ret_3d"), "ma20_ratio": best.get("ma20_ratio"), "vol_ratio": best.get("vol_ratio"), "vol_vol_ratio": best.get("vol_vol_ratio"), "recent_min_ret": best.get("recent_min_ret")})
        elif row.get("market_crash", False):
            filtered["crash"] += 1

        position_value = sum((get_stock_row(stock_map, p["stock"], date)["close"] * p["shares"]) for p in positions if get_stock_row(stock_map, p["stock"], date) is not None)
        nav.append({"date": date, "nav": cash + position_value, "cash": cash, "position_value": position_value, "pending": len(pending), "market_regime": regime})

    if positions:
        for pos in positions:
            row_s = get_stock_row(stock_map, pos["stock"], last_date)
            if row_s is None:
                continue
            sell_price = row_s["close"] * (1 - CONFIG["slippage_sell"] - CONFIG["stamp_tax"] - CONFIG["commission"])
            cash += sell_price * pos["shares"]
            hold_days = (pd.to_datetime(last_date) - pd.to_datetime(pos["buy_date"])).days
            trades.append({**pos, "sell_date": last_date, "sell_price": sell_price, "profit": (sell_price - pos["buy_price"]) / pos["buy_price"], "reason": "force_close", "sell_regime": regime_map.get(last_date, "unknown"), "hold_days": hold_days})
    nav.append({"date": f"{last_date}_close", "nav": cash, "cash": cash, "position_value": 0, "pending": 0, "market_regime": "closed"})
    return pd.DataFrame(trades), pd.DataFrame(nav), filtered


def diagnose_funnel(df_index, index_by_date, stock_map, get_daily_pool, st_codes, list_date_map, regime_map):
    print("\n" + "=" * 72)
    print("4.12 漏斗诊断")
    print("=" * 72)
    counts = {key: 0 for key, _ in FUNNEL_STEPS}
    total_scan = 0
    chaos_days = 0
    skipped_empty_pool = 0
    near_misses = []

    scan_rows = df_index[(df_index["chaos_signal"]) & (~df_index["market_crash"])].copy()
    for day_idx, (_, row) in enumerate(scan_rows.iterrows()):
        date = row["trade_date"]
        pool = [code for code in get_daily_pool(date) if code not in st_codes]
        if not pool:
            skipped_empty_pool += 1
            continue
        chaos_days += 1
        if day_idx % 100 == 0:
            print(f"   诊断进度 {day_idx}/{len(scan_rows)} 日期={date} 股票池={len(pool)}")
        for code in pool:
            total_scan += 1
            reason, result, probe = evaluate_dead_fish(code, date, stock_map, index_by_date, list_date_map)
            counts[reason] = counts.get(reason, 0) + 1
            if reason != "pass":
                probe["淘汰层"] = FUNNEL_LABELS.get(reason, reason)
                near_misses.append(probe)

    print(f"\n混沌可扫描日={chaos_days} 空池={skipped_empty_pool} 扫描={total_scan}次")
    remaining = total_scan
    print("\n层级".ljust(20) + "淘汰".rjust(8) + "占总量".rjust(10) + "层级淘汰率".rjust(12) + "剩余".rjust(8))
    print("-" * 62)
    for key, label in FUNNEL_STEPS:
        if key == "pass":
            continue
        killed = counts.get(key, 0)
        layer_rate = killed / remaining if remaining else np.nan
        remaining -= killed
        total_rate = killed / total_scan if total_scan else 0
        layer_text = "N/A" if np.isnan(layer_rate) else f"{layer_rate:.1%}"
        print(f"{label:<20}{killed:>8}{total_rate:>10.1%}{layer_text:>12}{remaining:>8}")
    print("-" * 62)
    print(f"{'最终候选':<20}{counts.get('pass', 0):>8}{(counts.get('pass', 0) / total_scan if total_scan else 0):>10.1%}")

    killers = [(FUNNEL_LABELS.get(k, k), v) for k, v in counts.items() if k != "pass" and v > 0]
    killers = sorted(killers, key=lambda x: x[1], reverse=True)[:5]
    if killers:
        print("\n主要杀手:")
        for i, (label, value) in enumerate(killers, 1):
            print(f"   {i}. {label}: {value}次")

    if near_misses:
        nm = pd.DataFrame(near_misses)
        metric_cols = [c for c in ["date", "stock_code", "淘汰层", "ret_3d", "recent_min_ret", "ma20_ratio", "beta", "r2", "hurst", "vol_ratio", "vol_vol_ratio", "score"] if c in nm.columns]
        for col in ["score", "vol_ratio", "vol_vol_ratio", "r2"]:
            if col not in nm.columns:
                nm[col] = np.nan
        nm["_rank"] = (
            nm["score"].fillna(-99) * 2
            + nm["vol_ratio"].fillna(0)
            + nm["vol_vol_ratio"].fillna(0)
            + nm["r2"].fillna(0)
        )
        top = nm.sort_values("_rank", ascending=False).head(CONFIG["near_miss_top_n"])
        print(f"\n差一点入选的前{len(top)}个样本:")
        print(top[[c for c in metric_cols if c in top.columns]].to_string(index=False))
        out_path = os.path.join(ROOT, "dead_fish_v412_near_misses.csv")
        top.drop(columns=["_rank"], errors="ignore").to_csv(out_path, index=False, encoding="utf-8-sig")
        print("\n已导出近似候选: dead_fish_v412_near_misses.csv")


def summarize(trades, nav, filtered):
    print("\n" + "=" * 72)
    print("Dead Fish Twitch - strict signal backtest")
    print("=" * 72)
    if trades.empty:
        print("无交易信号。")
        print(f"交易阶段过滤器={filtered}")
        return
    if "hold_days" not in trades.columns:
        trades["hold_days"] = (pd.to_datetime(trades["sell_date"]) - pd.to_datetime(trades["buy_date"])).dt.days
    nav["peak"] = nav["nav"].cummax()
    nav["drawdown"] = nav["nav"] / nav["peak"] - 1
    nav["daily_ret"] = nav["nav"].pct_change()
    final_nav = nav["nav"].iloc[-1]
    total_return = final_nav / CONFIG["initial_cash"] - 1
    sharpe = nav["daily_ret"].mean() / nav["daily_ret"].std() * np.sqrt(252) if nav["daily_ret"].std() > 0 else 0
    avg_win = trades[trades["profit"] > 0]["profit"].mean()
    avg_loss = abs(trades[trades["profit"] <= 0]["profit"].mean())
    payoff = avg_win / avg_loss if avg_loss > 0 else np.nan
    print(f"最终净值={final_nav:,.0f} 收益={total_return:.2%} 夏普={sharpe:.2f} 最大回撤={nav['drawdown'].min():.2%}")
    print(f"交易={len(trades)}次 胜率={(trades['profit'] > 0).mean():.2%} 平均收益={trades['profit'].mean():.2%} 盈亏比={payoff:.2f}")
    print(f"交易阶段过滤器={filtered}")
    print("\n按市场状态:")
    regime_name = {"up": "上涨", "sideways": "震荡", "down": "下跌", "unknown": "未知"}
    for regime in ["up", "sideways", "down"]:
        sub = trades[trades["signal_regime"] == regime]
        if len(sub):
            print(f"  {regime_name.get(regime, regime)}: 交易={len(sub)}次 胜率={(sub['profit'] > 0).mean():.1%} 平均={sub['profit'].mean():.2%}")
    print("\n按批次:")
    for batch in [1, 2]:
        sub = trades[trades["batch"] == batch]
        if len(sub):
            print(f"  第{batch}批: 交易={len(sub)}次 胜率={(sub['profit'] > 0).mean():.1%} 平均={sub['profit'].mean():.2%}")

    detail_cols = [
        "stock", "batch", "signal_date", "buy_date", "sell_date", "hold_days",
        "buy_price", "sell_price", "profit", "reason", "signal_regime",
        "hurst", "signal_score", "signal_beta", "signal_r2", "signal_ret_3d",
        "signal_ma20_ratio", "signal_vol_ratio", "signal_vol_vol_ratio", "signal_recent_min_ret",
    ]
    detail_cols = [c for c in detail_cols if c in trades.columns]
    detail = trades[detail_cols].copy().sort_values("profit")
    print("\n交易明细-亏损最深:")
    print(detail.head(10).to_string(index=False))
    big_loss = trades[trades["profit"] <= -0.05].copy()
    if len(big_loss):
        print(f"\n大亏交易诊断: {len(big_loss)}笔亏损超过5%，需要重点检查买点和隔日卖出机制。")
        print(big_loss[detail_cols].sort_values("profit").to_string(index=False))


if __name__ == "__main__":
    data = prepare_data()
    df_trades, df_nav, filtered = run_backtest(*data)
    summarize(df_trades, df_nav, filtered)
    if CONFIG["run_rank_validation"]:
        rank_validation(*data)
    if CONFIG["run_tail_filter_validation"]:
        tail_filter_validation(*data)
    if CONFIG["run_tail_trade_validation"]:
        tail_trade_validation(*data)
    if CONFIG["run_funnel_diagnosis"]:
        diagnose_funnel(*data)
    df_trades.to_csv(os.path.join(ROOT, "dead_fish_v412_optimized_trades.csv"), index=False, encoding="utf-8-sig")
    df_nav.to_csv(os.path.join(ROOT, "dead_fish_v412_optimized_nav.csv"), index=False, encoding="utf-8-sig")
    print("\n已导出 dead_fish_v412_optimized_trades.csv / dead_fish_v412_optimized_nav.csv")
