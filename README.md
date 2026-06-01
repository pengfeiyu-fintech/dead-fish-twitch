对，我改的是你电脑 work 文件夹里的本地 `README.md`，还没有自动改 GitHub 上的 README。

你**不用删除 GitHub 原文件再上传**，最简单是直接在 GitHub 网页编辑：

1. 打开你的仓库 README。
2. 点右上角铅笔图标。
3. 全选原来的 README 内容。
4. 用下面这份内容全部替换。
5. Commit message 写：
   ```text
   Expand research design section
   ```
6. 点 `Commit changes`。

直接复制这份：

```md
# Dead Fish Twitch

An experimental A-share mean-reversion research project.

This project explores a short-term oversold rebound idea: under chaotic market conditions, certain ignored or weakly traded stocks may show a brief "twitch" before continuing their prior path.

The final result is intentionally conservative:

> The original trading hypothesis did not survive empirical validation. The research pipeline did.

## What This Project Does

- Fetches A-share index, constituent, adjustment-factor, and daily-bar data with Tushare.
- Repairs historical CSI 300 constituent coverage by fetching index weights month by month.
- Normalizes return units to avoid mixing percentage and decimal `pct_chg` fields.
- Builds an event-style backtest with fees, stamp tax, slippage, limit-up/down checks, position sizing, and holding-period exits.
- Adds funnel diagnostics to identify which filters eliminate candidate signals.
- Runs ranking validation to test whether the score has predictive power.
- Runs tail-filter validation and a fixed-5-day trade validation.

## Key Research Design

This repository is not just a backtest script. It is a research pipeline built to test, decompose, and eventually reject a trading hypothesis in a reproducible way.

### 1. Market Regime Classification via Weather Entropy

Instead of using only index return, moving averages, or volume, the project defines a market "chaos" signal with the entropy of recent intraday amplitude distribution.

The idea is that mean-reversion opportunities, if they exist, may be more likely to appear when the market structure is noisy rather than clearly trending. A `chaos_signal` is triggered when weather entropy exceeds its rolling threshold.

### 2. Dead Fish Composite Score

The strategy scores each candidate with a composite signal designed to describe an "ignored, weak, temporarily oversold" state:

- **Hurst exponent**: lower values indicate stronger mean-reversion tendency.
- **Volatility regime shift**: short-term volatility relative to longer-term volatility.
- **Volume shock**: current volume relative to recent average volume.
- **Index beta**: penalizes candidates that are too strongly explained by market movement.
- **3-day decline**: captures short-term oversold pressure without blindly buying every falling stock.

This score was later tested independently through rank validation instead of relying only on full-system backtest results.

### 3. Two-Stage Position Logic with Unified Stop Loss

The early strategy design used a two-batch entry mechanism:

- **Batch 1**: initial entry after a signal.
- **Batch 2**: conditional add-on after further decline.
- **Unified stop loss**: portfolio-level protection for the same stock position.

This was an attempt to balance a mean-reversion assumption with risk control. The experiment also showed why this kind of "martingale-lite" structure can be dangerous when the signal is weak.

### 4. Funnel Diagnostics

Every candidate passes through a multi-stage diagnostic funnel:

1. listing age and data availability,
2. suspension and limit-up/down checks,
3. recent decline and moving-average filters,
4. liquidity checks,
5. beta, volatility, and R-squared filters,
6. Hurst, volume, and score filters.

The funnel reports how many candidates are eliminated at each stage, making it possible to identify whether the strategy fails because of data coverage, overly strict filters, or lack of genuine signal.

### 5. Three-Level Validation Framework

The project separates factor research from trade execution:

- **Rank validation**: tests whether the score predicts forward returns before applying full trading logic.
- **Tail-filter validation**: checks whether large-loss candidates can be identified and removed.
- **Trade validation**: applies costs, slippage, position limits, and fixed holding periods.

This separation is the main lesson of the project: a weak statistical trace may appear in factor tests, but still disappear after execution constraints and tail losses.

## Research Conclusion

The strict mean-reversion version produced too few tradable signals. Relaxing the filters created more candidates, but the signal was too weak to cover trading costs and tail losses.

In the final trade validation, the strategy remained negative after applying:

- high-score selection,
- tail-risk filters,
- fixed 5-day holding,
- position limits,
- fees, tax, and slippage.

This repository is therefore a research case study, not a profitable trading system.

## Requirements

```bash
pip install -r requirements.txt
```

Set your Tushare token before running:

```bash
set TUSHARE_TOKEN=your_token_here
```

PowerShell:

```powershell
$env:TUSHARE_TOKEN="your_token_here"
```

## Run

```bash
python dead_fish_twitch_research.py
```

The script writes cache and result files locally. These files are intentionally ignored by Git.

## Notes

- This is not financial advice.
- This is not intended for live trading.
- Tushare data permissions and rate limits may affect reproducibility.
- The old exploratory notebook may contain local credentials and should not be committed.
```
