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

## Suggested Resume Framing

Built an A-share quantitative research pipeline using Python, Pandas, and Tushare; repaired historical constituent data, implemented backtest execution constraints, built signal funnel diagnostics, and validated a failed mean-reversion hypothesis through ranking and tail-risk analysis.
