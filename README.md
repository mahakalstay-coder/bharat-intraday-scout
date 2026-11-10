# Bharat Intraday Scout 3.1 — Repaired Adaptive Paper-Validation Edition

**Read this first.** This is a patched version of the uploaded `bharat-intraday-scout-3 (1).zip`. It is an exploratory **paper research and notification tool**, not a profitable or guaranteed trading system. It does not execute broker orders.

## What was broken and repaired

The uploaded `scanner.py` had an incorrectly merged `add_signal()` function containing the scan-cycle code, but was missing separate `run_scan_cycle()`, `deliver_pending_signals()`, `send_telegram_message()`, and `rank_pre_explosion_candidates()` functions. Running the continuous scanner would have failed. These functions are restored and covered by a smoke test. Telegram delivery uses deduplication and suppresses old unvalidated pending signals.

The original ZIP had **24 stored historical signals and zero resolved outcomes**. This remains an unproven strategy. Existing scanner paper records are retained in this archive as supplied; if your GitHub repo has a newer database, **retain the newer database**, not this older one.

## Automated workflow and learning

1. During NSE market hours the scanner continues to sample the selected liquid NSE universe (200 stocks by default), evaluate 6 technical signal formulas, check the latest volume, evaluate Kronos-mini if available, fetch limited news headlines, and optionally consult the Tauric API if a key is available.
2. A *pre-breakout research watchlist* logs the top 10 daily near-resistance/support compression setups. These are **not trade alerts**. End-of-day studies their later maximum favorable price excursion, subject to complete 5-minute data.
3. Evaluated candidate trades (subject to the per-cycle processing limit) become permanent paper records, with an entry context including sector/industry, peers, six matched technical formulas, volume, Kronos forecast and a label indicating whether it was approved or was research-only.
4. The 15:35 IST end-of-day task grades paper paths with conservative same-bar stop-first rules, and excludes incomplete/no-data sessions. The new journal also scores Kronos 30-minute forecast direction and absolute error against later bars, without treating missing data as a wrong prediction.
5. Six formula weights are adapted only after **20 trading sessions** with a 70/30 chronological session split, at least 30 training and 15 holdout observations per formula, with conservative criteria. Single-day replay weights remain diagnostic.
6. The new journal evaluates *indicators and context*, including RSI, MFI, ADX, same-slot and session RVOL, Bollinger width, industry peers, Kronos alignment and time of entry. Indicators are described as **associated with** outcomes, not assumed to cause them.
7. The journal may suggest a higher minimum same-time RVOL floor (2.0x or 2.2x) but only when learned on the training partition and confirmed to improve net paper results on the held-out partition. Otherwise 1.8x remains the validated-alert floor.
8. **During the learning phase, no candidate trade alerts are sent.** Signals are saved as shadow paper trades (`delivered=2`) and will not be sent retroactively. Daily learning reports and a morning status message can still be sent.
9. Only when *every* condition below passes does `VALIDATED_PAPER_MODE` begin. Even then, only setups with the additional strong-signal gate result in a research alert; others continue as shadow observations.

### Promotion conditions (all required)

- At least **14 distinct sessions with resolved paper trades** (not simply calendar days); configurable via `OBSERVATION_MIN_SESSIONS`.
- At least **50 resolved trades with the transaction-cost assumption frozen at entry**, controlled by `ACTIVATION_MIN_TRADES`.
- A chronological last-30%-of-sessions holdout containing at least **15 trades**, with mean estimated net R >= **+0.10R**, and win rate >= **40%**.
- At least **20 scored Kronos forecasts** with directional hit rate >= **53%**; scores use the stated 30-minute horizon, and omit forecasts that extend past the valid market session or have missing candles.
- Each delivered setup must be Tier-1, score >= **85/100**, same-slot RVOL >= learned floor (initially 1.8x), cumulative RVOL >= 1.3x, Kronos aligned, and its sector/peer snapshot must agree directionally with sufficient sampled coverage.

14 sessions is a **review point, NOT a countdown to guaranteed accuracy**. Promotion may take far longer or never happen. If validation deteriorates, the gate reverts automatically to `RESEARCH_ONLY`. Profitability and live execution quality were **not** tested.

### Files and reports

- `scanner.py` — fixed live scanning orchestration; paper storage, Telegram, Kronos inference and original indicator rules.
- `market_learning.py` — existing sampled-sector heatmap, frozen entry paper cost model, and conservative validated six-formula weights.
- `adaptive_journal.py` — new shadow watch studies, Kronos forecast error tracking, feature outcome breakdowns, volume threshold holdout calibration, and evidence-based promotion gate.
- `runtime/bis20.sqlite3` — SQLite signal, outcome, watch, snapshot, and learning database (preserve the most recent live database).
- `reports/heatmap.html` — offline *sampled* sector/industry breadth, not an exchange or tick-level heatmap.
- `reports/adaptive_report.json` — machine-readable activation checks, historical indicator outcome associations, last 25 paper-trade lesson tags, and watch and Kronos outcomes.
- `reports/paper_trades.csv` / `performance.json` — existing ledger and cost-adjusted summary.
- `test_adaptive_journal.py` / `test_market_learning.py` — offline tests.
- `README-preaudit-notes.md` — the supplied package's earlier notes, preserved for reference.

## Install/replace the existing repo in ONE upload

1. **Back up your existing repo and `runtime/bis20.sqlite3`** if it has more recent signals than the ZIP. Be careful not to expose this database in a public repo.
2. Extract this ZIP and copy **the entire contents** of this directory to your existing GitHub repository **root**, keeping the `.github/workflows/bis20.yml` path unchanged. Commit once.
3. Keep the existing GitHub Actions secrets `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`. Keep `TAURIC_API_KEY` only if desired; an external AI API may incur charges.
4. In GitHub Actions, manually run `refresh-universe`, followed by `refresh-metadata` several times to build sector/industry coverage (50 symbols per run, network permitting). Test `test-kronos`, `test-telegram`, then choose `scan` during market hours. `daily-learn` runs automatically via cron or can be triggered manually after close.
5. After the daily-learn workflow, download the **market-and-paper-learning** artifact to inspect `reports/adaptive_report.json`, `heatmap.html`, and `paper_trades.csv`. These reports **do not stream automatically** in a previously downloaded browser tab.
6. Verify GitHub Actions schedules actually ran; hosted runner scheduling can be delayed, network data can be missing, and 6-hour continuous jobs are not guaranteed to remain online. The workflow does not currently use an authoritative NSE holiday calendar.

### Local tests (Python 3.11 with internet access for dependencies)

```bash
python -m pip install -r requirements.txt
python -m unittest discover -v
python scanner.py --task report
```

When running locally, `scan` runs a repeated intraday loop; it is not a one-iteration dry run. Never supply real trading credentials: this package has **no broker integration**.

### Tested and not tested here

- Passed **19 offline tests**, covering the original 11 deterministic tests, new validation, holdout rejection, forecast scoring, missing-data safeguards, and repaired scan-cycle wiring.
- Python compilation passed, and a report was generated against the supplied 24-signal/zero-outcome database. All activation checks properly remained **false**.
- External `yfinance`/NSE downloads, GitHub Actions remote execution, actual Telegram messages, real Kronos model inference/download, and a multiweek real-time trading experiment were **not available to test in this environment**. `yfinance` was stubbed *only in the local test harness* (not in the shipped application), because the package was unavailable in the audit runtime. Users must install dependencies as above.

### What this version does not claim

No autonomous fine-tuning of neural Kronos weights; no exact identification of why a day's high/low occurred (it records observable price/volume/peer/news **clues**, not proven causality); no fully NSE-wide tick-based sector heatmap; no HFT order-book model; no direct merged integration with the other GitHub repositories; no broker auto-trading; no assured 10–14-day profits; and no guaranteed free GitHub Actions compute/API use.

All thresholds can be inspected in `adaptive_journal.py`. Raising or lowering minimum sample sizes may impact statistical validity. A 14-day positive result remains vulnerable to selection bias, correlated symbols, market-regime changes, slippage, survivorship bias, and changing data vendors.
