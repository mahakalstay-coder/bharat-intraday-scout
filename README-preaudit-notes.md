# Bharat Intraday Scout — audited paper-learning update

Read this file before replacing files in your GitHub repository. This package adds sampled sector/industry monitoring and a conservative formula adaptation layer. It does not establish profitable performance or implement neural self-training.

## What the supplied version actually contained

- 24 stored signals; no outcomes table in the supplied database.
- learned_filters.json: zero resolved results, warmup mode.
- Six formula weights derived from the November 9 replay, with only 2–11 triggers per formula. Its favorable-excursion score was not a stop-aware realized win rate.
- Functions for daily outcome resolution, retrospective threshold changes and formula scoring existed. Kronos was inference only, and an LLM prompt mentioning reinforcement learning did not train a model.
- No sector/industry metadata, sector heatmap or same-industry peer analysis.

## Added / changed

1. `market_learning.py` computes equal-weight sector breadth, advancing-stock percentage, change since the preceding snapshot and same-industry peers. Returns use the preceding session's close from downloaded bars, not the old universe cache. Only completed, current-session bars no older than 15 minutes are included. Unknown or old classifications stay Unknown.
2. A bounded Yahoo metadata refresh runs before scans. Defaults: 50 symbols per run, refresh at 30 days, classification excluded after 90 days. Coverage grows over several runs. A failure never guesses a classification.
3. Sector snapshots and a 30-day history live in SQLite. Signals retain their sector/industry/peer context at entry. Snapshot reports continue after the alert quota fills, until the existing entry window ends.
4. `reports/heatmap.html` is a self-contained offline report with stock/industry filtering, breadth tiles, snapshot time and coverage. `market_snapshot.json`, `paper_trades.csv`, and `performance.json` accompany it. It updates on scanner cycles, not in an already-downloaded browser page. This is sampled breadth, not an exchange index or tick feed.
5. The paper ledger separates target, stop, end-of-day, pending and no-data outcomes. It includes gross R and estimated net R. Default round-trip cost is 20 bps of entry notional; configure `PAPER_ROUND_TRIP_BPS` to your own estimated all-in cost. It is a model, not a verified broker charge. New signals freeze that assumption; old records are explicitly marked legacy estimated and excluded from new weight tuning.
6. Outcome grading requires all expected post-signal five-minute bars through 15:25. Incomplete sessions remain unresolved. Same-bar target+stop counts as stop, and adverse opening gaps may lose more than 1R. No-data is not counted as a loss.
7. The new weight learner uses at least 20 completed trading sessions, a chronological 70/30 session split, at least 30 training and 15 validation observations per formula, direction agreement and a conservative 3-standard-error validation screen. Stable weights change at most ±15% from baseline. Otherwise weights revert to baseline. Audit records are stored in SQLite and `data/validated_weights.json` supplies live formula weights.
8. Single-day replay writes `data/replay_candidate_weights.json`; it no longer overwrites live weights. Legacy threshold and target/stop tuning remain research diagnostics; live use is disabled by default. Do not enable `ENABLE_EXPERIMENTAL_LEGACY_FILTERS` without separate validation.
9. A required but unavailable Kronos model now blocks new signals instead of silently falling back. The minimum price is strictly above ₹60. The underlying NSE equity/liquidity universe remains broader than NIFTY constituents, although only the selected 200 stocks are scanned per run.
10. State persistence no longer uses destructive `git reset --hard`. Failed pushes retain local state in workflow artifacts. Quota reset no longer deletes the trade ledger.

## Install in the existing repository

1. Extract this ZIP. Copy the contents of `bharat-intraday-scout-2-main` to your repository root, including `.github/workflows/bis20.yml`, `scanner.py`, `market_learning.py`, and the test file.
2. IMPORTANT: retain your current live `runtime/bis20.sqlite3` if it contains newer signals than the supplied archive. The archive's database is the historical input with additive empty tables. Replacing a newer database would lose newer records. No automatic merging is attempted.
3. Keep existing Telegram secrets and optional TAURIC_API_KEY. No secrets are included or changed by this update.
4. In GitHub Actions, run `refresh-metadata` as needed until coverage is sufficient (50 per run by default). Then run `scan` during market hours. `daily-learn` runs after close; `report` exports saved state without fetching candles or sending Telegram.
5. Download the `market-and-paper-learning` artifact from the completed Actions run, extract it, and open `reports/heatmap.html`. For the CSV, open `reports/paper_trades.csv` in Excel. Reports from a running job are not a hosted live dashboard.
6. Run `python -m unittest -v test_market_learning.py` after installing requirements to verify the offline safeguards.

The workflow retains the original long-running GitHub Actions approach. GitHub scheduling and free data are not guaranteed real-time infrastructure. The 360-minute runner timeout may end a scan before the exchange close, so the separate 15:35 IST daily-learning job remains necessary. Persistent state requires repository write permission. Recovery artifacts last 30 days. No cloud deployment, repository push or Telegram message was performed while preparing this ZIP.

## What learning means here

The model adapts bounded indicator/formula weights from paper outcomes. It does not invent indicators, optimize every indicator period, retrain Kronos, or guarantee improved future trades. Validation is of conditional formula expectancy, not a full replay comparison of champion versus challenger portfolios. Correlated trades, repeated holdout inspection, selection bias from recording only alerted trades and changing market regimes remain limitations. A genuinely independent future paper evaluation is needed before relying on an adjustment. There is no shadow ledger of every rejected stock in this release.

The ledger assumes the signal's candle-close reference price; it does not know actual alert latency, fills, partial fills, slippage or your personal executions. Net R uses modeled costs. Full-bar coverage is conservative and may leave illiquid/missing-data sessions pending. No broker order placement exists.

## Repository audit — reviewed November 10, 2026

| Repository | Relevant mechanism | Status in this package |
|---|---|---|
| https://github.com/shiyu-coder/Kronos | Pretrained OHLCV forecasts; separate fine-tuning pipeline | Existing inference retained; neural fine-tuning not implemented |
| https://github.com/pradeepsiddappa/indian-trading-agent | Documents memory/reflection, paper outcomes, signal weight tuning and regime analysis | Outcome/weight-learning design reference; its LLM agents and web terminal are not integrated |
| https://github.com/sswapnil2/tradingview-mcp-india | MCP market-data, indicator and backtesting tools | Data/analysis design reference; no MCP client/server is integrated |
| https://github.com/pkjmesra/PKScreener | NSE scanners, sectoral indices, strategies and backtests | Scanner/sector design reference; its scanner catalog is not imported |
| https://github.com/bradleyboyuyang/ML-HFT | Supervised ML on level-II futures order-book data | Validation/data-requirements reference only; no order-book factors or HFT model are integrated |

No source code from these four additional projects was copied into the new module. They are not interchangeable plug-ins. Combining their code would require separate API adapters, dependency/license checks and integration tests. ML-HFT additionally needs a suitable tick/depth dataset; five-minute OHLCV cannot supply its order-book imbalance. All five repository pages were reviewed, but this is not an exhaustive source audit of every repository.

## Verification performed

11 deterministic tests passed: incomplete session, missing candle, same-bar stop-first, adverse stop gap, sell target, no-data exclusion, costs, fresh peers, insufficient learning data, holdout reversal rejection and stable promotion. Python compilation passed; the report command ran against the supplied 24-signal database. No live NSE/Yahoo feed, Telegram delivery, remote GitHub workflow, Kronos download/inference or profitability validation was executed here. Initial bundled reports truthfully contain pending trades and no market snapshot.
