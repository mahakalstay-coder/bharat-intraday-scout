# Bharat Intraday Scout 2.0 — GitHub Actions (no VPS)

This edition is a one-shot Python runner designed for a **private GitHub repository**. GitHub Actions starts a temporary Ubuntu runner on each scheduled run, performs one scan, sends Telegram alerts directly from Python, commits the small SQLite deduplication state, and shuts down. It does not run a continuously hosted API or require Apps Script.

## Important limits

- GitHub Actions scheduled workflows have a minimum five-minute interval, but scheduled starts can be delayed or dropped by GitHub. This is research alerting, not exchange-grade real-time execution.
- GitHub Free includes 2,000 Actions minutes/month for private repositories. Billing rounds each job's partial minutes up to a whole minute. A full 09:20–15:30 IST schedule is about 75 jobs per NSE trading day, or about 1,650 jobs in a 22-session month. For this to fit, the average scan job must generally finish within one billed minute, leaving about 350 minutes for setup/universe refresh/other runs. If average jobs take 2 billed minutes, the schedule will exceed the free allowance. Monitor Actions usage before relying on it.
- The initial source uses yfinance/Yahoo Finance, which is unofficial, can be delayed/throttled/incomplete, and is not a guaranteed real-time NSE feed.
- Kronos-mini is loaded on CPU for shortlisted technical candidates. It is a forecast, not a calibrated probability of profit.
- The scanner initially checks the top 100 stocks by 20-session average traded value from eligible NSE EQ shares (not only NIFTY 50/200). Price floor is ₹60 and average turnover floor is ₹10 crore.
- No broker orders are placed.

## Files

- `scanner.py`: one-shot market scanner, Kronos inference, Telegram alerts, SQLite dedupe state.
- `requirements.txt`: Python libraries (PyTorch CPU wheel is installed separately by the workflow).
- `.github/workflows/bis20.yml`: weekday scan schedule plus weekly universe refresh.
- `runtime/bis20.sqlite3`: created by the first run; tracks signal deduplication and daily count.
- `data/liquid_universe.json`: created by the initial universe refresh; keeps the selected liquid NSE EQ symbols.

## Setup — browser only; no local Python install

1. Create a **private** repository on GitHub called `bharat-intraday-scout-2`.
2. Extract the ZIP on your Windows computer. Open the extracted folder. You should see `scanner.py`, `requirements.txt`, `.gitignore`, `data`, `runtime`, and `.github`.
3. In the new GitHub repository choose **Add file → Upload files**. Drag the *contents* of the extracted folder into the upload area, including the `.github` folder. Make sure the final repository paths include `.github/workflows/bis20.yml`, not `.github/workflows/.github/workflows/bis20.yml`.
4. Commit the uploaded files to the `main` branch.
5. Open **Settings → Secrets and variables → Actions → New repository secret**.
   - Name: `TELEGRAM_BOT_TOKEN`; value: your existing Telegram bot token.
   - Name: `TELEGRAM_CHAT_ID`; value: your existing Telegram chat ID.
   Never put either value in `scanner.py`, README, workflow YAML or ordinary repository files.
6. Open **Settings → Actions → General**. Under **Workflow permissions**, select **Read and write permissions** and save. The workflow needs `contents: write` so it can preserve deduplication and the liquid-universe cache across temporary runners.
7. Open **Actions**. If GitHub asks to enable Actions for this repository, enable them.
8. Open **Actions → Bharat Intraday Scout 2.0 → Run workflow**.
9. Select `test-telegram`, then **Run workflow**. Confirm you receive a Telegram test message.
10. Run the workflow again, this time select `test-kronos`. Wait for a successful run. This downloads and caches the Kronos-mini and tokenizer weights. The first run can take longer.
11. Run it again with `refresh-universe`. It downloads the NSE EQ list and builds the liquid universe. This can take several minutes; do not run the market schedule until this succeeds.
12. Check that `data/liquid_universe.json` and `runtime/bis20.sqlite3` appear in the repository after a successful run. These are state files and must remain committed to preserve deduplication across temporary runners.
13. Open **Settings → Secrets and variables → Actions → Variables → New repository variable**. Create `BIS_SCANNER_ENABLED` with value `true`. Until this variable exists, timed schedules are skipped safely; manual setup/test runs still work.
14. Automatic scans now run on weekdays at 09:20, 09:25, every five minutes from 09:30 to 15:25, and 15:30 India time. Weekly liquidity refresh is scheduled Sundays at 08:30 India time. Schedule times are converted from UTC in the workflow file.

## Monitor and control

- Go to **Actions → Bharat Intraday Scout 2.0** to inspect each scan's logs and duration.
- To pause automatic timed runs, change the repository variable `BIS_SCANNER_ENABLED` to `false`. Manual workflow runs remain possible.
- To stop all scanning immediately, use **Actions → Bharat Intraday Scout 2.0 → ... → Disable workflow**.
- To change the scanned universe size, edit `INTRADAY_UNIVERSE_SIZE` in `.github/workflows/bis20.yml`. Start at 100. Increase only after checking execution time and Actions usage.
- The workflow requires `contents: write` so it can commit the SQLite deduplication state and the universe cache. Do not add untrusted third-party workflows to this repository.

## First-run validation

Do not trade from alerts immediately. First run at least several sessions in paper mode and check:

- all symbols have fresh five-minute candles;
- the previous five same-time volume samples exist;
- Kronos is loaded and forecast direction is logged;
- Telegram alerts are not duplicated after consecutive workflows;
- Actions jobs finish fast enough to stay within the private-repository free allowance;
- performance after brokerage, STT, exchange charges, GST, stamp duty, spread and slippage.

## Technical configuration

Defaults in `scanner.py`:

- Minimum price ₹60.
- Minimum 20-session average traded value ₹10 crore.
- Top 100 qualifying main-board EQ symbols by average traded value.
- Current 5-minute volume compared with same clock-time candles from five previous sessions; all five are required.
- Cumulative session-volume comparison against the previous five sessions.
- Trend, VWAP, breakout, candle body/close-position and RSI confirmation.
- Anti-chasing limit 0.5 ATR beyond breakout/breakdown.
- Initial stop reference 1.2 ATR and target reference 2R.
- Maximum three alerts per day.
- Kronos direction confirmation is required by default; failure to load or predict suppresses alerts.

These are initial research parameters, not proven optimal values. The 100-stock shortlist is for Actions-runtime control; it is still selected from the broader NSE EQ list, not an index membership list.

## v3 self-learning (added)

- Every alert is graded after the session by walking the real 5-minute bars: did **stop** or **T1** hit first (stop assumed first if both in one bar), plus how far price went for/against you (MFE/MAE). Stored in `outcomes` table.
- `daily-learn` (auto, 15:35 IST) also studies the day's **top gainers and top losers**, replays the scanner on them, records caught/missed and why, and the volume at ignition.
- `data/learned_filters.json` is rebuilt from all graded history (last 40 days): blocks buckets (side / tier / time window / formula) with enough samples and negative expectancy, raises the score bar when recent results are negative, learns a minimum RVOL, and calibrates target (T1 R-multiple) and stop width per BUY/SELL.
- **Warm-up**: until 20 alerts are graded, alerts are still sent but tagged 🧪 LEARNING. After that only alerts passing the learned gate are sent (🎯 PROVEN-ONLY).
- Telegram: 09:20 morning brief (rules live today) + 15:35 daily learning report (results, movers study, what changed).
