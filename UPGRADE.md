# Upgrade pack v4 (cumulative: includes v1, v2 and v3)

Copy these files over your repo. Keep your own `runtime/bis20.sqlite3` and secrets untouched.
If your live `scanner.py` is newer than the zip you sent, merge instead of overwriting.

**Changed:** `scanner.py`, `adaptive_journal.py`, `.github/workflows/bis20.yml`, `test_scout_research.py`
**New:** `backtest_history.py`, `test_backtest_history.py`, `features_pro.py`, `feature_labels.py`, `feature_learner.py`, `mirror_study.py`, `news_events.py`, `market_regime.py`,
`trading_calendar.py`, `data/nse_holidays.json`, `data/event_calendar.json`, `test_hardening.py`, `test_pro_features.py`, `test_smart_features.py`

## New in v4
**Meaningful-target floor** (`scanner.py`): the old level-calibration code could pick any target multiple from 1.0R to 3.0R, whichever
maximised simulated results, so nothing stopped it drifting toward small targets that flatter the win rate. Targets are now forced to be
at least `MIN_T1_R` (default 2.0R) and at least `MIN_TARGET_PCT` (default 1.5% of price) AFTER any learned level is applied, and the
calibrator no longer considers anything below the floor. The daily digest also prints the break-even win rate for the current target
(about 36% at 2.5R after costs) with a warning that 90%+ would mean tiny targets or a data error.

**Historical backtest** (`backtest_history.py`): replays the real `evaluate_symbol` over months of 5-minute data and grades with the real
`resolve_outcome` and cost model. No look-ahead (tested: future data cannot change past signals; daily metadata uses earlier sessions
only). Reports expectancy in R and % per trade net of costs, profit factor, drawdown, month-by-month stability, how often price really
travels 1.5/2/3/5% (before the stop), validated features and combinations (same train/holdout + multiple-testing control as live), the
opposite-trade control, and a stop/target grid that only contains meaningful targets (>= 1.5%). Win rate is shown for information only.

    python backtest_history.py --data ./history --out ./backtest_out --workers 4

Input: a folder with one CSV/Parquet per symbol (or one CSV with a `symbol` column); columns datetime/open/high/low/close/volume;
1-minute data is resampled to 5; naive times are treated as IST. A fast volume prefilter makes it about 19x quicker and is tested to give
identical trades (`--exact` turns it off). Expect roughly 30-90 minutes for ~200 symbols over 6 months on a few cores.

**What the backtest cannot reproduce:** Kronos, news, the sector/peer alert gate and the Nifty/VIX regime (no historical inputs), and the
live "tier-2 only when no tier-1 exists" rule is applied per symbol. It tests the raw signal engine plus the new chart features, not the
full alert gate. Beware survivorship (only stocks that exist today), unadjusted splits/bonuses (days with a >20% overnight gap are skipped),
and optimistic fills on small caps (default cost 20 bps round trip; raise `--cost-bps` to stress it).

## New in v3
**Chart structure** (all in `features_pro.py`, rule-based heuristics, logged with every signal):
support/resistance zones (clustered swing highs/lows with touch counts, "at support / at resistance" flags), swing-based patterns,
head and shoulders + inverse, fair value gaps (unfilled, retest flag), order blocks (first retest of an untouched block),
wick and body anatomy (wick ratios, rejection wicks, body vs average). Confluence is now out of 7 groups
(volume profile, candles, patterns, trend stack, flow, levels, FVG/order block).

**Combination learning** (`feature_learner.py`): besides single features it now tests PAIRS (e.g. "Candle: hammer && Level: at support").
A combination is credited only if it beats EACH of its own parts, with the same multiple-testing bar as singles.
Named combos are also logged directly (candle + level, wick rejection at level, FVG/OB retest + volume profile, ...).
Simulation: false promotion on noise ~5-7%; a real single +0.8R feature found ~93%; a real combo-only +1.2R effect ~98%.

**Opposite-trade test** (`mirror_study.py`): every logged signal also gets a paper trade in the opposite direction at the same entry,
same risk, same R target; graded by the same resolver and cost model. They live in their own table and never touch the real ledger,
learning, promotion or alerts. The daily report compares original vs opposite (edge, % where original is better, lower bound) and says in
words whether the setup has directional edge. Recent existing signals are backfilled automatically. Disable with `ENABLE_MIRROR_TRADES=false`.

**News**: adds Google News RSS (free) per candidate, de-duplicated, recency-weighted (<=6h counts double, >48h ignored), alongside the Yahoo
headlines. Disable with `ENABLE_GOOGLE_NEWS=false`.

## Carried over
v2: Kronos advisory, volume profile, candlesticks, indicators, news/event tags, learned score adjustment (+-3, capped +-6, only after 20 sessions).
v1: bootstrap promotion check, replay weights opt-in, holiday skip, crash/missed-run Telegram alerts, Nifty/VIX regime.

## Limits (read these)
- Pattern detectors are simple geometric rules; FVG and order blocks are heuristic versions of discretionary ideas. They are logged and
  judged by results, not assumed to work.
- Rare combos need lots of trades: a combo seen in ~5% of trades needs hundreds of trades before it can be evaluated. At ~5 trades per
  day expect months. A universe-wide pattern event study (all stocks, every bar) would get data much faster but is not built yet.
- Mirror trades share the same market day as their originals, so a pair-difference removes market direction but not everything.
- News is headline keywords. Google News RSS and Yahoo RSS can be sparse or rate-limited for small caps.
- Tests: 106 pass offline. Checked against a copy of your real DB (mirror payloads for all 83 signals, learner runs). Not tested here: live
  Yahoo/NSE/Google data, real Telegram, Kronos inference, GitHub Actions.
