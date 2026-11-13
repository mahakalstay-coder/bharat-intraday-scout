# Upgrade pack: hardening patch

Copy these files over your repo (keep your own `runtime/bis20.sqlite3` and secrets untouched):
`scanner.py`, `adaptive_journal.py`, `.github/workflows/bis20.yml`, `market_regime.py`,
`trading_calendar.py`, `data/nse_holidays.json`, `test_hardening.py`.
If your live repo's `scanner.py` is newer than the one you zipped, re-apply the diff instead of overwriting.

## What changed (and only this)
1. **Promotion gate**: new `holdout_confidence` check. A deterministic bootstrap lower bound (20th percentile,
   `ACTIVATION_LCB_PCT`) of holdout mean net R must be >= `ACTIVATION_MIN_LCB_R` (default 0.0). Stops a lucky 12-trade
   holdout from promoting the system. Set `ACTIVATION_MIN_LCB_R=-99` to disable.
2. **Replay weights are no longer live by default.** Single-day `replay_candidate_weights.json` is used only if
   `ALLOW_REPLAY_WEIGHTS=true`. Validated weights behave exactly as before.
3. **Holiday calendar**: `data/nse_holidays.json` is checked by the scanner and the workflow (scheduled scan/daily-learn
   skip cleanly). Seeded only with dates I could confirm (2026-11-24, 2026-12-25). Verify against NSE's circular and add the rest.
4. **Regime features**: every processed signal now stores Nifty trend + India VIX bucket. The journal's feature scorecard
   reports results by regime. Log-only, never a gate; any data failure becomes UNKNOWN.
5. **Telegram safety nets**: a crash sends a short alert (bot token redacted); the 15:35 job warns if no scan cycle ran that day.

## Deliberately unchanged
Entry logic, six formulas, Kronos usage, stop/target math (0.8% minimum stop already caps cost at ~0.25R),
shadow logging of rejected candidates, cost model, session holdout learner, outcome grading, state persistence.

Tests: 34 pass (24 original + 10 new). Not tested here: live Yahoo/NSE data, real Telegram delivery, Kronos inference, GitHub Actions.
