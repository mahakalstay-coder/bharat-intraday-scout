# Validation — 17 November 2026

Source baseline: a5603442f61a00a51ed7a11fd5c2f4642a11add8.

- Full offline regression: **144 tests passed**. After the last continuous-learning/weight-rounding changes, all **22 four-header tests passed again**.
- All modified/new Python modules compile. Existing `scanner.py --help` retains the workflow task interface and `--symbol` argument.
- `git diff --check` passes.
- Frozen original four-strategy module matches uppercase `Scanner.py` byte-for-byte: SHA-256 3158467038110023cd2d1ea912b6374d56c14d8f3d7ce8f2488fe578975b52f5.
- Seven existing core function ASTs are unchanged: evaluate_symbol, build_signal_dict, learned_gate, resolve_outcome, load_learned_weights, run_backtest_and_learn, recompute_filters.
- Existing SQLite database bytes match the baseline. The install ZIP excludes the database, saved classifications/weights, reports, secrets and workflow.
- No live Telegram message, broker order or GitHub push was performed. Official CSV/quote availability and real Telegram delivery remain deployment checks.

The test suite covers legacy behavior, all four header routing, quotas/dedup, sender failure, stale-message expiry, complete-bars-only simulation, stop/target ties, adverse gaps, conservative trail accounting, missing data, exactly-once LEARNING_SETUP updates, unchanged fixed arrays, separate intraday recovery labels, regime persistence, peer self-exclusion, source refresh retention, forward-only promotion and later withdrawal.

These checks validate code behavior, not a profitable trading edge. Background sampling is bounded; reference prices are not broker fills. Read FOUR_HEADERS_INSTALL.md before upload.
