# Bharat Scout: four headers and forward paper learning

Prepared against main commit a5603442f61a00a51ed7a11fd5c2f4642a11add8 on 17 November 2026.
This package is source code for installation. It has not been deployed to GitHub or verified against your Telegram account.

## Install once

1. Let the current market-day scanner finish before uploading. An already-running workflow keeps its old Python process; new files take effect in the next run.
2. Extract the ZIP. Upload its files into the ROOT of your existing repository, replacing lowercase `scanner.py`, `feature_labels.py`, and `requirements.txt`; add the four new modules, test file, and this guide. Do not upload the ZIP itself as the scanner.
3. Keep ALL other repository files. In particular, retain `runtime/bis20.sqlite3`, `data/`, `.github/workflows/bis20.yml`, your secrets, and uppercase `Scanner.py`. This ZIP deliberately contains no database, old reports, classifications, weights, or replacement workflow.
4. Commit the uploads together. In GitHub Actions choose the existing workflow, Run workflow, task `test-telegram` once. The test should list all four headers. It verifies delivery only, not trade performance.
5. Normal weekday scanning and daily learning stay automatic under your existing schedule. You may use `report` for a local/logged report; a manual market-hours `scan` is optional. No repeated manual learning run is required.
6. The next market-session morning message lists all four headers. At close, a separate FOUR-HEADER PAPER LEARNING digest shows graded sent calls and learning state. A header can correctly have zero calls when its original conditions do not match.

The existing workflow's requirements-file hash rebuilds its dependency environment. The new requirements add LightGBM and scikit-learn. All other original requirements remain.

## What is preserved

- The four original candidate/evaluator formulas, feature transforms, embedded model arrays, conviction thresholds, target/stop percentages and trailing configuration are retained byte-for-byte in `strategy_rules_original.py` (copied from uppercase `Scanner.py`). Its SHA-256 is `3158467038110023cd2d1ea912b6374d56c14d8f3d7ce8f2488fe578975b52f5`.
- `LEARNING_SETUP` retains its original online logistic weight update, with an exactly-once audit per closed paper alert. The other three strategies' model weights stay fixed.
- The original lowercase scanner's signal evaluation, legacy learned gate, target builder, outcome grader, replay logic, validated-paper gates and replay-weight setting are unchanged. Its old alerts and ledger continue separately.
- The new four-header engine runs alongside that original scanner BEFORE its legacy quota early return. Legacy quota exhaustion does not disable these four engines.
- Original per-header default is at most two paper calls per day, at most two for the same symbol with 30 minutes spacing. Thus up to eight new four-header calls can occur, in addition to existing legacy previews; zero is also possible. No forced calls to fill a quota.

## New headers

| Header | Target | Stop | Trail |
|---|---:|---:|---|
| EARLIER_SETUP | 2.00% | 1.00% | None |
| LEARNING_SETUP | 1.50% | 1.00% | None |
| 90PCT_SETUP_V9 | 2.00% | 1.50% | Original 0.30% lock trigger |
| BEST_SETTING_CALL | 1.50% | 1.00% | None |

Names containing 90PCT or BEST are historical names, not evidence of 90% accuracy. Conviction is a model score, not an established probability. These are PAPER RESEARCH messages; there is no broker execution.

## Learning and market adaptation

Every new record freezes its candidate, rule identity, cost assumption, market regime, available sector/peer evidence, and the companion policy available at that moment.

- Nifty's intraday direction is measured from completed 5-minute bars using VWAP and recent slope/momentum. It is separate from the prior-session background and daily percentage return, so a gap-down recovery is not blindly treated as ongoing selling pressure.
- A regime change needs two distinct completed snapshots; repeated polling of the same candle cannot confirm a change. Missing/stale evidence is UNKNOWN. These new labels never alter original entry gates.
- India VIX, sampled market breadth, sector strength and industry-peer relative strength are captured when available. BUY and SELL results are grouped separately by intraday regime; recent SELL success is not promoted into a universal SELL preference.
- Winners are graded GREAT or GOOD using net reward, target completion and adverse excursion; negative net outcomes are BAD. Rejected setups get separate hypothetical outcome studies. Neither rejected candidates nor background candidates are presented as sent alerts.
- Additional Renko/SMC/cycle/Gann observations and heatmap tiles already present in the repository are now connected as observation-only research. Their optional failures do not stop original scans. Their labels are opt-in and do not enter the legacy feature-score adjustment by default.

### Controlled setting proposals

The fixed baseline strategies never receive a new filter or changed target/stop from this addition. The companion learner tests conditional selection rules (RVOL floors, peer agreement and intraday agreement) separately for each header, side and regime.

1. Collect at least 20 historical sessions and 60 eligible outcomes in that category.
2. Evaluate four predeclared candidate filters, accounting for the opportunity cost of skipped winners. Require at least 30 retained examples and a conservative daily improvement margin.
3. Freeze the selected proposal and record its accept/skip decision prospectively at each NEW entry. No same-day or historical backfilled decision counts as forward evidence.
4. Only after at least 15 completely new sessions, 45 opportunities and 20 retained outcomes can it become VALIDATED_PAPER. Require positive retained net expectancy and positive daily improvement after a 3-standard-error margin.
5. After a rejection/withdrawal, require a fresh 20-session training window before proposing a successor; do not repeatedly search the same failed validation history. Track later performance. After at least five new post-validation sessions and ten retained outcomes, withdraw a companion that is losing and underperforming its baseline. Store the decision in the audit table.

Promotion changes the paper companion's status, not the original strategy. It does not automatically alter original stops/targets, switch on live trading, or create guaranteed winning calls. Conservative sample requirements can take substantially longer than 10–14 days. Sparse categories remain collecting.

## Sector and peer data

The updater starts with 12 common Nifty sector benchmark identifiers and discovers additional official sector pages from the NSE Indices directory. It fetches official constituent CSVs in bounded batches, preserves the last successful source/version, and refreshes weekly. First-time coverage fills incrementally, typically three sectors per metadata refresh.

- Source: https://www.niftyindices.com/indices/equity/sectoral-indices
- A new sector can have constituent mapping before the free quote provider supports its index. Missing/failed benchmark quotes stay UNKNOWN, never replaced with a fabricated official index return.
- Official index performance and equal-weight sampled breadth are labelled separately.
- True peers are matched on available industry classification, exclude the stock itself and require synchronized timestamps. Compare against the median peer return, not simply the biggest winner.
- Eligible non-index midcap/smallcap stocks remain in the existing scan universe; industry mappings can associate them with relevant benchmark context. Unsupported classifications remain unknown.
- No historical trade is assigned today's constituents or market regime retrospectively. Vendor industry labels may not match official classification, reducing coverage; no guessed mapping is used.

## Reports and state

New SQLite tables live INSIDE the existing `runtime/bis20.sqlite3`; no table is dropped and no legacy signal/outcome is rewritten by the new four-header ledger:

- `four_header_calls`: separate paper alerts, approved background studies, rejected studies, delivery and outcome records.
- `four_header_cycles`: bounded recent per-header matching/rejection/error counts.
- `four_weight_audit`: original LEARNING_SETUP weight changes with before/after values.
- `four_learning_policies`, `four_policy_audit`: frozen forward proposals and promotion/rejection/rollback decisions.

Generated outputs:

- `reports/four_header_performance.html`: readable scoreboard.
- `reports/four_header_performance.json`: sent-vs-background statistics, regimes, associations, latest calls and policy evidence.
- `reports/four_header_ledger.csv`: individual records.

The persistence list now includes reports, sector mappings and ML files as well as the existing database, fixing stale repository summaries. Existing workflow artifacts already retain reports and the database. It does not delete your learning history.

## Honest grading and delivery limits

The four-header grader fixes an optimistic accounting problem in the unused old tracker, while preserving configured strategy percentages. It considers complete candles starting after the record timestamp, assumes stop first when target and stop are touched in the same candle, accounts for adverse stop gaps, and never uses a newly armed trail to invent a same-candle profit. Ambiguities are flagged and excluded from companion promotion. This can produce lower win rates than the old optimistic tracker.

Outcome calculations remain reference-price paper simulations, not proven executable fills. Full spread, market impact, actual broker charges and latency are not independently verified. The default cost is the existing 20 basis points round trip, frozen per record. Missing intermediate bars remain pending and eventually NO_DATA; they are not silently counted as losses or wins. Pending and missing-data coverage always accompany win rates.

Delivery is marked SENT only after the sender confirms success. Failed messages remain pending for a fresh retry; stale messages expire after 15 minutes. A remote Telegram acceptance followed by a connection failure can still create an ambiguous delivery/retry; no Telegram exactly-once guarantee is claimed. The daily report is the comparison source of truth, not a header name.

The immutable original four-header candidate engine contains fixed ADX/MFI inputs in parts of its feature construction. They are retained to honor unchanged strategy logic. Added evidence does not turn those placeholders or model scores into validated indicators/probabilities. The capped background/rejection sample is not an exhaustive backtest of the entire market.

## Validation

- Existing repository suite and new tests run offline with fixture data; no Telegram message, broker order, GitHub push or live scan is performed by validation.
- Full regression suite: 144 tests passed; the new engine's 22 tests passed again after the final learning lifecycle/rounding changes. See VALIDATION.md for final checks.
- Original strategy module is byte-for-byte identical to uppercase `Scanner.py`; seven legacy core functions match their original ASTs; original SQLite bytes remain unchanged in the prepared checkout.
- Tests cover independent four-header quotas, delivery failure/retry, no future-candle leakage, missing bars, same-bar stop/target ambiguity, conservative trails, adverse gaps, no-data exclusion, exactly-once online learning, regime persistence, self-excluded peers, forward-only promotion and subsequent rollback.

Improvement is a hypothesis that must survive new paper evidence; this package does not promise future profitability.
