"""Offline T1-before-stop study. Inputs are frozen when signals are recorded."""
import math


def features(p):
    stock = p.get('market_context') or {}
    risk = abs(p['price'] - p['stop'])
    return [float(p.get(k) if p.get(k) is not None else float('nan'))
            for k in ('score','relative_volume','cumulative_relative_volume','adx','bb_bandwidth_pct')
            ] + [abs(p['target'] - p['price']) / risk if risk else float('nan'),
                 float(stock.get('peer_relative_pct') or 0) if stock.get('peer_relative_pct') is not None else float('nan'),
                 p['side'], str(p.get('tier') or 'Unknown'), str(stock.get('sector') or 'Unknown')]


def evaluate(ledger, payloads):
    rows = [r for r in ledger if r['result'] in ('T1','STOP','EOD_WIN','EOD_LOSS','EOD_FLAT')
            and r['cost_source']=='entry snapshot' and r['net_r'] is not None and math.isfinite(r['net_r'])]
    days = sorted({r['day'] for r in rows})
    report = dict(status='collecting',paper_only=True, samples=len(rows), sessions=len(days),
                  hits=sum(r['result']=='T1' for r in rows))
    if len(days)<12:
        return report
    first_test = days[int(len(days)*0.7)]
    train = [r for r in rows if r['day']<first_test]
    test = [r for r in rows if r['day']>=first_test]
    labels = [int(r['result']=='T1') for r in train]
    if min(sum(labels),len(labels)-sum(labels))<10 or len(test)<12:
        return report
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(iterations=60, depth=3, learning_rate=.05, verbose=False,
                               allow_writing_files=False, thread_count=2, random_seed=42)
    model.fit([features(payloads[r['id']]) for r in train], labels, cat_features=[7,8,9])
    probs = model.predict_proba([features(payloads[r['id']]) for r in test])[:,1]
    base = sum(labels)/len(labels)
    actual = [int(r['result']=='T1') for r in test]
    subset = sorted(range(len(test)), key=lambda i:-probs[i])[:math.ceil(len(test)/4)]
    report.update(status='evaluated',train=len(train),holdout=len(test),holdout_from=first_test,
                  brier=round(sum(float(p-y)**2 for p,y in zip(probs,actual))/len(test),4),
                  baseline_brier=round(sum((base-y)**2 for y in actual)/len(test),4),
                  top_quarter_mean_net_r=round(sum(test[i]['net_r'] for i in subset)/len(subset),3),
                  all_holdout_mean_net_r=round(sum(r['net_r'] for r in test)/len(test),3))
    return report
