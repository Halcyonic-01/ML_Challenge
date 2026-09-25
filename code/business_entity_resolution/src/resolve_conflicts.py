"""Mutual-exclusivity resolution for Source 1 -> Source 2/3 links.

Training ground truth assigns every Source 2/3 record to at most one Source 1
entity (7,638,365 links over 7,638,365 distinct targets, max 1 claim each), so
any target claimed by two Source 1 entities carries at least one false positive.
The pair model scores candidates independently and cannot see those collisions.

This stage rescores only the contested pairs, keeps the highest-scoring claimant
per target, and drops the rest. Scoring needs no FTS index: every field comes
from the test source TSVs.

The two retrieval-channel features are not recoverable after the fact, so both
are set to 1 for every contested pair. That is uniform within each contested
group, which is what the arg-max comparison depends on.
"""
import argparse, collections, csv, json, sys
from pathlib import Path

import joblib
import numpy as np

from features import pair_features

RETRIEVAL_FLAGS = (1, 1)


def read_links(path):
    with open(path, encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            sid, _, rest = line.rstrip('\n').partition('\t')
            yield sid, [x for x in rest.split(',') if x]


def find_contested(matching_path):
    claims = collections.Counter()
    rows = links = 0
    for _, ids in read_links(matching_path):
        rows += 1
        links += len(ids)
        claims.update(ids)
    contested = {t for t, c in claims.items() if c > 1}
    return contested, rows, links, len(claims)


def collect_claimants(matching_path, contested):
    by_target = collections.defaultdict(list)
    for sid, ids in read_links(matching_path):
        for t in ids:
            if t in contested:
                by_target[t].append(sid)
    return by_target


def load_records(path, wanted):
    out = {}
    with open(path, encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        col = {name: i for i, name in enumerate(header)}
        eid, bn, ba, cy = col['entity_id'], col['business_name'], col['business_address'], col['country']
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if parts[eid] in wanted:
                out[parts[eid]] = (parts[bn], parts[ba], parts[cy])
    return out


def resolve(matching_path, data_root, model_dir, out_path, report_path=None):
    data_root = Path(data_root)
    model = joblib.load(Path(model_dir) / 'best_model.joblib')

    contested, rows, links, distinct = find_contested(matching_path)
    excess = links - distinct
    print(f'rows {rows} links {links} distinct targets {distinct} '
          f'contested targets {len(contested)} excess links {excess}', flush=True)

    by_target = collect_claimants(matching_path, contested)
    needed_s1 = {s for v in by_target.values() for s in v}
    print(f'loading {len(contested)} target records and {len(needed_s1)} source 1 records', flush=True)

    s1 = load_records(data_root / 'test' / 'test_source1.tsv', needed_s1)
    targets = {}
    for name in ('test_source2.tsv', 'test_source3.tsv'):
        targets.update(load_records(data_root / 'test' / name, contested))
    missing_s1 = len(needed_s1) - len(s1)
    missing_t = len(contested) - len(targets)
    print(f'loaded source1 {len(s1)} (missing {missing_s1}) targets {len(targets)} (missing {missing_t})', flush=True)

    drop = collections.defaultdict(set)
    group_sizes = collections.Counter()
    top_scores = []
    margins = []
    unresolved = 0

    for tid, claimants in by_target.items():
        group_sizes[len(claimants)] += 1
        trec = targets.get(tid)
        if trec is None:
            unresolved += 1
            continue
        tn, ta, tc = trec
        rows_x = []
        usable = []
        for sid in claimants:
            srec = s1.get(sid)
            if srec is None:
                continue
            sn, sa, sc = srec
            rows_x.append(pair_features(sn, sa, sc, tn, ta, tc, *RETRIEVAL_FLAGS))
            usable.append(sid)
        if len(usable) < 2:
            unresolved += 1
            continue
        scores = model.predict_proba(np.vstack(rows_x))[:, 1]
        order = np.argsort(-scores)
        keep = usable[order[0]]
        top_scores.append(float(scores[order[0]]))
        margins.append(float(scores[order[0]] - scores[order[1]]))
        for sid in usable:
            if sid != keep:
                drop[sid].add(tid)

    dropped = sum(len(v) for v in drop.values())
    print(f'resolved groups {len(top_scores)} unresolved {unresolved} '
          f'links dropped {dropped} entities affected {len(drop)}', flush=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    in_place = out_path.exists() and Path(matching_path).samefile(out_path)
    write_to = out_path.with_suffix(out_path.suffix + '.tmp') if in_place else out_path
    kept = 0
    with write_to.open('w', encoding='utf-8', newline='') as out:
        w = csv.writer(out, delimiter='\t', lineterminator='\n')
        w.writerow(['source1_entity_id', 'matched_entity_ids'])
        for sid, ids in read_links(matching_path):
            if sid in drop:
                bad = drop[sid]
                ids = [x for x in ids if x not in bad]
            kept += len(ids)
            w.writerow([sid, ','.join(ids)])
    if in_place:
        write_to.replace(out_path)

    stats = dict(
        rows=rows, links_before=links, links_after=kept, links_dropped=links - kept,
        distinct_targets_before=distinct, contested_targets=len(contested),
        excess_links=excess, groups_resolved=len(top_scores), groups_unresolved=unresolved,
        entities_affected=len(drop),
        group_size_distribution={str(k): v for k, v in sorted(group_sizes.items())},
        kept_score_mean=float(np.mean(top_scores)) if top_scores else None,
        kept_score_median=float(np.median(top_scores)) if top_scores else None,
        kept_score_p10=float(np.percentile(top_scores, 10)) if top_scores else None,
        margin_mean=float(np.mean(margins)) if margins else None,
        margin_median=float(np.median(margins)) if margins else None,
        retrieval_flags_assumed=list(RETRIEVAL_FLAGS),
    )
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in stats.items() if k != 'group_size_distribution'}, indent=2), flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matching', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--model-dir', default='models')
    ap.add_argument('--out', required=True)
    ap.add_argument('--report')
    a = ap.parse_args()
    resolve(a.matching, a.data_root, a.model_dir, a.out, a.report)


if __name__ == '__main__':
    sys.exit(main())
