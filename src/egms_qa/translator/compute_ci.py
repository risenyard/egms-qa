"""Bootstrap 95% CIs per task from an eval generation dump (gen_*.jsonl with
{task, av, ok}). Publication rigor: thin-class tasks (e.g. S2 rare, n=6) get
honestly wide CIs instead of a single point estimate.

Usage: python compute_ci.py <gen_dump.jsonl> [n_boot=2000]
"""
import json, sys, collections, random


def balanced_acc(samples):
    by_cls = collections.defaultdict(lambda: [0, 0])
    for av, ok in samples:
        by_cls[av][1] += 1; by_cls[av][0] += int(ok)
    recs = [c / n for c, n in by_cls.values() if n]
    return sum(recs) / len(recs) if recs else 0.0


def plain_acc(samples):
    return sum(ok for _, ok in samples) / len(samples) if samples else 0.0


def majority(samples):
    # accuracy of always predicting the most-frequent CLASS (natural-dist baseline);
    # here approximated as the largest class share among the sampled labels
    c = collections.Counter(av for av, _ in samples)
    return max(c.values()) / len(samples) if samples else 0.0


NUMERIC = {"A1", "A2", "A3", "A4", "A5"}


def main(dump: str, n_boot: int = 2000) -> None:
    rng = random.Random(0)
    rows = [json.loads(line) for line in open(dump)]
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r["task"]].append((str(r["av"]), bool(r["ok"])))

    print(f"{'task':6} {'bal_acc':>7} {'95% CI':>15} {'plain':>6} {'major':>6} {'n':>5} note")
    print("-" * 62)
    for t in sorted(by_task):
        s = by_task[t]
        metric = plain_acc if t in NUMERIC else balanced_acc
        point = metric(s)
        boots = []
        for _ in range(n_boot):
            bs = [s[rng.randrange(len(s))] for _ in range(len(s))]
            boots.append(metric(bs))
        boots.sort()
        lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
        pa = plain_acc(s); maj = majority(s)
        note = ""
        if hi - lo > 0.20: note += "WIDE-CI "
        if t not in NUMERIC and pa < maj - 0.02: note += "PLAIN<MAJORITY"   # balancing cost
        print(f"{t:6} {point:>7.3f} [{lo:.3f},{hi:.3f}] {pa:>6.3f} {maj:>6.3f} {len(s):>5} {note}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
