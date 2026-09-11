#!/usr/bin/env python3
"""Validate existing PP knapsack exports and regenerate aggregate paper assets.

This imports measurements; it does not rerun the matching algorithm or validate
funding identities. No address/hash pairs are copied to the output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
import statistics
import tempfile
import zipfile

PAPER = Path(__file__).resolve().parents[1]
PREFIXES = {1: "knapsack_1_withdraw", 2: "knapsack_2_withdraws", 3: "knapsack_3_withdraws"}
COUNTS = {1: "matches_count", 2: "pairs_count", 3: "triplets_count"}
SUFFIXES = {1: "matches_per_deposit", 2: "pairs_per_deposit", 3: "triplets_per_deposit"}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def index_rows(rows, label):
    require(bool(rows), f"{label}: empty measurement population")
    require(all(r.get("dep_hash") for r in rows), f"{label}: missing deposit IDs")
    result = {r["dep_hash"]: r for r in rows}
    require(len(result) == len(rows), f"{label}: duplicate deposit IDs")
    return result


def summarize(rows, k):
    key = COUNTS[k]
    for r in rows:
        require(int(r[key]) >= 0, f"{k}-sum: negative count")
        require(r["truncated"] in ("0", "1"), f"{k}-sum: invalid truncation flag")
        require(not int(r["truncated"]) or int(r[key]) > 0,
                f"{k}-sum: truncated zero-count search")
    n = len(rows)
    require(n > 0, f"{k}-sum: empty measurement population")
    singleton = sum(int(r[key]) == 1 and r["truncated"] == "0" for r in rows)
    zero = sum(int(r[key]) == 0 for r in rows)
    capped = sum(int(r["truncated"]) for r in rows)
    return dict(k=k, deposits=n, matched=n-zero, matched_pct=100*(n-zero)/n,
                no_match=zero, singleton=singleton, truncated=capped,
                multiple_complete=n-zero-singleton-capped,
                truncated_pct=100*capped/n,
                recorded_combinations=sum(int(r[key]) for r in rows))


def check_details(stream, rows, k):
    """Reconcile counts, distinct combination IDs, and linked withdrawal counts."""
    indexed = index_rows(rows, f"{k}-sum counters")
    counts = Counter()
    linked = defaultdict(set)
    seen = set()
    max_residual = Decimal(0)
    for r in csv.DictReader(stream):
        dep = r["dep_hash"]
        require(dep in indexed, f"{k}-sum: detail has unknown deposit ID")
        hashes = tuple(sorted(r[f"w{i}_hash"] for i in range(1, k+1)))
        require(len(set(hashes)) == k, f"{k}-sum: repeated withdrawal within combination")
        item = (dep, hashes)
        require(item not in seen, f"{k}-sum: duplicate combination")
        seen.add(item)
        require(Decimal(r["dep_amount"]) == Decimal(indexed[dep]["dep_amount"]),
                f"{k}-sum: detail/counter deposit amount differs")
        amounts = [Decimal(r[f"w{i}_amount"]) for i in range(1, k+1)]
        amount = Decimal(r["dep_amount"])
        require(amount.is_finite() and all(a.is_finite() for a in amounts),
                f"{k}-sum: non-finite amount")
        max_residual = max(max_residual, abs(sum(amounts)-amount))
        counts[dep] += 1
        linked[dep].update(hashes)
    require(all(counts[h] == int(r[COUNTS[k]]) for h, r in indexed.items()),
            f"{k}-sum: detail/counter combination counts differ")
    if k > 1:
        require(all(len(linked[h]) == int(r["linked_withdraws_count"])
                    for h, r in indexed.items()),
                f"{k}-sum: distinct withdrawal counts differ")
    return dict(detail_rows=sum(counts.values()), counts_reconciled=True,
                maximum_exported_amount_residual_eth=str(max_residual))


def check_scores(rows, scores, k):
    """Reject whole score snapshot if populations or shared fields differ."""
    a = index_rows(rows, "counters")
    b = index_rows(scores, "scores")
    require(a.keys() == b.keys(), "score and counter populations differ")
    shared = ["dep_amount", "dep_time", "truncated"]
    shared += [COUNTS[k]] if k < 3 else []
    shared += ["linked_withdraws_count"] if k > 1 else []
    values = []
    for h, r in a.items():
        s = b[h]
        require(all(r[field] == s[field] for field in shared),
                "score and counter fields differ")
        denominator = int(s["naive_anonymity_set"])
        require(denominator >= 0, "negative score denominator")
        if r["truncated"] == "1":
            require(not s["s_score"].strip(), "truncated score must be missing")
            continue
        numerator = int(r["matches_count"] if k == 1 else r["linked_withdraws_count"])
        require(numerator <= denominator, "score numerator exceeds denominator")
        expected = numerator / denominator if denominator else 0.0
        actual = float(s["s_score"])
        require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
                "score does not equal numerator / denominator")
        if int(r[COUNTS[k]]) > 0:
            values.append(actual)
    return values


def write_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_outputs(out, summaries, score_values, report):
    write_csv(out / "knapsack_summary.csv", summaries)
    score_rows = [dict(k=k, score=value, cdf=(i+1)/len(values))
                  for k, values in score_values.items()
                  for i, value in enumerate(sorted(values))]
    write_csv(out / "knapsack_score_ecdf.csv", score_rows, ["k", "score", "cdf"])
    lines = [r"% Generated by scripts/refresh_pp_measurements.py; do not edit.",
             r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Model & Deposits & Any match & One combination & Capped\\", r"\midrule"]
    for s in summaries:
        lines.append(f"$1\\to {s['k']}$ & {s['deposits']:,} & "
                     f"{s['matched']:,} ({s['matched_pct']:.1f}\\%) & "
                     f"{s['singleton']:,} & {s['truncated']:,}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "knapsack_table.tex").write_text("\n".join(lines)+"\n")
    notes = []
    if report["cohorts"]["two_equals_one_without_singletons"] and report["cohorts"]["two_equals_three"]:
        notes.append("The two- and three-withdrawal exports use the same deposit cohort, "
                     "obtained by excluding the one-withdrawal singleton cases from the first cohort.")
    else:
        notes.append("The exports do not form a common sequential cohort; their marginal rates "
                     "must not be interpreted as a paired comparison or added together.")
    notes.append("Capped searches have incomplete combination counts. A single recorded combination "
                 "is counted as unique only for non-capped searches. These are exported model outcomes, "
                 "not verified funding links.")
    (out / "knapsack_cohort_note.tex").write_text("\n".join(notes)+"\n")
    # Matplotlib is needed only for the figure. CSV/LaTeX calculations use stdlib.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.3), layout="constrained")
    bottom = [0.0]*len(summaries)
    for field, label, color in [
        ("no_match", "No recorded match", "#b9bdc5"),
        ("singleton", "One combination (not capped)", "#0072b2"),
        ("multiple_complete", "Multiple combinations (not capped)", "#56b4e9"),
        ("truncated", "Capped search", "#e69f00")]:
        heights = [100*s[field]/s["deposits"] for s in summaries]
        ax.bar(range(len(summaries)), heights, bottom=bottom, label=label, color=color, width=.55)
        bottom = [a+b for a,b in zip(bottom, heights)]
    ax.set(xticks=range(len(summaries)),
           xticklabels=[f"1 → {s['k']}\nN = {s['deposits']:,}" for s in summaries],
           ylim=(0,100), ylabel="Share of exported deposit cohort (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(.5,1.29), ncol=2, frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "knapsack_outcomes.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / "knapsack_outcomes.png", dpi=170)
    plt.close(fig)
    (out / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")


def refresh(source, output, strict=False):
    report = {"schema_version": 1, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "inputs": {}, "warnings": [], "details": {}, "scores": {}}
    def track(path):
        report["inputs"][str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    root = source / "results/data/Heuristics/Knapsack"
    populations, summaries, score_values = {}, [], {}
    for k, prefix in PREFIXES.items():
        folder = root / f"{k}-sum"
        rows = read_csv(track(folder / f"{prefix}_{SUFFIXES[k]}.csv"))
        index_rows(rows, f"{k}-sum counters")
        populations[k] = rows
        summaries.append(summarize(rows, k))
        detail = folder / f"{prefix}_matches.csv"
        if detail.exists():
            with track(detail).open(encoding="utf-8-sig", newline="") as stream:
                report["details"][k] = check_details(stream, rows, k)
        else:
            with zipfile.ZipFile(track(detail.with_suffix(".zip"))) as archive:
                members = [n for n in archive.namelist() if Path(n).name == detail.name]
                require(len(members) == 1, f"{k}-sum: expected exactly one detail CSV in ZIP")
                with archive.open(members[0]) as raw:
                    report["details"][k] = check_details(io.TextIOWrapper(raw, encoding="utf-8-sig"), rows, k)
        score_path = folder / f"{prefix}_s_score.csv"
        if not score_path.exists():
            report["warnings"].append(f"{k}-sum score omitted: missing CSV")
            report["scores"][k] = {"accepted": False, "reason": "missing CSV"}
            continue
        scores = read_csv(track(score_path))
        try:
            values = check_scores(rows, scores, k)
        except (ValueError, KeyError) as error:
            report["scores"][k] = {"accepted": False, "reason": str(error), "rows": len(scores)}
            report["warnings"].append(f"{k}-sum score omitted: {error}")
        else:
            score_values[k] = values
            report["scores"][k] = dict(accepted=True, positive_complete_n=len(values),
                                       median=statistics.median(values) if values else None,
                                       denominator_reconstructed=False)
    ids = {k: set(r["dep_hash"] for r in rows) for k, rows in populations.items()}
    remaining = {r["dep_hash"] for r in populations[1]
                 if not (int(r["matches_count"]) == 1 and r["truncated"] == "0")}
    report["cohorts"] = dict(two_equals_three=ids[2] == ids[3],
                             two_equals_one_without_singletons=ids[2] == remaining)
    if not all(report["cohorts"].values()):
        report["warnings"].append("Cohort relationship changed; review the paper interpretation")
    deposit_path = source / "data/processed/processed_privacypools_eth_pool_deposits.csv"
    deposit_rows = read_csv(track(deposit_path))
    missing = sorted({"tx_hash", "amount_eth_net"} - set(deposit_rows[0] if deposit_rows else {}))
    report["raw_input_missing_columns"] = missing
    if missing:
        report["warnings"].append("Matching cannot be rerun from supplied deposit CSV; missing columns: " + ", ".join(missing))
    report["scope"] = "Aggregate import and export consistency checks; not event reconstruction, timing verification, matching rerun, or funding validation. Score denominators are imported, not independently reconstructed."
    for warning in report["warnings"]:
        print("WARNING:", warning)
    if strict and report["warnings"]:
        raise ValueError("Strict validation failed; paper assets were not changed")
    # Validate and render in staging so failures leave previous outputs intact.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pp-refresh-", dir=output.parent) as tmp:
        stage = Path(tmp)
        render_outputs(stage, summaries, score_values, report)
        output.mkdir(parents=True, exist_ok=True)
        for path in stage.iterdir():
            path.replace(output / path.name)
    print(f"Generated PP assets in {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PAPER.parent / "privacypools-deanonymization-main")
    parser.add_argument("--output", type=Path, default=PAPER / "Figures/generated_pp")
    parser.add_argument("--strict", action="store_true", help="Fail on optional-input warnings without changing assets")
    args = parser.parse_args()
    try:
        refresh(args.source.resolve(), args.output.resolve(), args.strict)
    except (ValueError, KeyError, OSError, ImportError, zipfile.BadZipFile) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
