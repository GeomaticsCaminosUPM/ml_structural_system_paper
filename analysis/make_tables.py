"""Tables 1-4 of the paper as LaTeX ``tabular`` environments.

Reads the JSON files written by the ``results`` step of reproduce_paper.py (and,
for Table 4, the per-building predictions of experiments/all_cities) and
writes results/tables/table{1,2,3,4}_*.tex. Tables whose inputs are missing
are skipped with a message.
"""
from __future__ import annotations

import json
from pathlib import Path

from results_analysis import load_class_names, load_sample_df, majority_baseline, model_metrics

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "tables"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = {  # display order: alphabetical, as in the paper
    "CatBoost": "CatBoost",
    "HistGradientBoosting": r"Hist.\ GB",
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
}
CITY_NAMES = {"Guatemala": "Guatemala City", "San Jose": r"San Jos\'e", "Santo Domingo": "Santo Domingo"}
BASE_EXP = {"Guatemala": "guatemala", "San Jose": "san_jose", "Santo Domingo": "santo_domingo"}
LOO = {"Guatemala": "loo_guatemala", "San Jose": "loo_san_jose", "Santo Domingo": "loo_santo_domingo"}
DISTRICT = {"Guatemala": "zona1_to_zona10", "San Jose": "mata_redonda_to_san_jose_sample",
            "Santo Domingo": "ensanche_quisquella_to_santo_domingo_center"}


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def f2(v: float) -> str:
    return f"{v:.2f}"


def ci(c: list[float]) -> str:
    return f"[{c[0]:.2f}, {c[1]:.2f}]"


def signed(v: float) -> str:
    return f"${'-' if v < 0 else '+'}{abs(v):.2f}$"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def emph_row(cells: list[str]) -> str:
    """Italic table row (majority baseline); empty cells stay empty."""
    return " & ".join(rf"\emph{{{c}}}" if c else "" for c in cells) + r" \\"


def write(name: str, lines: list[str]) -> None:
    path = OUT / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n")
    print("written", path)


def table1() -> None:
    stats = load("data_stats")
    if stats is None:
        print("Table 1 skipped: run data_stats.py first")
        return
    classes = ["ADO", "CR", "M", "S"]
    header = [c if c != "S" else "S/W" for c in classes]  # code S is light steel and wood
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", "City & $N$ & " + " & ".join(header) + r" \\", r"\midrule"]
    for key, label in [*CITY_NAMES.items(), ("All", "All three cities")]:
        if key == "All":
            lines.append(r"\midrule")
        counts = stats["class_counts"][key]
        n = sum(counts.values())
        cells = [f"{thousands(counts[c])} ({round(100 * counts[c] / n)}\\%)" for c in classes]
        lines.append(f"{label} & {thousands(n)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table1_class_imbalance", lines)


def table2() -> None:
    res, boot = load("results_analysis"), load("block_bootstrap_ci")
    if res is None or boot is None:
        print("Table 2 skipped: run the results step first")
        return
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{Macro F1} & & & \\", r"\cmidrule(lr){3-4}",
             r"City & Model & Value & 95\% CI & Acc. & $\kappa$ & TSS \\"]
    for city, exp in BASE_EXP.items():
        lines.append(r"\midrule")
        entry = res["base_cities"][city]
        by_model = {r["model"]: r for r in entry["ranking"]}
        best = entry["ranking"][0]["model"]
        for i, (m, label) in enumerate(MODEL_NAMES.items()):
            r = by_model[m]
            f1 = rf"\textbf{{{f2(r['macro_f1'])}}}" if m == best else f2(r["macro_f1"])
            first = CITY_NAMES[city] if i == 0 else ""
            lines.append(f"{first} & {label} & {f1} & {ci(boot[exp]['models'][m]['ci95'])} & "
                         f"{f2(r['accuracy'])} & {f2(r['cohen_kappa'])} & {f2(r['tss_macro'])} \\\\")
        maj = entry["majority_baseline"]
        lines.append(emph_row(["", "Majority baseline", f2(maj["macro_f1"]), "--", f2(maj["accuracy"]),
                               f2(maj["cohen_kappa"]), f2(maj["tss_macro"])]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table2_base_cities", lines)


def table3() -> None:
    pcm, boot = load("present_class_metrics"), load("block_bootstrap_ci")
    if pcm is None or boot is None:
        print("Table 3 skipped: run the results step first")
        return
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{Macro F1} & & & \\", r"\cmidrule(lr){3-4}",
             r"Transfer & Best model & Value & 95\% CI & $\Delta$F1 & $\kappa$ & TSS \\", r"\midrule"]
    for title, experiments in [(r"Leave-one-city-out (trained on the other two cities)", LOO),
                               (r"Single-district survey $\to$ larger-area survey, same city", DISTRICT)]:
        if experiments is DISTRICT:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{title}}}}} \\")
        for city, exp in experiments.items():
            t = pcm["transfer"][exp]
            best = t["ranking"][0]
            m = best["model"]
            lines.append(f"{CITY_NAMES[city]} & {MODEL_NAMES[m]} & {f2(best['macro_f1'])} & "
                         f"{ci(boot[exp]['models'][m]['ci95'])} & {signed(t['delta_f1']['delta'])} & "
                         f"{f2(best['cohen_kappa'])} & {f2(best['tss_macro'])} \\\\")
            maj = t["majority_baseline"]
            lines.append(emph_row(["", "Majority baseline", f2(maj["macro_f1"]), "--", "--",
                                   f2(maj["cohen_kappa"]), f2(maj["tss_macro"])]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table3_transfer", lines)


def table4() -> None:
    pcm, boot = load("present_class_metrics"), load("block_bootstrap_ci")
    if pcm is None or boot is None:
        print("Table 4 skipped: run the results step first")
        return
    df = load_sample_df("all_cities")
    labels = sorted(load_class_names("all_cities"))
    per_city = pcm["pooled_per_city"]
    overall = {m: boot["all_cities"]["models"][m]["macro_f1"] for m in MODEL_NAMES}
    kappa = {m: model_metrics(df, m, labels)["cohen_kappa"] for m in MODEL_NAMES}
    cols = [("All", overall)] + [(c, {m: per_city[c]["models"][m]["macro_f1"] for m in MODEL_NAMES})
                                 for c in CITY_NAMES]
    best = {name: max(vals.values()) for name, vals in cols}

    def cell(name: str, vals: dict, m: str) -> str:  # bold the best per column, exact ties included
        return rf"\textbf{{{f2(vals[m])}}}" if abs(vals[m] - best[name]) < 1e-9 else f2(vals[m])

    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r" & \multicolumn{5}{c}{Macro F1} & \\",
             r"\cmidrule(lr){2-6}", r"Model & All & 95\% CI & GT & SJ & SD & $\kappa$ \\", r"\midrule"]
    for m, label in MODEL_NAMES.items():
        lines.append(f"{label} & {cell('All', overall, m)} & {ci(boot['all_cities']['models'][m]['ci95'])} & "
                     + " & ".join(cell(c, vals, m) for c, vals in cols[1:]) + f" & {f2(kappa[m])} \\\\")
    maj = majority_baseline(df, labels)
    lines.append(emph_row(["Majority baseline", f2(maj["macro_f1"]), "--", "--", "--", "--",
                           f2(maj["cohen_kappa"])]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("table4_pooled", lines)


if __name__ == "__main__":
    table1()
    table2()
    table3()
    table4()
