"""Write report/numbers.tex: every number quoted in the report, as a LaTeX macro.

Run after the experiments (``make report`` does it). Nothing in the report text
is typed by hand, so re-running the pipeline updates the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).with_name("numbers.tex")


def load(name: str):
    path = RES / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}\\%"


def f3(x: float) -> str:
    return f"{x:.3f}"


def pval(p: float) -> str:
    """Relation and value, used as $p\\Macro{}$ in the text: '<0.001' or '=0.04'."""
    if p < 0.001:
        return "<0.001"
    return f"={p:.3f}" if p < 0.01 else f"={p:.2f}"


def by(rows: list[dict], **match) -> dict:
    for r in rows:
        if all(r.get(k) == v for k, v in match.items()):
            return r
    raise KeyError(match)


def macros() -> dict[str, str]:
    m: dict[str, str] = {}

    eda = load("e00_eda")
    if eda:
        lc = eda["label_counts"]
        m["EdaIllicitShare"] = pct(lc["illicit"] / (lc["illicit"] + lc["licit"]))
        m["EdaShareTrain"] = pct(eda["illicit_share_train"])
        m["EdaShareTest"] = pct(eda["illicit_share_test"])
        m["EdaIllicitPost"] = f"{eda['illicit_after_shutdown_43_49']:,}"
        m["EdaIllicitPre"] = f"{eda['illicit_test_35_42']:,}"
        m["EdaComponents"] = str(eda["graph"]["weak_components"])
        m["EdaPairsNinetyFive"] = str(eda["correlation"]["pairs_abs_r_gt_0.95"])
        m["EdaFeaturesAfterFilter"] = str(eda["correlation"]["features_after_0.95_filter"])
        h = eda["homophily"]
        m["EdaHomIll"] = pct(h["p_illicit_neighbour_given_illicit"])
        m["EdaHomLic"] = pct(h["p_illicit_neighbour_given_licit"])
        m["EdaHomBase"] = pct(h["base_rate_illicit"])

    e1 = load("e01_reproduce")
    if e1:
        for r in e1["summary"]:
            short = r["Model"].replace("RandomForest", "RF").replace("LogReg", "LR")
            key = f"Rep{short}{r['Features']}"
            m[key] = f3(r["F1"])
            if r["F1 sd"] == r["F1 sd"]:  # not NaN
                m[key + "Sd"] = f3(r["F1 sd"])

    e2 = load("e02_models")
    if e2:
        names = {"RandomForest": "RF", "XGBoost": "XGB", "MLP": "MLP", "LogReg": "LR"}
        for r in e2["summary"]:
            k = names[r["Model"]]
            m[f"Main{k}Fone"] = f3(r["F1"])
            m[f"Main{k}FoneCI"] = r["F1 95% CI"].replace(",", ",\\,")
            m[f"Main{k}PRAUC"] = f3(r["PR-AUC"])
            m[f"Main{k}PRAUCCI"] = r["PR-AUC 95% CI"].replace(",", ",\\,")
            m[f"Main{k}Recall"] = f3(r["Recall"])
            m[f"Main{k}Precision"] = f3(r["Precision"])
            m[f"Main{k}Budget"] = pct(r["Recall@100/step"], 0)
        for p in e2["paired"]:
            a, b = p["Comparison"].split(" vs ")
            k = f"Paired{names[a]}{names[b]}"
            m[k + "FoneP"] = pval(p["F1 p"])
            m[k + "PRAUCP"] = pval(p["PR-AUC p"])
            m[k + "FoneDiff"] = f3(p["F1 diff"])

    e3 = load("e03_imbalance")
    if e3:
        names = {"RandomForest": "RF", "XGBoost": "XGB", "MLP": "MLP", "LogReg": "LR"}
        for r in e3["summary"]:
            k = f"Imb{names[r['Model']]}{r['Imbalance'].capitalize()}"
            m[k + "FixedFone"] = f3(r["F1 @0.5"])
            m[k + "TunedFone"] = f3(r["F1 @tuned"])
            m[k + "PRAUC"] = f3(r["PR-AUC"])

    e4 = load("e04_graph")
    if e4:
        for r in e4["summary"]:
            feat = r["Features"].replace("+", "").replace(" (GNN)", "").replace(" ", "")
            k = f"Graph{r['Model'].replace('RandomForest', 'RF').replace('XGBoost', 'XGB')}{feat}"
            m[k + "Fone"] = f3(r["F1"])
            m[k + "PRAUC"] = f3(r["PR-AUC"])
            m[k + "FoneCI"] = r["F1 95% CI"].replace(",", ",\\,")
        for p in e4["paired"]:
            k = f"GraphPaired{p['Model'].replace('RandomForest', 'RF').replace('XGBoost', 'XGB')}"
            k += p["Variant vs AF"].replace("+", "")
            m[k + "PRAUCDiff"] = f3(p["PR-AUC diff"])
            m[k + "PRAUCP"] = pval(p["PR-AUC p"])
            m[k + "FoneDiff"] = f3(p["F1 diff"])
            m[k + "FoneP"] = pval(p["F1 p"])

    e5 = load("e05_hybrid")
    if e5:
        for r in e5["summary"]:
            k = "Hyb" + r["Model"].replace("RandomForest", "RF").replace("XGBoost", "XGB")
            m[k + "Fone"] = f3(r["F1"])
            m[k + "PRAUC"] = f3(r["PR-AUC"])
        for p in e5["paired"]:
            k = "Hyb" + p["Model"].replace("RandomForest", "RF").replace("XGBoost", "XGB")
            m[k + "FoneDiff"] = f3(p["F1 diff"])
            m[k + "FoneP"] = pval(p["F1 p"])
            m[k + "FoneDrop"] = f3(abs(p["F1 diff"]))
    e5b = load("e05b_memorisation")
    if e5b:
        for r in e5b["rows"]:
            k = "Mem" + {"gcn": "GCN", "dirsage": "DirSAGE"}[r["GNN"]]
            m[k + "TrainPRAUC"] = f"{r['GNN PR-AUC train rows']:.4f}"
            m[k + "TestPRAUC"] = f3(r["GNN PR-AUC test rows"])
            m[k + "RFImp"] = pct(r["RF importance on embeddings"], 1)

    e6 = load("e06_unsupervised")
    if e6:
        m["UnsBaseRate"] = f3(e6["test_base_rate"])
        for r in e6["isolation_forest"]:
            k = f"IF{r['Features']}{'Unsup' if r['Fitted on'].startswith('all') else 'Licit'}"
            k += "Typical" if r["Flag as illicit"].endswith("typical") else "Anomalous"
            m[k + "PRAUC"] = f3(r["PR-AUC"])
            m[k + "ROCAUC"] = f3(r["ROC-AUC"])
        curve = e6["label_efficiency"]
        import collections

        agg = collections.defaultdict(list)
        n_by_frac = {}
        for r in curve:
            agg[r["fraction"]].append(r["pr_auc"])
            n_by_frac[r["fraction"]] = r["n_labels"]
        means = {f: sum(v) / len(v) for f, v in agg.items()}
        unsup = by(
            e6["isolation_forest"],
            Features="AF",
            **{"Fitted on": "all rows, no labels", "Flag as illicit": "most typical"},
        )
        beat = [f for f in sorted(means) if means[f] > unsup["PR-AUC"]]
        if beat:
            m["LabelsToBeatIF"] = f"{n_by_frac[beat[0]]:,}"
        m["LabelsFullPRAUC"] = f3(means[max(means)])
        # Macro names cannot contain digits: fractions are lettered A, B, ... in order.
        for i, f in enumerate(sorted(means)):
            m[f"LabelEff{chr(65 + i)}"] = f3(means[f])
            m[f"LabelEffN{chr(65 + i)}"] = f"{n_by_frac[f]:,}"

    e7 = load("e07_drift")
    if e7:
        for model, v in e7["mean_f1_steps_35_42"].items():
            m[f"Drift{model.replace('RandomForest', 'RF').replace('XGBoost', 'XGB')}Pre"] = f3(v)
        for model, v in e7["mean_f1_steps_43_49"].items():
            m[f"Drift{model.replace('RandomForest', 'RF').replace('XGBoost', 'XGB')}Post"] = f3(v)
        adv = e7["adversarial_validation"]
        for variant, tag in (("with_clock_features", "Clock"), ("without_clock_features", "")):
            for cls in ("illicit", "licit"):
                a = adv[variant][cls]
                k = f"Adv{cls.capitalize()}{tag}"
                m[k + "AUC"] = f3(a["auc_before_vs_after_43"])
                m[k + "KSShare"] = pct(a["share_features_ks_gt_0.3"], 0)
        m["AdvIllicitN"] = str(adv["without_clock_features"]["illicit"]["n_each_period"])
        m["AdvLicitN"] = f"{adv['without_clock_features']['licit']['n_each_period']:,}"
        clocks = e7["clock_features"]
        m["ClockCount"] = str(len(clocks))
        m["ClockList"] = ", ".join(c.replace("_", "\\_") for c in clocks)
        m["ClockMinRho"] = f3(min(clocks.values()))
        for r in e7["clock_ablation"]:
            if r["Features"] != "AF":
                k = f"ClockAbl{r['Model'].replace('RandomForest', 'RF').replace('XGBoost', 'XGB')}"
                m[k + "FoneDiff"] = f3(r["F1 diff vs AF"])
                m[k + "FoneP"] = pval(r["F1 p"])
                m[k + "PRAUCDiff"] = f3(r["PR-AUC diff vs AF"])
                m[k + "PRAUCP"] = pval(r["PR-AUC p"])
                m[k + "Fone"] = f3(r["F1"])
        nv = e7["novelty"]
        m["NovelPre"] = pct(nv["share_closer_to_licit_35_42"], 0)
        m["NovelPost"] = pct(nv["share_closer_to_licit_43_49"], 0)
        for r in e7["walk_forward"]:
            s = r["strategy"]
            k = (
                "Static"
                if s.startswith("static")
                else "Expanding"
                if "expanding" in s
                else "Sliding"
            )
            m[f"WF{k}Pre"] = f3(r["35-42"])
            m[f"WF{k}Post"] = f3(r["43-49"])

    e8 = load("e08_explain")
    if e8:
        sh = e8["shap_group_share"]
        m["ShapLocal"] = pct(sh["local"], 0)
        m["ShapAgg"] = pct(sh["aggregated"], 0)
        m["ShapStruct"] = pct(sh["structure"], 0)
        for r in e8["group_permutation"]:
            m[f"Perm{r['group'].capitalize()}"] = f3(r["pr_auc_drop_mean"])
        m["ShapTopFeature"] = e8["top_features"][0]["feature"].replace("_", "\\_")
    return m


def main() -> None:
    lines = ["% Generated by build_numbers.py from results/*.json. Do not edit.\n"]
    for k, v in sorted(macros().items()):
        if not k.isalpha():
            raise ValueError(f"LaTeX macro names must be letters only: {k}")
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    OUT.write_text("".join(lines))
    print(f"wrote {len(lines) - 1} macros to {OUT}")


if __name__ == "__main__":
    main()
