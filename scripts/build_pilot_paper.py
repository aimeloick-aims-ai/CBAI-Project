"""Build figures, data-derived tables and a self-contained LaTeX manuscript archive."""

import csv
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/gcia_pilot"
COLORS = ["#0072B2", "#D55E00", "#009E73"]


def read(name):
    return json.loads((ROOT / name).read_text())


def save(fig, name):
    for extension in ("pdf", "png"):
        fig.savefig(PAPER / "figures" / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    (PAPER / "tables").mkdir(exist_ok=True)
    (PAPER / "data").mkdir(exist_ok=True)
    xai = read("reports/cell_xai/summary.json")
    coherent = read("reports/cell_xai/coherent_summary.json")
    classification = read("reports/cell_gnn_comparison/results.json")
    segmentation = read("reports/segmentation_evaluation/summary.json")
    names = list(xai["results"])
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), layout="constrained", sharey=True)
    for ax, name, color in zip(axes, names, COLORS, strict=True):
        budgets, means, lo, hi = [], [], [], []
        for fraction, r in xai["results"][name]["gnnexplainer"].items():
            budgets.append(100 * float(fraction))
            d = r["difference"]
            means.append(100 * d["mean"])
            lo.append(100 * d["patient_bootstrap_95"][0])
            hi.append(100 * d["patient_bootstrap_95"][1])
        ax.errorbar(
            budgets,
            means,
            yerr=[np.array(means) - lo, np.array(hi) - means],
            fmt="o-",
            capsize=4,
            color=color,
        )
        ax.axhline(0, c="gray", lw=1, ls="--")
        ax.set(title=name, xlabel="Replaced nodes (%)", xticks=[5, 10, 20])
    axes[0].set_ylabel("GNNExplainer minus random\nconfidence drop (percentage points)")
    save(fig, "gnnexplainer")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained", sharey=True)
    for ax, factor in zip(axes, ["0.95", "1.05"], strict=True):
        for i, (name, color) in enumerate(zip(names, COLORS, strict=True)):
            d = coherent[name][factor]["target_minus_coherent_control"]
            value = 100 * d["mean"]
            lo, hi = np.array(d["patient_bootstrap_95"]) * 100
            ax.errorbar(
                i, value, yerr=[[value - lo], [hi - value]], fmt="o", capsize=5, color=color
            )
        ax.axhline(0, c="gray", ls="--", lw=1)
        ax.set(
            title=f"Nuclear area {(float(factor) - 1) * 100:+.0f}% (axes coupled)",
            xticks=range(3),
            xticklabels=names,
            xlim=(-0.5, 2.5),
        )
    axes[0].set_ylabel("Target minus coherent control\nabsolute P(IC) change (percentage points)")
    save(fig, "gcia_controls")
    fig, ax = plt.subplots(figsize=(6, 3.2), layout="constrained")
    matrix = np.array(
        [
            [xai["results"][n]["probe_r2"][c] for c in ["area", "eccentricity", "solidity"]]
            for n in names
        ]
    )
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set(
        xticks=range(3),
        xticklabels=["Mean area", "Mean eccentricity", "Mean solidity"],
        yticks=range(3),
        yticklabels=names,
        title="Hidden-representation probes: test-patient R²",
    )
    for (y, x), v in np.ndenumerate(matrix):
        ax.text(x, y, f"{v:.3f}", ha="center", va="center", color="white" if v > 0.65 else "black")
    fig.colorbar(im, ax=ax, label="R²")
    save(fig, "probes")
    rows = list(csv.DictReader((ROOT / "reports/segmentation_evaluation/metrics.csv").open()))
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4), layout="constrained", sharey=True)
    ids = sorted({r["image"] for r in rows})
    for ax, key, title in zip(
        axes,
        ["foreground_dice", "detection_f1", "panoptic_quality"],
        ["Foreground Dice", "Detection F1", "Panoptic quality"],
        strict=True,
    ):
        for pid in ids:
            values = [
                float(next(r[key] for r in rows if r["image"] == pid and r["method"] == m))
                for m in ["watershed", "hovernet"]
            ]
            ax.plot([0, 1], values, "o-", color="#0072B2", alpha=0.4, ms=3)
        ax.set(title=title, xticks=[0, 1], xticklabels=["Watershed", "HoVer-Net"], ylim=(0, 1.03))
    axes[0].set_ylabel("Per-image score (14 complete images)")
    save(fig, "segmentation")

    def table(name, header, body):
        (PAPER / "tables" / name).write_text(
            header + "\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8"
        )

    table(
        "classification.tex",
        r"\begin{tabular}{lrrr}\toprule Model & Accuracy (\%) & Balanced accuracy (\%) & AUC \\ \midrule",
        [
            f"{n} & {100 * r['accuracy']:.1f} & {100 * r['balanced_accuracy']:.1f} & {r['auc']:.3f} \\\\"
            for n, r in classification["test"].items()
        ],
    )
    table(
        "gnnexplainer.tex",
        r"\begin{tabular}{lrrrr}\toprule Model & Nodes (\%) & Explainer & Random & Difference [95\% CI] \\ \midrule",
        [
            f"{n} & {100 * float(f):.0f} & {100 * r['gnnexplainer']:.2f} & {100 * r['random']:.2f} & {100 * r['difference']['mean']:.2f} [{100 * r['difference']['patient_bootstrap_95'][0]:.2f}, {100 * r['difference']['patient_bootstrap_95'][1]:.2f}] \\\\"
            for n in names
            for f, r in xai["results"][n]["gnnexplainer"].items()
        ],
    )
    table(
        "gcia.tex",
        r"\begin{tabular}{lrrrr}\toprule Model & Area change & Target & Coherent control & Difference [95\% CI] \\ \midrule",
        [
            f"{n} & {(float(f) - 1) * 100:+.0f}\\% & {100 * xai['results'][n]['gcia'][f]['gcia_morphology']:.3f} & {100 * r['coherent_control_mean_absolute_change']:.3f} & {100 * r['target_minus_coherent_control']['mean']:.3f} [{100 * r['target_minus_coherent_control']['patient_bootstrap_95'][0]:.3f}, {100 * r['target_minus_coherent_control']['patient_bootstrap_95'][1]:.3f}] \\\\"
            for n in names
            for f, r in coherent[n].items()
        ],
    )
    sources = [
        "reports/cell_xai/summary.json",
        "reports/cell_xai/coherent_summary.json",
        "reports/cell_xai/protocol.json",
        "reports/cell_xai/coherent_control_protocol.json",
        "reports/cell_xai/deletions.csv",
        "reports/cell_xai/interventions.csv",
        "reports/cell_xai/coherent_controls.csv",
        "reports/cell_xai/probes.csv",
        "reports/cell_gnn_comparison/results.json",
        "reports/cell_gnn_comparison/predictions.csv",
        "reports/segmentation_evaluation/summary.json",
        "reports/segmentation_evaluation/metrics.csv",
        "reports/cell_cohort/completion.json",
        "reports/cell_cohort/excluded.csv",
    ]
    index = []
    for name in sources:
        path = ROOT / name
        dest = PAPER / "data" / (path.parent.name + "_" + path.name)
        shutil.copyfile(path, dest)
        index.append(
            {
                "source": name,
                "archive_path": str(dest.relative_to(PAPER)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (PAPER / "data/source_index.json").write_text(json.dumps(index, indent=2))
    (PAPER / "README.md").write_text(
        "# GCIA exploratory pilot manuscript\n\nEnglish research draft; author and affiliation information must be completed.\nImport this ZIP into Overleaf, select main.tex, compile with pdfLaTeX (twice).\nNo bibliography processor is needed. Figures are included as vector PDF and 300-dpi PNG.\nThe data folder contains the recorded numerical results and SHA256 provenance.\nNo raw histology images or model weights are redistributed.\n\nThis is not a submission-ready clinical validation. BACH and expert review were not performed; no claim of independently validated clinical concept use is supported.\nNo local LaTeX compiler was available when creating the archive; source checks are not PDF compilation.\n\nRecreate figures/tables/archive from the GCIA repository using scripts/build_pilot_paper.py. The main.tex manuscript is maintained separately.\n",
        encoding="utf-8",
    )
    # The manuscript must exist before packaging; refuse a figures-only deliverable.
    if not (PAPER / "main.tex").exists():
        raise FileNotFoundError("Write main.tex before packaging")
    manuscript = (PAPER / "main.tex").read_text(encoding="utf-8")
    for figure in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", manuscript):
        if not (PAPER / figure).is_file():
            raise FileNotFoundError(figure)
    for table_path in re.findall(r"\\input\{([^}]+)\}", manuscript):
        if not (PAPER / table_path).is_file():
            raise FileNotFoundError(table_path)
    citations = {
        k.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", manuscript)
        for k in group.split(",")
    }
    bibliography = set(re.findall(r"\\bibitem\{([^}]+)\}", manuscript))
    if citations - bibliography:
        raise ValueError("Undefined citations")
    (PAPER / "validation.json").write_text(
        json.dumps(
            {
                "figure_dependencies_checked": 4,
                "table_dependencies_checked": 3,
                "citations_resolved": len(citations),
                "latex_compilation": "not performed: no local compiler",
                "raw_numerical_sources": len(index),
            },
            indent=2,
        )
    )
    archive = ROOT / "paper/GCIA_pilot_latex.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(PAPER.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(PAPER).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(
        json.dumps(
            {
                "archive": str(archive),
                "bytes": archive.stat().st_size,
                "figures": 4,
                "segmentation": segmentation["methods"]["hovernet"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
