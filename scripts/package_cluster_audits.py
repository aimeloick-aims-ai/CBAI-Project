"""Package the exploratory manuscript with clearly scoped cluster supplements."""

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    structural = json.loads((ROOT / "reports/cluster_structure_audit/summary.json").read_text())
    features = json.loads((ROOT / "reports/candidate_cluster_audit/summary.json").read_text())
    folder = ROOT / "paper/cluster_supplement"
    folder.mkdir(exist_ok=True)
    lines = [
        r"\section{Supplementary exploratory audit of patch clusters}",
        "This supplementary analysis concerns the frozen patch GCN ensemble, not the cellular GNNs. Five previously fitted training-only clusters received AI-proposed histological names. These names have not been validated by a pathologist. The internal test cohort was already examined; structural analyses followed inspection of feature-occlusion results.",
        "Feature occlusion replaces assigned node embeddings with their training mean. Node removal deletes assigned nodes and incident edges. Boundary removal deletes edges between the assigned cluster and other clusters, retaining all nodes. Twenty random controls per patient match node count and original degree histogram, or undirected edge count, grid length and unordered endpoint degrees. Controls may overlap target elements. They do not match spatial contiguity or resulting connectedness; feature controls do not match perturbation norm.",
        r"\begin{center}\small\begin{tabular}{llrr} Intervention & Cluster & Patients & Excess drop [95\% CI], pp \\\hline",
    ]
    for s in features:
        lo, hi = s["difference"]["patient_bootstrap_95"]
        lines.append(
            f"Features & {s['cluster']} & {s['patients']} & {100 * s['difference']['mean']:.2f} [{100 * lo:.2f}, {100 * hi:.2f}] "
            + r"\\"
        )
    for s in structural:
        if not s["patients"]:
            continue
        lo, hi = s["difference"]["patient_bootstrap_95"]
        label = "Nodes" if s["mode"] == "node_removal" else "Boundary"
        lines.append(
            f"{label} & {s['cluster']} & {s['patients']} & {100 * s['difference']['mean']:.2f} [{100 * lo:.2f}, {100 * hi:.2f}] "
            + r"\\"
        )
    lines += [
        r"\end{tabular}\end{center}",
        "Positive values indicate greater loss of confidence in the originally predicted class than under matched controls. Intervals use paired patient bootstrap and are descriptive, without multiplicity adjustment. Cohorts differ by cluster presence; empty-graph interventions are excluded. Large node removals can yield out-of-distribution graphs.",
        "Cluster 0 shows pronounced sensitivity to its node content. The proposed interface cluster 4 does not show a clear excess boundary effect. Neither observation establishes dependence on a clinically validated concept. Feature occlusion increases the mean fraction assigned to cluster 2, invalidating its interpretation as concept suppression. Cluster fractions are compositional: changes cannot establish independence of other clinical concepts.",
        "Clinical type annotations, independent concept probes, selective image/cellular interventions, histological plausibility and external-cohort validation remain outstanding. No assertion of biological causality or priority over existing methods is made.",
        r"\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{figures/cluster_structure.pdf}\caption{Exploratory structural perturbations of patch clusters. Names are unvalidated.}\end{figure}",
    ]
    (folder / "cluster_supplement.tex").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    report = ROOT / "reports/BILAN_FINAL_PILOTE.md"
    report.write_text(
        """# Bilan du pilote GCIA

Le pilote technique dispose de résultats reproductibles. Le programme clinique complet de l'abstract n'est pas terminé.

## Réalisé

- Segmentation évaluée sur MoNuSeg, graphes cellulaires construits, GCN/SAGE/GAT entraînés ; résultats dans segmentation_evaluation et cell_gnn_comparison.
- GNNExplainer et audit de morphologie sur modèles cellulaires : cell_xai/RESULTS.md.
- Cinq clusters du GCN de patches, prototypes et noms proposés par IA : cluster_review. Pas de validation médicale ni de transfert automatique aux cellules.
- Trois perturbations des clusters : caractéristiques (candidate_cluster_audit), retrait des nœuds et retrait des arêtes de frontière (cluster_structure_audit). Tous les résultats, contrôles, exclusions et protocoles sont sauvegardés.
- Supplément LaTeX et archive du manuscrit exploratoire, sans remplacer l'archive historique.

## Conclusion permise

Le modèle est sensible à certains groupes de patches, notamment le cluster 0. Cela ne prouve pas qu'il utilise le concept histologique proposé. Le cluster 2 échoue au contrôle du sens de l'occlusion ; le cluster 4 ne montre pas d'effet de frontière clairement supérieur aux témoins. Les intervalles sont exploratoires, sans correction de multiplicité, sur un test déjà consulté.

## Ce qui reste nécessaire pour l'abstract complet

1. Annotations indépendantes des concepts et des types cellulaires : la proposition de noms par IA ne les remplace pas.
2. Correspondance validée entre concepts de patches et graphes cellulaires ; probes entraînées sur ces annotations, sans circularité K-means.
3. Interventions cellulaires et sur les images préservant les autres concepts, avec validation de plausibilité. Les retraits actuels ne satisfont pas cette exigence.
4. Évaluation indépendante, cohorte externe BACH et intégration des annotations NuCLS/BCSS selon le protocole clinique.
5. Revue bibliographique vérifiée avant toute revendication de nouveauté. Les références du texte joint ne sont pas toutes vérifiées ici.

## Lire et reproduire

- Résultats : candidate_cluster_audit/RESULTS.md et cluster_structure_audit/RESULTS.md.
- Figures : effects.png et effects.pdf dans ces deux dossiers.
- Code : scripts/audit_candidate_clusters.py, scripts/audit_cluster_structure.py, src/graphs/interventions.py.
- Tests : `.\\.venv\\Scripts\\python.exe -m pytest -q` depuis GCIA.
- Les scripts d'audit refusent d'écraser leurs dossiers existants. Pour une réplication, travailler dans une copie du projet en conservant les résultats originaux ; aucun téléchargement requis.
- Archive : paper/GCIA_pilot_with_cluster_audits.zip. Ouvrir main.tex dans Overleaf ; auteur et affiliation à compléter. Compilation LaTeX non vérifiée localement.

Les chiffres hypothétiques du document joint ne sont pas des résultats du projet. La finalisation d'un pilote ne signifie pas validation de l'ensemble de l'hypothèse clinique.
""",
        encoding="utf-8",
    )
    archive = ROOT / "paper/GCIA_pilot_with_cluster_audits.zip"
    paper = ROOT / "paper/gcia_pilot"
    included = {}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(paper.rglob("*")):
            if path.is_file():
                name = path.relative_to(paper).as_posix()
                data = path.read_bytes()
                if name == "main.tex":
                    text = data.decode("utf-8")
                    assert text.count(r"\end{document}") == 1
                    data = text.replace(
                        r"\end{document}", "\\input{cluster_supplement.tex}\n\\end{document}"
                    ).encode("utf-8")
                z.writestr(name, data)
                included[name] = hashlib.sha256(data).hexdigest()
        additions = [
            (folder / "cluster_supplement.tex", "cluster_supplement.tex"),
            (ROOT / "reports/cluster_structure_audit/effects.pdf", "figures/cluster_structure.pdf"),
            (report, "BILAN_FINAL_PILOTE.md"),
        ]
        for directory in ("candidate_cluster_audit", "cluster_structure_audit"):
            additions.extend(
                (p, f"data/{directory}/{p.name}")
                for p in (ROOT / "reports" / directory).iterdir()
                if p.is_file()
            )
        for path, name in additions:
            data = path.read_bytes()
            z.writestr(name, data)
            included[name] = hashlib.sha256(data).hexdigest()
        z.writestr(
            "cluster_package_manifest.json",
            json.dumps({"sha256": included, "latex_compiled": False}, indent=2),
        )
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(
            hashlib.sha256(z.read(name)).hexdigest() == sha for name, sha in included.items()
        )
    print(f"Verified package: {archive}")


if __name__ == "__main__":
    main()
