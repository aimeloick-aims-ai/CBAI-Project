"""Validate and preserve a returned annotation batch with its declared provenance."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\Users\Public\Downloads\Downloads")


def main():
    annotations = SOURCE / "ANNOTATIONS.csv"
    evaluator = SOURCE / "EVALUATEUR.json"
    metadata = json.loads(evaluator.read_text(encoding="utf-8-sig"))
    with annotations.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping_path = ROOT / "reports/expert_round_1_private_mapping.json"
    mapping = {r["review_id"]: r for r in json.loads(mapping_path.read_text())["mapping"]}
    ids = [r["image_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(mapping):
        raise ValueError("Missing, duplicate or unknown annotation identifiers")
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed CSV row")
        if row["interpretable_oui_non"].strip().lower() not in {"oui", "non"}:
            raise ValueError("Invalid interpretability value")
        if row["confiance_1_a_5"].strip() not in {"1", "2", "3", "4", "5"}:
            raise ValueError("Invalid confidence score")
        if not row["concept_principal"].strip() or not row["description_libre"].strip():
            raise ValueError("Missing concept or description")
    # This import is specific to the supplied return: never silently certify a reviewer.
    if metadata.get("evaluateur_id") != "AI_REV_GPT56SOL_R1":
        raise ValueError("Different reviewer: review provenance before creating a new import")
    out = ROOT / "reports/annotation_returns/round1_ai_review"
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in (annotations, evaluator):
        raw = path.read_bytes()
        (out / path.name).write_bytes(raw)
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
    joined = []
    for row in rows:
        linked = mapping[row["image_id"]]
        joined.append(
            {
                **row,
                "cluster": linked["cluster"],
                "patient_id": linked["patient_id"],
                "prototype_selection": linked["selection"],
                "annotation_origin": "AI_self_declared_non_pathologist",
                "independent_expert_validation": False,
            }
        )
    with (out / "linked_annotations.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)
    groups = []
    for cluster in range(5):
        subset = [r for r in joined if r["cluster"] == cluster]
        groups.append(
            {
                "cluster": cluster,
                "examples": len(subset),
                "patients": len({r["patient_id"] for r in subset}),
                "literal_label_counts": dict(
                    Counter(r["concept_principal"].strip() for r in subset)
                ),
                "confidence_counts": dict(Counter(r["confiance_1_a_5"] for r in subset)),
            }
        )
    summary = {
        "status": "imported_exploratory_ai_only",
        "rows": len(rows),
        "evaluator": metadata,
        "expert_validation": False,
        "blinded_review": False,
        "source_sha256": hashes,
        "mapping_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
        "groups": groups,
        "limitations": [
            "AI reviewer, not pathologist",
            "Prior exposure to group names",
            "Selected prototypes, not random purity sample",
            "Single reviewer",
            "Labels not semantically harmonized",
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Retour d'annotations : provenance IA",
        "",
        f"Les {len(rows)} identifiants attendus sont présents, uniques et reliés à leur groupe. Les champs obligatoires et scores ont été vérifiés. Les fichiers originaux sont conservés octet pour octet avec SHA256.",
        "",
        "L'utilisateur présente ce retour comme provenant du spécialiste. Toutefois EVALUATEUR.json déclare un relecteur IA non pathologiste et une exposition préalable aux noms/groupes. Ce conflit de provenance empêche de retenir une validation humaine indépendante. Aucun statut clinique ni nom officiel de concept n'est modifié.",
        "",
        "## Descriptions principales déclarées, sans fusion de synonymes",
        "",
        "Ces fréquences décrivent seulement les prototypes reçus ; elles ne mesurent ni la pureté des clusters dans la population, ni l'exactitude médicale.",
        "",
    ]
    for group in groups:
        lines += [f"### Cluster {group['cluster']} — {group['examples']} exemples", ""]
        lines.extend(
            f"- {label} : {count}" for label, count in group["literal_label_counts"].items()
        )
        lines.append("")
    lines += [
        "## Suite nécessaire",
        "",
        "Faire remplir ou réellement relire chaque annotation par l'anatomopathologiste. Enregistrer son identifiant pseudonyme, sa qualification, la date, les modifications et son exposition préalable aux suggestions. Ne pas seulement changer le champ qualification. Une revue humaine de propositions IA doit être déclarée comme assistée et non comme aveugle. Pour une validation indépendante, utiliser un nouveau lot sans suggestions, après fixation des définitions.",
        "",
        "Aucune preuve d'utilisation fonctionnelle clinique, d'accord inter-évaluateur ou de validation externe ne résulte de cet import.",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert all(
        hashlib.sha256((out / name).read_bytes()).hexdigest() == sha for name, sha in hashes.items()
    )
    print(
        json.dumps(
            {"rows": len(rows), "status": summary["status"], "groups": groups}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
