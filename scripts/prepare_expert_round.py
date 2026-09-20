"""Create a blinded discovery round, not an unbiased purity evaluation."""

import csv
import hashlib
import json
import random
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    provenance = json.loads((ROOT / "reports/cluster_review_provenance.json").read_text())
    examples = provenance["selection"].copy()
    random.Random(20260920).shuffle(examples)
    out = ROOT / "reports/expert_round_1"
    out.mkdir(exist_ok=False)
    mapping = []
    fields = [
        "image_id",
        "description_libre",
        "concept_principal",
        "autres_concepts",
        "interpretable_oui_non",
        "confiance_1_a_5",
        "commentaire",
    ]
    with (out / "ANNOTATIONS.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for number, item in enumerate(examples, 1):
            identifier = f"R{number:03d}"
            writer.writerow({"image_id": identifier})
            folder = out / identifier
            folder.mkdir()
            for suffix in ("patch", "original", "contexte"):
                source = (
                    ROOT
                    / "reports/cluster_review"
                    / f"cluster_{item['cluster']}"
                    / f"{item['example']}_{suffix}.png"
                )
                shutil.copyfile(source, folder / f"{suffix}.png")
            mapping.append({"review_id": identifier, **item})
    (out / "LIRE.md").write_text(
        """# Première revue indépendante — découverte des concepts

Objectif : décrire les motifs observables, sans connaître les groupes du modèle ni ses prédictions.

Pour chaque identifiant Rxxx, examiner original.png puis contexte.png ; patch.png représente la vue utilisée par le modèle. Compléter la ligne correspondante dans ANNOTATIONS.csv. Les images ne comportent pas d'échelle physique certifiée.

- description_libre : observations morphologiques concrètes ; ne pas forcer un diagnostic.
- concept_principal : votre propre nom du motif dominant, ou indéterminé.
- autres_concepts : motifs coexistants séparés par un point-virgule.
- interpretable_oui_non : oui ou non.
- confiance_1_a_5 : 1 faible, 5 forte ; ne pas laisser une impression de précision injustifiée.
- commentaire : ambiguïtés, artefacts et informations manquantes.

Compléter EVALUATEUR.json avec un identifiant pseudonyme, la date et l'exposition éventuelle aux anciens noms ou planches. Ne pas inclure de données personnelles de patients. Ne pas rechercher les prédictions du modèle. Conserver les identifiants Rxxx.

Ce premier lot contient des exemples choisis pour explorer les groupes ; il ne mesure pas sans biais leur pureté dans la cohorte. Les exemples peuvent partager un patient. Aucun test nouveau n'est ouvert. Les définitions seront fixées après cette revue, puis évaluées sur un échantillon distinct choisi aléatoirement. Avec un seul évaluateur, aucun accord inter-évaluateur ne sera calculé.

Retourner ANNOTATIONS.csv et EVALUATEUR.json. Les noms de concepts proposés par IA ne font pas partie de ce lot.
""",
        encoding="utf-8",
    )
    (out / "EVALUATEUR.json").write_text(
        json.dumps(
            {
                "evaluateur_id": "",
                "date": "",
                "qualification": "",
                "a_deja_vu_noms_IA_ou_groupes": None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    private = ROOT / "reports/expert_round_1_private_mapping.json"
    private.write_text(
        json.dumps(
            {"purpose": "researcher-only linkage; do not send to reviewer", "mapping": mapping},
            indent=2,
        ),
        encoding="utf-8",
    )
    archive = ROOT / "reports/expert_round_1_blinded.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(out).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any("cluster" in name or "mapping" in name for name in z.namelist())
    (ROOT / "reports/expert_round_1_receipt.json").write_text(
        json.dumps(
            {
                "examples": len(examples),
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "expert_completed": False,
            },
            indent=2,
        )
    )
    print(f"Prepared {len(examples)} blinded examples: {archive}")


if __name__ == "__main__":
    main()
