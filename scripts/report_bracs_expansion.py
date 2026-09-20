"""Summarize a completed acquisition without modifying historical data splits."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    out = ROOT / "reports/bracs_expansion"
    audit = json.loads((out / "acquisition.json").read_text())
    if audit["status"] != "complete":
        raise ValueError("Acquisition not complete; cannot issue final report")
    rows = read(ROOT / "data/bracs_expansion_manifest.csv")
    selected = {r["image_id"] for r in rows}
    slides = {r["slide_id"]: r["patient_id"] for r in read(ROOT / "data/bracs_slide_metadata.csv")}
    prior = set()
    all_patients = set()
    files = list((ROOT / "data/raw/BRACS/RoI").rglob("*.png"))
    for path in files:
        slide = "_".join(path.stem.split("_")[:2])
        if slide not in slides:
            raise ValueError("Unknown acquired slide")
        all_patients.add(slides[slide])
        if path.name not in selected:
            prior.add(slides[slide])
    patients = {r["patient_id"] for r in rows}
    data = {
        **audit,
        "patients_in_new_images": len(patients),
        "new_patient_ids_relative_to_previous_local_roi": sorted(patients - prior),
        "new_patients": len(patients - prior),
        "local_bracs_roi": len(files),
        "local_bracs_patients": len(all_patients),
        "new_normal_images": sum(r["roi_label"] == "N" for r in rows),
        "new_invasive_images": sum(r["roi_label"] == "IC" for r in rows),
    }
    (out / "completed_summary.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    lines = [
        "# Extension BRACS terminee",
        "",
        f"- Nouvelles images : {len(rows)} (normal : {data['new_normal_images']}, invasif : {data['new_invasive_images']}).",
        f"- Volume verifie de cette extension : {audit['verified_bytes']:,} octets.",
        f"- Patients couverts : {len(patients)}, dont {data['new_patients']} absents des RoI locales anterieures.",
        f"- Total local BRACS : {len(files)} RoI de {len(all_patients)} patients.",
        "",
        "Plafond autorise : 1 000 000 000 octets. Controle taille serveur, empreinte SHA256",
        "locale et verification PNG pour chaque fichier. Les empreintes locales ne sont",
        "pas des empreintes publiees par le fournisseur. Identifiants FTP non enregistres.",
        "",
        "Selection : ordre hash fixe, parcours par patient, filtrage par budget,",
        "sans choix selon les scores. La contrainte de taille peut introduire un biais.",
        "",
        "Les classes concernent les regions, pas necessairement le diagnostic du patient.",
        "Les nouvelles images de patients deja examines ne sont pas de nouveaux patients",
        "de test independants. Aucun entrainement ni nouveau decoupage effectue ici.",
        "Le pilote historique a 40 patients et ses predictions restent conserves.",
        "",
        "Manifest : data/bracs_expansion_manifest.csv. Reçus : fichiers *.receipt.json.",
        "Audit et comptages : completed_summary.json.",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
