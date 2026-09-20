"""Package the pilot with a separately attributed report of specialist confirmation."""

import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "paper/GCIA_pilot_with_cluster_audits.zip"
    returned = ROOT / "reports/annotation_returns/round1_ai_review"
    acknowledgment = {
        "confirmation_source": "user_statement_in_conversation",
        "user_statement": "le specilaliste m'a rassure de tout confirmer ,IA + specialiste c'est le rendu",
        "review_type": "AI_generated_annotations_with_specialist_confirmation_reported_by_user",
        "scope": "returned 68-example annotation batch",
        "independent_blinded_validation": False,
        "specialist_identity_recorded": False,
        "specialist_review_date_recorded": False,
        "original_evaluator_metadata_preserved": True,
        "clinical_intervention_validation": False,
    }
    acknowledgment_path = returned / "SPECIALIST_CONFIRMATION_REPORTED.json"
    acknowledgment_path.write_text(
        json.dumps(acknowledgment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with zipfile.ZipFile(source) as z:
        payload = {name: z.read(name) for name in z.namelist() if not name.endswith("/")}
    for name in ("cluster_package_manifest.json", "validation.json"):
        payload.pop(name, None)
    main_text = payload["main.tex"].decode("utf-8")
    main_text = main_text.replace(
        "no expert review or ethics approval is asserted here.",
        "specialist confirmation of the supplementary AI-generated patch descriptions is reported by the study owner, as documented in the supplement. No independent blinded expert validation or ethics approval is asserted here.",
    )
    payload["main.tex"] = main_text.encode("utf-8")
    supplement = payload["cluster_supplement.tex"].decode("utf-8")
    supplement = supplement.replace(
        "These names have not been validated by a pathologist.",
        "The study owner subsequently reported specialist confirmation of the 68 AI-generated prototype descriptions. This is treated as AI-assisted review, not independent blinded concept validation. The original evaluator metadata identifies a non-pathologist AI reviewer with prior exposure to group names and is preserved alongside this later report. Specialist identity, review date and a separate specialist attestation were not supplied. Confirmation of individual descriptions does not establish a one-to-one clinical label for each cluster.",
    )
    supplement = supplement.replace(
        "Names are unvalidated.",
        "Clinical cluster-level validity is not independently established; AI-assisted specialist confirmation of prototype descriptions is reported by the study owner.",
    )
    payload["cluster_supplement.tex"] = supplement.encode("utf-8")
    for path in sorted(returned.iterdir()):
        if path.is_file():
            payload[f"annotations/{path.name}"] = path.read_bytes()
    payload["README.md"] = """# GCIA — manuscript package with AI-assisted review

Open main.tex in Overleaf and compile with pdfLaTeX twice. All figure and table files are included. The archive includes the cellular pilot, supplementary patch-cluster audits, and the returned 68-example annotations.

The study owner reports that a specialist confirmed the AI-generated descriptions. This is AI-assisted review, not independent blinded validation. Original EVALUATEUR.json metadata and the historical import report are retained unchanged in annotations/. SPECIALIST_CONFIRMATION_REPORTED.json records the subsequent user statement; it is not a specialist-signed attestation. Concept names are not automatically clinical cell types and cluster purity was not independently established.

Historical results and protocols in data/ describe their status when computed. The specialist confirmation does not change numerical results or validate intervention plausibility. Independent concept probes, selective cellular interventions and external validation remain outstanding.

This package is a finalized exploratory manuscript delivery, not a claim that the full clinical project is complete. Complete author names, affiliations and applicable disclosure/data-use statements before submission. Local LaTeX compilation has not been performed. Archive integrity, hashes, and figure/input dependencies have been checked.
""".encode()
    payload["BILAN_FINAL_PILOTE.md"] = """# Statut de cette livraison

Manuscrit exploratoire, figures, résultats et annotations rassemblés. L'utilisateur rapporte une confirmation des 68 descriptions par le spécialiste après leur production par IA. Cette confirmation est consignée séparément, sans modifier les fichiers originaux.

La revue est assistée par IA, non indépendante et non aveugle. Elle ne démontre ni la pureté de chaque cluster, ni le typage des cellules, ni la plausibilité des interventions, ni une validation externe. Les rapports historiques conservent leur statut à la date du calcul/import ; le supplément LaTeX explicite le retour ultérieur de l'utilisateur.

Les résultats numériques restent inchangés. Le fichier main.tex est prêt à être importé dans Overleaf ; la compilation PDF n'a pas été effectuée localement. Les informations d'auteur doivent être complétées avant soumission.
""".encode()
    checked = []
    for name, data in payload.items():
        if not name.endswith(".tex"):
            continue
        text = data.decode("utf-8")
        references = re.findall(r"\\(?:input|includegraphics)(?:\[[^]]*\])?\{([^}]+)\}", text)
        for reference in references:
            if reference not in payload:
                raise FileNotFoundError(f"{name}: {reference}")
            checked.append(reference)
    payload["FINAL_MANIFEST.json"] = json.dumps(
        {
            "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()},
            "checked_latex_dependencies": checked,
            "latex_compiled": False,
            "review_provenance": acknowledgment,
        },
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")
    destination = ROOT / "paper/GCIA_final_IA_specialiste.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in payload.items():
            z.writestr(name, data)
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        manifest = json.loads(z.read("FINAL_MANIFEST.json"))
        assert all(
            hashlib.sha256(z.read(name)).hexdigest() == sha
            for name, sha in manifest["sha256"].items()
        )
        assert z.read("annotations/EVALUATEUR.json") == (returned / "EVALUATEUR.json").read_bytes()
    print(f"Verified: {destination} ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
