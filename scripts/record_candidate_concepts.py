"""Record user-supplied AI concept names without claiming expert validation."""

import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "reports/cluster_review"
SOURCE = Path(
    r"C:\Users\Mahugnon\.codex\attachments\bbe51d67-d36c-4340-bbdc-c1e8e0d15276\pasted-text.txt"
)
LABELS = [
    (
        "Periglandular fibrocollagenous stroma",
        "Stroma fibrocollagénique périglandulaire",
        "Hypothèse : tissu fibreux situé autour de structures glandulaires.",
        "Distinguer le stroma lui-même de la frontière épithélium–stroma (cluster 4).",
    ),
    (
        "Dense collagenous / desmoplastic stroma",
        "Stroma collagénique dense / desmoplastique",
        "Hypothèse : prédominance de matrice fibreuse dense.",
        "Le qualificatif desmoplastique reste à vérifier ; ne pas déduire une réaction tumorale du cluster seul.",
    ),
    (
        "Fibroblasts / spindle stromal cells",
        "Fibroblastes / cellules stromales fusiformes",
        "Hypothèse : présence de cellules stromales à morphologie allongée.",
        "Une forme fusiforme ne suffit pas à établir le type cellulaire fibroblastique.",
    ),
    (
        "Invasive epithelial / tumor cells",
        "Cellules épithéliales invasives / tumorales",
        "Hypothèse : composante épithéliale cellulaire.",
        "Le caractère tumoral et surtout invasif nécessite une confirmation indépendante ; aucun diagnostic n'est établi ici.",
    ),
    (
        "Tumor–stroma interface / epithelial–stromal boundary",
        "Interface tumeur–stroma / frontière épithélium–stroma",
        "Hypothèse : juxtaposition de compartiments épithélial et stromal.",
        "Vérifier la frontière et le caractère tumoral ; chevauchement possible avec les clusters 0–2.",
    ),
]


def main():
    source_bytes = SOURCE.read_bytes()
    source_copy = ROOT / "reports/cluster_concepts_source.txt"
    if source_copy.exists() and source_copy.read_bytes() != source_bytes:
        raise RuntimeError("Existing source differs; preserve prior provenance.")
    source_copy.write_bytes(source_bytes)
    concepts = []
    for cluster, (original, french, definition, uncertainty) in enumerate(LABELS):
        assert (REVIEW / f"cluster_{cluster}" / "PLANCHE.pdf").exists()
        concepts.append(
            {
                "cluster_id": cluster,
                "source_label": original,
                "label_fr": french,
                "working_definition": definition,
                "uncertainty": uncertainty,
                "annotation_origin": "user_supplied_chatgpt_interpretation",
                "status": "candidate_unvalidated",
                "expert_validated": False,
                "purity": None,
                "expert_confidence": None,
                "reported_ai_confidence": "approximately 3/4" if cluster == 4 else None,
            }
        )
        (REVIEW / f"cluster_{cluster}" / "CONCEPT_PROVISOIRE.md").write_text(
            f"# Cluster {cluster} — concept candidat\n\n**{french}**\n\n"
            f"Nom fourni : {original}.\n\n{definition}\n\n{uncertainty}\n\n"
            "Provenance : interprétation ChatGPT transmise par l'utilisateur. "
            "Ce document n'est pas une annotation d'anatomopathologiste. "
            "Aucune pureté, validation clinique ou utilisation fonctionnelle n'est établie.\n\n"
            "Le groupe provient du GCN de patches. Ne pas transférer automatiquement "
            "ce nom aux cellules des graphes cellulaires. Les annotations individuelles "
            "restent à effectuer. Pour une revue indépendante, examiner d'abord les "
            "images sans consulter ces propositions.\n",
            encoding="utf-8",
        )
    registry = {
        "version": 1,
        "scope": "existing_patch_GCN_KMeans5_only",
        "source": {
            "file": "reports/cluster_concepts_source.txt",
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "shared_url": "https://chatgpt.com/share/6ab02f21-71d0-83ea-a327-3071f80de64a",
            "shared_url_accessed_successfully": False,
        },
        "concepts": concepts,
    }
    (REVIEW / "CONCEPTS_PROVISOIRES.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (REVIEW / "CONCEPTS_PROVISOIRES.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(concepts[0]))
        writer.writeheader()
        writer.writerows(concepts)
    (REVIEW / "STATUT_CONCEPTS.md").write_text(
        "# Cinq concepts candidats enregistrés\n\n"
        "Les noms proposés sont disponibles dans CONCEPTS_PROVISOIRES.csv et .json "
        "et dans CONCEPT_PROVISOIRE.md de chacun des cinq dossiers. "
        "Les noms originaux sont conservés ; les définitions de travail ne sont pas des observations vérifiées.\n\n"
        "Les fiches NOMMER_LES_GROUPES.csv et ANNOTATIONS_EXEMPLES.csv restent intactes : "
        "aucun avis médical, score de cohérence ou annotation individuelle n'a été inventé. "
        "La confiance 3/4 du cluster 4 est celle du texte fourni, pas celle d'un spécialiste.\n\n"
        "## Suite expérimentale\n\n"
        "1. Faire annoter les exemples, y compris les contre-exemples, indépendamment des noms proposés.\n"
        "2. Figer les définitions et vérifier leur cohérence sur des exemples distincts ; autoriser fusion ou rejet.\n"
        "3. Pour les graphes cellulaires, obtenir des annotations cellule/compartiment et vérifier la correspondance avec ces concepts de patches.\n"
        "4. Évaluer les probes avec des annotations indépendantes : prédire l'identifiant K-means ne valide pas un concept histologique.\n"
        "5. Tester des interventions ciblées avec contrôles appariés, changements des autres concepts et plausibilité.\n\n"
        "Le nommage ne produit pas de nouveau résultat GCIA. Les expériences antérieures "
        "sur la morphologie ne démontrent pas l'utilisation de ces cinq concepts. "
        "Les exemples numériques du texte fourni ne sont pas des résultats de ce projet.\n\n"
        "L'archive cluster_review_concepts_provisoires.zip inclut ces propositions et peut "
        "influencer une revue. Pour une première lecture sans noms suggérés, utiliser "
        "l'archive originale cluster_review_specialiste.zip, conservée sans modification.\n",
        encoding="utf-8",
    )
    archive = ROOT / "reports/cluster_review_concepts_provisoires.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as zipped:
        for path in sorted(REVIEW.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(REVIEW.parent))
    with ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert sum(name.endswith("/CONCEPT_PROVISOIRE.md") for name in zipped.namelist()) == 5
    loaded = json.loads((REVIEW / "CONCEPTS_PROVISOIRES.json").read_text(encoding="utf-8"))
    assert len(loaded["concepts"]) == 5
    assert all(not item["expert_validated"] for item in loaded["concepts"])
    print(f"Five provisional concepts recorded; archive verified: {archive}")


if __name__ == "__main__":
    main()
