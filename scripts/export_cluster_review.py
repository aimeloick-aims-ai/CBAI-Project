"""Export blinded train-only cluster examples for human semantic review, without reclustering."""

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/cluster_review"


def write_csv(path, rows, fields=None):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUT.exists():
        raise FileExistsError(
            "Preserve reviewer annotations; choose a versioned output before rerunning"
        )
    source = ROOT / "reports/bracs_three_xai"
    manifest_path = ROOT / "data/bracs_expanded_study_manifest.csv"
    manifest = list(csv.DictReader(manifest_path.open()))
    saved = np.load(source / "representations.npz", allow_pickle=False)
    clustering = np.load(source / "gcexplainer_clusters.npz", allow_pickle=False)
    representations, centers, assignments = (
        saved["layer2"],
        clustering["centers"],
        clustering["assignments"],
    )
    expected = json.loads((source / "started.json").read_text())["manifest_sha256"]
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected:
        raise ValueError("Manifest differs from the clustering experiment")
    if list(saved["patients"]) != [r["patient_id"] for r in manifest] or len(centers) != 5:
        raise ValueError("Patient alignment or cluster count mismatch")
    train = [i for i, r in enumerate(manifest) if r["proposed_split"] == "train"]
    distance = np.linalg.norm(representations[:, :, None, :] - centers[None, None, :, :], axis=-1)
    if not np.array_equal(distance.argmin(-1), assignments):
        raise ValueError("Stored assignments disagree with centers")
    OUT.mkdir(parents=True)
    provenance, group_rows, coverage = [], [], []
    rng = np.random.default_rng(2026)
    for cluster in range(5):
        folder = OUT / f"cluster_{cluster}"
        folder.mkdir()
        candidates = sorted(
            [
                (float(distance[i, node, cluster]), i, node)
                for i in train
                for node in range(64)
                if assignments[i, node] == cluster
            ]
        )
        # Nearest member from each patient, then up to 12 different patients.
        distinct = {}
        for candidate in candidates:
            distinct.setdefault(manifest[candidate[1]]["patient_id"], candidate)
        core = sorted(distinct.values())[:12]
        chosen = {(i, node) for _, i, node in core}
        used = {manifest[i]["patient_id"] for _, i, _ in core}
        additional = []
        for index in rng.permutation(len(candidates)):
            candidate = candidates[index]
            _, i, node = candidate
            if (i, node) not in chosen and manifest[i]["patient_id"] not in used:
                additional.append(candidate)
                used.add(manifest[i]["patient_id"])
                if len(additional) == 4:
                    break
        selected = [("central", c) for c in core] + [("complementaire", c) for c in additional]
        if not selected:
            raise ValueError("No training examples for a cluster")
        review, tiles = [], []
        for number, (selection, (dist, i, node)) in enumerate(selected, start=1):
            row = manifest[i]
            path = ROOT / row["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Image checksum mismatch")
            identifier = f"C{cluster}_E{number:02d}"
            yy, xx = divmod(node, 8)
            with Image.open(path) as image:
                native = image.convert("RGB")
                width, height = native.size
                resized = native.resize((768, 768), Image.Resampling.BILINEAR)
                tile = resized.crop((xx * 96, yy * 96, (xx + 1) * 96, (yy + 1) * 96))
                tile.save(folder / f"{identifier}_patch.png")
                box = (
                    int(np.floor(xx * width / 8)),
                    int(np.floor(yy * height / 8)),
                    int(np.ceil((xx + 1) * width / 8)),
                    int(np.ceil((yy + 1) * height / 8)),
                )
                native.crop(box).save(folder / f"{identifier}_original.png")
                context = resized.copy()
                draw = ImageDraw.Draw(context)
                draw.rectangle(
                    (xx * 96, yy * 96, (xx + 1) * 96 - 1, (yy + 1) * 96 - 1),
                    outline="yellow",
                    width=3,
                )
                context.save(folder / f"{identifier}_contexte.png")
            tiles.append((identifier, selection, np.array(tile)))
            review.append(
                {
                    "exemple": identifier,
                    "selection": selection,
                    "description_observee": "",
                    "qualite_image": "",
                    "compatible_avec_concept_du_groupe": "",
                    "confiance_1_a_5": "",
                    "commentaire": "",
                }
            )
            provenance.append(
                {
                    "cluster": cluster,
                    "example": identifier,
                    "selection": selection,
                    "patient_id": row["patient_id"],
                    "source_image": row["path"],
                    "source_sha256": row["sha256"],
                    "split": row["proposed_split"],
                    "node_index": node,
                    "distance_to_centroid": dist,
                    "native_xyxy": box,
                }
            )
        fig, axes = plt.subplots(
            int(np.ceil(len(tiles) / 4)),
            4,
            figsize=(10, 2.8 * int(np.ceil(len(tiles) / 4))),
            squeeze=False,
        )
        for ax in axes.flat:
            ax.axis("off")
        for ax, (identifier, selection, tile) in zip(axes.flat, tiles):
            ax.imshow(tile, interpolation="nearest")
            ax.set_title(f"{identifier} / {selection}", fontsize=9)
        fig.suptitle(f"Cluster {cluster} — groupe candidat, sans nom clinique", fontsize=12)
        fig.tight_layout()
        fig.savefig(folder / "PLANCHE.png", dpi=160)
        fig.savefig(folder / "PLANCHE.pdf", bbox_inches="tight")
        plt.close(fig)
        write_csv(folder / "ANNOTATIONS_EXEMPLES.csv", review)
        (folder / "LIRE.md").write_text(
            f"# Cluster {cluster}\n\n{len(core)} exemples centraux et {len(additional)} exemples complémentaires ; un patient différent par exemple.\nCommencer par PLANCHE.pdf, puis consulter les fichiers *_original.png et *_contexte.png.\nLe fichier *_patch.png est exactement la vue 96×96 utilisée par le modèle, après redimensionnement de la RoI à 768×768.\nL'image *_original.png conserve les pixels natifs de la zone correspondante. Aucun grossissement physique n'est attribué.\n\nDécrire les traits communs et les contre-exemples dans ANNOTATIONS_EXEMPLES.csv.\nVous pouvez conclure : groupe hétérogène, artefact, information insuffisante ou absence de concept médical identifiable.\n",
            encoding="utf-8",
        )
        group_rows.append(
            {
                "cluster": cluster,
                "nom_concept_propose": "",
                "definition_visuelle": "",
                "coherence_1_a_5": "",
                "confiance_1_a_5": "",
                "contre_exemples": "",
                "groupe_heterogene_oui_non": "",
                "fusion_ou_scission_a_envisager": "",
                "commentaires": "",
                "evaluateur": "",
                "date": "",
            }
        )
        coverage.append(
            {
                "cluster": cluster,
                "training_patches": len(candidates),
                "training_patients": len(distinct),
                "review_examples": len(selected),
            }
        )
    write_csv(OUT / "NOMMER_LES_GROUPES.csv", group_rows)
    (OUT / "LIRE_EN_PREMIER.md").write_text(
        """# Revue des cinq clusters candidats

Objectif : déterminer si chaque groupe correspond à une notion histologique cohérente,
puis proposer un nom et une définition observable. Il est acceptable de ne pas nommer
un groupe ou de proposer une fusion/scission : cinq clusters ne garantissent pas cinq concepts.

1. Ouvrir PLANCHE.pdf dans chacun des cinq dossiers.
2. Examiner les images originales et leur contexte, notamment les contre-exemples.
3. Remplir ANNOTATIONS_EXEMPLES.csv par image puis NOMMER_LES_GROUPES.csv pour la synthèse.
4. Préciser les ambiguïtés, la qualité et la confiance (1 faible, 5 forte).

Les dossiers proviennent du K-means existant à cinq groupes, sur les représentations
du GCN de **patches**. Ce ne sont pas les clusters des nouveaux GNN cellulaires.
Aucun nouveau clustering ni sélection fondée sur le diagnostic n'a été effectué.
Seuls les patients d'apprentissage sont utilisés. Les diagnostics et prédictions ne
sont pas inclus dans ce dossier afin de limiter leur influence sur l'interprétation.

Par groupe : jusqu'à 12 prototypes proches du centroïde, avec un patient distinct
par prototype, puis jusqu'à 4 exemples supplémentaires aléatoires d'autres patients.
Ces exemples sont sélectionnés pour la revue ; ils ne permettent pas d'estimer sans
biais la pureté globale du cluster. Les PNG patch sont les vues exactes du modèle ;
les PNG original conservent les pixels natifs correspondants ; le contexte indique
la zone en jaune. Les mesures physiques ne sont pas certifiées.

Après cette première revue, figer les définitions et les vérifier sur un ensemble
distinct annoté sans ajuster les clusters. Ne pas confondre nommage, pureté mesurée
et preuve d'utilisation : GCIA devra ensuite tester les interventions associées aux
concepts validés, leur sélectivité et leur plausibilité.

Les CSV s'ouvrent dans un tableur. Conserver les identifiants d'exemples et les IDs
des groupes. Pour deux spécialistes, travailler sur deux copies séparées avant consensus.
""",
        encoding="utf-8",
    )
    sources = [manifest_path, source / "representations.npz", source / "gcexplainer_clusters.npz"]
    (ROOT / "reports/cluster_review_provenance.json").write_text(
        json.dumps(
            {
                "sources": {
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sources
                },
                "coverage": coverage,
                "selection": provenance,
                "scope": "training-only semantic discovery, not independent validation",
            },
            indent=2,
        )
    )
    archive = ROOT / "reports/cluster_review_specialiste.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(
        json.dumps(
            {"coverage": coverage, "total_examples": len(provenance), "archive": str(archive)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
