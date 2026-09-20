---
noteId: "043917c0b4ca11f1ac794713d9e3a04f"
tags: []

---

# Lire le code GCIA

## État vérifié

Tu peux commencer la lecture. Les derniers contrôles couvrent **44 tests réussis**.
Au moment de cette vérification : **3/14 images MoNuSeg évaluées**, construction des
66 graphes BRACS encore en attente. Les résultats cellulaires finaux ne sont donc
pas encore disponibles. Le [suivi automatique](reports/segmentation_progress.html)
fait foi pour la progression ultérieure. Ne pas relancer le pipeline déjà actif.

## Ordre conseillé

| Étape | Fichier | À comprendre |
|---|---|---|
| 1. Données | [Manifeste BRACS](data/bracs_expanded_study_manifest.csv) | Une image par patient, diagnostic et partition. |
| 2. Réseau | [gnn.py](src/models/gnn.py) | GCN, GraphSAGE, GATv2 : convolution, agrégation et classification. |
| 3. Segmentation | [hovernet.py](src/segmentation/hovernet.py) | Inférence par tuiles et assemblage des noyaux candidats. |
| 4. Évaluation | [nuclei.py](src/segmentation/nuclei.py), puis [evaluate_nuclei.py](scripts/evaluate_nuclei.py) | Appariement aux annotations, Dice, F1 et qualité panoptique. |
| 5. Graphes | [cells.py](src/graphs/cells.py), puis [prepare_cell_cohort.py](scripts/prepare_cell_cohort.py) | Un nœud par instance, morphologie, voisinage spatial et manifeste. |
| 6. Entraînement | [train_cell_gnns.py](scripts/train_cell_gnns.py) | Normalisation sur l'apprentissage, trois architectures, trois graines et test verrouillé. |
| 7. Explications | [run_bracs_three_xai.py](scripts/run_bracs_three_xai.py) | GNNExplainer, adaptation GCExplainer et premier audit GCIA sur proxies visuels. |
| 8. Concepts annotés | [run_annotated_gcia.py](scripts/run_annotated_gcia.py) | Probes BCSS, substitutions de patches et modifications du graphe avec contrôles. |

Dans chaque script, lire d'abord `main()`, puis les fonctions qu'il appelle.
`src/` contient les composants réutilisables ; `scripts/` assemble les expériences ;
`tests/` vérifie le logiciel ; `reports/` conserve les résultats réellement produits.

## Ne pas confondre

- Le **90,9 % BRACS** vient de [run_bracs_binary.py](scripts/run_bracs_binary.py),
  un GCN de **patches**, différent des futurs GNN cellulaires.
- Les audits XAI existants portent sur ce modèle de patches, pas encore sur les nouveaux graphes cellulaires.
- `superpixels.py` est un ancien extracteur de patches, pas un segmentateur de noyaux.
- Un test logiciel réussi ou un score MoNuSeg ne valide pas automatiquement la segmentation BRACS.
- Les interventions GCIA restent exploratoires ; la plausibilité histologique n'est pas validée.

## Résultats à lire avec le code

- [Classification BRACS](reports/bracs_expanded/RESULTS.md)
- [Trois audits XAI](reports/bracs_three_xai/RESULTS.md)
- [Audit avec annotations BCSS](reports/annotated_gcia/RESULTS.md)
- [Guide d'exécution des graphes cellulaires](reports/cell_graph_quickstart.md)

Dans VS Code : **Ctrl+Shift+V** pour afficher ce guide formaté.
