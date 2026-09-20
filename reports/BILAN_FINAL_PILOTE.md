# Bilan du pilote GCIA

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
- Tests : `.\.venv\Scripts\python.exe -m pytest -q` depuis GCIA.
- Les scripts d'audit refusent d'écraser leurs dossiers existants. Pour une réplication, travailler dans une copie du projet en conservant les résultats originaux ; aucun téléchargement requis.
- Archive : paper/GCIA_pilot_with_cluster_audits.zip. Ouvrir main.tex dans Overleaf ; auteur et affiliation à compléter. Compilation LaTeX non vérifiée localement.

Les chiffres hypothétiques du document joint ne sont pas des résultats du projet. La finalisation d'un pilote ne signifie pas validation de l'ensemble de l'hypothèse clinique.
