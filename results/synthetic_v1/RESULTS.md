---
noteId: "4e12af10b5c811f19aa73b13fcac3a6b"
tags: []

---

# Benchmark synthétique v1

Modèles réellement entraînés, sélection sur validation. Aucune image histologique utilisée.

Plan factoriel : used influence le label ; unused est une caractéristique sans rôle dans le label ; relational modifie les arêtes à composition et degrés constants ; irrelevant n'est pas fourni au modèle. Label = 2*used + relational. Les réplicats partagent les états factoriels : ils ne représentent pas des patients indépendants.

| Modèle | Graine | Accuracy test |
|---|---:|---:|
| DeepSets | 11 | 0.500 |
| DeepSets | 23 | 0.500 |
| DeepSets | 37 | 0.500 |
| GCN | 11 | 1.000 |
| GCN | 23 | 1.000 |
| GCN | 37 | 1.000 |
| GraphSAGE | 11 | 1.000 |
| GraphSAGE | 23 | 1.000 |
| GraphSAGE | 37 | 1.000 |
| GATv2 | 11 | 1.000 |
| GATv2 | 23 | 1.000 |
| GATv2 | 37 | 1.000 |

Les tables ground_truth_recovery.csv et recovery_metrics.csv évaluent un dépistage descriptif de sensibilité, pas la vérité causale du modèle entraîné. Les probes sont ajustées sur apprentissage ; TCAV est une sensibilité directionnelle du logit ; l'effacement orthogonal CAV n'est pas LEACE. Les effets mesurent la variation totale des probabilités.

**Passage aux grandes expériences histologiques non autorisé par ce benchmark v1.** Restent les contrôles appariés non triviaux, l'équivalence, GNNExplainer sur ce benchmark, LEACE et les contrôles de randomisation. Les résultats, même favorables, ne clôturent pas ces exigences.
