---
noteId: "adcd5e00b5c811f19aa73b13fcac3a6b"
tags: []

---

# Avancement — première étape full study

## Exécuté

- Branche dédiée, protocole et configuration du benchmark.
- 12 entraînements : DeepSets, GCN, GraphSAGE, GATv2 × 3 graines ; trois couches, mêmes features et pooling.
- Accuracy sur la tâche artificielle à quatre classes : DeepSets 0,50 ; GNN 1,00 pour chaque graine. Le résultat est attendu sur un mécanisme simple, pas transférable à l'histologie.
- Le concept indépendant du label est parfaitement décodable (AUROC 1,00) dans la dernière couche de chaque modèle. Son changement induit une variation totale maximale des probabilités inférieure à 0,00184 dans les cas testés. Ce constat est restreint au benchmark fini ; il ne constitue pas une procédure générale d'équivalence.
- Probes à toutes les couches, sensibilité directionnelle CAV et effacement orthogonal CAV calculés. Ce dernier n'est pas LEACE. Aucun score GNNExplainer nouveau n'est encore calculé sur ce benchmark.
- DeepSets est exactement invariant à l'intervention sur les arêtes. La concordance avec la dépendance du générateur est moins bonne pour DeepSets par conception, pas à cause d'une erreur GCIA.

## Conditions encore ouvertes

Le fichier `synthetic_v1/summary.json` conserve `progression_gate_passed: false`. Manquent notamment les témoins appariés non triviaux, l'équivalence, LEACE, GNNExplainer sur ce benchmark et les randomisations complètes. Les catégories encoded/used restent des dépistages descriptifs, pas des conclusions confirmatoires.

Concepts humains : les annotations antérieures restent du matériel de découverte. Pas de nouvelle validation indépendante réalisée ; définir l'ontologie avant de constituer son lot d'évaluation.

BRACS : l'inventaire antérieur de patients candidats est conservé. Pas de verrouillage prématuré ni de MPP/type cellulaire inventé. Le travail technique d'acquisition multirégion, de vérification de résolution et de typage reste à réaliser après les conditions du benchmark.

Pas de nouveau manuscrit principal ni de figure clinique fabriquée. Les archives du pilote restent inchangées.
