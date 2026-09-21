---
noteId: "4df22ec0b5c811f19aa73b13fcac3a6b"
tags: []

---

# GCIA — protocole de la nouvelle étude

Le pilote est conservé comme provenance, sans optimisation supplémentaire ni réouverture de ses 22 patients comme test confirmatoire. Nouvelle branche : `experiments/full-study`. Configuration du premier benchmark : `experiments/full_study/config.json`.

## A. Vérification logicielle

`python scripts/run_all_checks.py` lance Ruff, format-check et pytest, produit JUnit et empreintes de tous les fichiers source. Un arbre de travail modifié est déclaré comme tel : la preuve concerne son contenu exact, pas artificiellement un HEAD propre. Le workflow CI ne constitue une preuve distante qu'après exécution effective.

## B. Benchmark à mécanismes contrôlés

Graphes à 12 nœuds, deux types en nombres égaux, cycles connectés de degré deux. Plan factoriel équilibré de quatre variables binaires : caractéristique pertinente, caractéristique indépendante, organisation homophile/alternée et variable non observée. Label à quatre classes : `2*used + relational`.

Les splits sont indépendamment générés mais partagent les états factoriels. C'est un test de mécanisme, pas une preuve de généralisation à une population de patients. La règle génératrice est connue ; la dépendance du classifieur appris doit être mesurée et peut différer. DeepSets doit être insensible aux arêtes par construction. GCN, GraphSAGE et GATv2 ont trois couches et le même pooling ; sélection sur validation uniquement, trois graines, aucun ajustement après lecture du test.

Sauvegarder poids, données, probes couche par couche, sensibilité directionnelle CAV, effacement orthogonal (ne pas appeler cela LEACE), effets des interventions et échecs. La v1 n'a pas encore tous les contrôles ni toutes les baselines. La marge descriptive de 0,01 ne constitue pas un test d'équivalence. Aucune transition vers l'étude histologique confirmatoire tant que ces critères restent incomplets.

## C. Concepts indépendants

Les 68 descriptions assistées par IA et la confirmation du spécialiste rapportée par l'utilisateur restent du matériel de découverte. Figer d'abord 3–5 définitions observables, unité, critères d'indétermination et modalités de mesure avant/après intervention. Obtenir un lot aléatoire distinct, sans noms de groupes ni suggestions. Ne pas générer de grille prétendument validée à partir des anciens noms automatiquement. Un évaluateur est disponible selon l'utilisateur ; ne pas calculer d'accord inter-évaluateur avec un seul spécialiste.

## D. Nouveau BRACS, échelle et typage

Les 30 patients candidats précédemment recensés ne sont pas un test verrouillé. Aucune nouvelle image de test n'est ouverte ici. Conditions de verrouillage : exclusions historiques revérifiées, tâche définie sur développement (ADH/DCIS/IC à examiner), inventaire complet, résolution physique sourcée par image, contrôle segmentation/typage, sélection multirégion fixée, modèles/interventions/statistiques gelés. Une valeur MPP par défaut n'est pas une mesure. Tout graphe dépourvu de MPP vérifié ou de typage validé reste inéligible aux conclusions multicohortes sur types et distances physiques.

Les règles générées ne doivent jamais inventer une résolution, un type cellulaire ou une attestation médicale. Le pipeline complet MPP/typage est encore à développer ; BACH demeure non acquis et non évalué.

## Critère de progression

Le prochain manuscrit principal attendra les contrôles de sélectivité/support, l'équivalence préspécifiée, les baselines communes, les annotations indépendantes puis une vraie réplication externe. Aucun résultat positif n'est garanti ni requis pour conserver une expérience. Les résultats négatifs et les échecs de récupération sont conservés.
