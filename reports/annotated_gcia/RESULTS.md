---
noteId: "149d3c20b4c211f1ac794713d9e3a04f"
tags: []

---

# GCIA : audit avec annotations tissulaires BCSS

Le GCN diagnostique BRACS est gelé. Les annotations BCSS de tumeur, stroma et
infiltrat lymphocytaire remplacent les propriétés de couleur du pilote précédent.
Huit patients BCSS déjà explorés sont audités. Aucun téléchargement ni réentraînement
du classifieur. Chaque probe est ajustée sur sept patients et évaluée sur le huitième.
Les patches ont le même poids au sein d'un patient, les patients le même poids au fitting.

**La performance diagnostique du GCN sur BCSS n'est pas établie.** L'échelle des
patches et le domaine diffèrent de BRACS ; cet audit ne constitue pas une validation externe.

## Décodage des concepts à la deuxième couche

MSE moyenne par patient ; plus faible est meilleur. La référence constante est calculée
uniquement sur les sept patients d'apprentissage. Toutes les couches sont dans probes.csv.

| Concept annoté | MSE probe | MSE constante |
|---|---:|---:|
| tumor | 0.1015 | 0.1468 |
| stroma | 0.1012 | 0.0908 |
| lymphocytic_infiltrate | 0.0585 | 0.1051 |

## Interventions et contrôles

Images : jusqu'à cinq patches tumoraux par patient remplacés par des patches stromaux
du même patient ; contrôles par autres patches tumoraux. Donneurs choisis selon leur
couleur moyenne, sans consulter la sortie du modèle. Les pixels donneurs sont réencodés
et l'équivalence avec la substitution d'embeddings est vérifiée.
La tumeur diminue et le stroma augmente simultanément : ce n'est pas une intervention
isolant la tumeur. Les deux autres changements annotés et les changements de probes sont conservés.

Graphes : suppression d'au plus cinq connexions tumeur-stroma ; 20 contrôles aléatoires
de même nombre et mêmes longueurs géométriques. Les degrés ne sont pas appariés.
Les attributs et coordonnées restent identiques. La connectivité est enregistrée.

Variation absolue moyenne de P(IC), en points ; moyenne des interventions par patient
puis moyenne des patients admissibles. Ces valeurs mesurent une sensibilité, pas une validité clinique.

| Intervention | Patients admissibles | Ciblée | Contrôle |
|---|---:|---:|---:|
| Patch tumeur vers stroma | 6 | 0.4548 | 0.3953 |
| Connexions tumeur-stroma | 5 | 0.0536 | 0.1236 |

Suppressions augmentant le nombre de composantes : 0/105.


## Corrections réalisées et limites restantes

- Concepts issus d'annotations indépendantes du diagnostic BRACS, au lieu de proxies de couleur.
- Probes à trois niveaux avec séparation stricte des patients pour leur apprentissage.
- Substitutions de patches réels avec réencodage et contrôles ; exemples PNG destinataire/donneur.
- Interventions de topologie avec contrôles et détection de déconnexion.
- Poids du classifieur vérifiés inchangés ; protocole et résultats bruts conservés.

Restent nécessaires pour l'abstract complet : segmentation cellulaire évaluée avec annotations
nucléaires, graphes cellulaires et nouveaux classifieurs correspondants, concepts spatiaux
validés, interventions sélectives et revue de plausibilité histologique, validation BACH
et comparaison multi-architectures. Les expériences actuelles utilisent des graphes de patches.
Les interventions présentes ne passent pas automatiquement un critère de validité GCIA.
Aucune conclusion « concept pathologique encodé et utilisé » n'est établie par ces seuls résultats.

Fichiers : `protocol.json`, `summary.json`, `coverage.csv`, `probes.csv`,
`patch_interventions.csv`, `graph_interventions.csv`, poids des probes `.npz` et paires `.png`.
