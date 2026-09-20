---
noteId: "2c609ec0b3b811f180a4c967c0077061"
tags: []

---

# Extension BCSS legere — 19 septembre 2026

Statut : experience exploratoire executee ; contribution GCIA non validee.

## Donnees et protocole

Quatre nouveaux patients BCSS : 12 855 946 octets de RGB et masques (12,86 Mo),
plus quelques metadonnees. Six patients au total, selectionnes par les six premieres
lignes des metadonnees deja disponibles, sans choix selon les scores. Plafond d'acquisition
fixe a 32 Mio. Aucun poids de modele telecharge, aucun nouvel entrainement.

Les 15 checkpoints BRACS existants sont geles : trois architectures, cinq graines.
Trois concepts issus des masques BCSS : tumeur, stroma et infiltrat lymphocytaire.
Sondes sur entree RGB, couche 1 et couche 2. Evaluation en laissant un patient a part,
ajustement sur les cinq autres ; normalisation apprise uniquement sur ces cinq patients.
Ridge alpha=1 fixe. Labels melanges a l'interieur de chaque patient d'ajustement.

810 evaluations de sondes et 1080 perturbations d'attributs. Ces calculs ne constituent
pas autant de sujets independants : l'unite independante reste le patient (n=6).
Protocole dans configs/extended_xai_protocol.json, empreintes dans summary.json.

## Decodage : erreur quadratique moyenne

Moyennes descriptives des six patients, trois architectures et cinq graines.
Plus faible est meilleur ; les graines et architectures ne sont pas des patients supplementaires.

| Concept | Entree RGB | Couche 1 | Couche 2 | Reference constante |
|---|---:|---:|---:|---:|
| Tumeur | 0,1190 | 0,1237 | 0,1203 | 0,1342 |
| Stroma | 0,0980 | 0,1102 | 0,0987 | 0,0817 |
| Infiltrat lymphocytaire | 0,1255 | 0,0977 | 0,1025 | 0,1079 |

Pour la tumeur, les representations apprises ne font pas mieux que les attributs RGB
en moyenne. Les sondes stromales font moins bien que la constante. Les couches apprises
ont une erreur moyenne plus faible pour l'infiltrat lymphocytaire que l'entree RGB.
Ce sont des observations exploratoires, sans test de significativite ni revendication
de generalisation. Les controles melanges detailles sont disponibles dans probes.csv.

## Comparaison d'attributions

Remplacement de huit noeuds par la moyenne des attributs du graphe. Classe cible fixee
a la prediction initiale. Le classement par gradients utilise la valeur absolue du gradient
multiplie par l'ecart a cette moyenne ; il peut donc selectionner des contributions negatives.
Dix selections aleatoires de huit noeuds par patient/modele servent de controles.

| Selection | Variation moyenne de probabilite (points de pourcentage) |
|---|---:|
| Masque tumoral | -0,0301 |
| Gradient fois ecart a la reference | +0,3792 |
| Aleatoire | +0,0589 |

Les moyennes signees peuvent cacher des effets de signes opposes. Elles ne prouvent
ni absence d'utilisation ni superiorite d'une methode. Les controles ne sont apparies
qu'en nombre de noeuds, pas en amplitude ou structure spatiale.

## Ce qui reste bloque

- Graphes cellulaires : segmentation de noyaux et validation NuCLS non executees.
- Interventions conceptuelles plausibles : aucune annotation independante avant/apres
  n'etablit le changement cible et la preservation des autres concepts. Un telechargement
  seul ne remplace pas cette verification histologique.
- Comparateurs specialises TCAV, GNNExplainer et GCExplainer non executes ; la comparaison
  par gradients n'est pas une implementation de ces methodes.
- BACH : pas de validation externe. Les sept classes BRACS ne correspondent pas directement
  aux quatre classes BACH ; le protocole de correspondance doit etre defini avant evaluation.
- Incertitude de population : six patients de convenance restent insuffisants pour soutenir
  les revendications generales de l'abstract. Les scores BCSS ne sont pas des diagnostics.

Ces limites sont scientifiques ; elles ne sont pas resolues en choisissant davantage
d'images jusqu'a ce que les chiffres augmentent.

## Reproduction

Depuis GCIA : `.\.venv\Scripts\python.exe scripts\run_extended_xai.py`.
Ce script reutilise les donnees et poids locaux, sans telechargement.
Il ecrit probes.csv, deletion_baselines.csv et summary.json dans ce dossier.
