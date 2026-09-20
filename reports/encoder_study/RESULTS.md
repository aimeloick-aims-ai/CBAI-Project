# Comparaison avec encodeur histopathologique gele

Experience exploratoire executee une seule fois selon configs/encoder_protocol.json.
Six patients de developpement deja explores ; deux nouveaux patients reserves.
Aucun reglage choisi sur leurs scores. Chaque condition est rapportee.

## Audit visuel

La planche alignment.png a ete examinee : pas de decalage grossier apparent a cette
echelle. Les contours regionaux sont compatibles visuellement avec l'image.
Cette inspection technique ne valide ni chaque annotation ni les interventions
par un pathologiste. Noir = autres classes ou zones ignorees, pas forcement fond.

## Resultats

Moyennes sur patients et trois graines ; plus faible est meilleur.
La fraction regionale est la moyenne des patches valides, pas celle de toute la lame.

| Ensemble | Caracteristiques | Modele | MSE patches | MAE region | MSE constante | MAE constante |
|---|---|---|---:|---:|---:|---:|
| development | pcam | gcn | 0.1347 | 0.1853 | 0.1524 | 0.1226 |
| development | pcam | mlp | 0.0810 | 0.0913 | 0.1524 | 0.1226 |
| development | rgb | gcn | 0.1265 | 0.1607 | 0.1524 | 0.1226 |
| development | rgb | mlp | 0.1267 | 0.1944 | 0.1524 | 0.1226 |
| held_out | pcam | gcn | 0.0694 | 0.1160 | 0.1338 | 0.0993 |
| held_out | pcam | mlp | 0.0697 | 0.0608 | 0.1338 | 0.0993 |
| held_out | rgb | gcn | 0.1338 | 0.2130 | 0.1338 | 0.0993 |
| held_out | rgb | mlp | 0.2035 | 0.3340 | 0.1338 | 0.0993 |

## Nouveaux patients separement

| Patient | Caracteristiques | Modele | MSE moyenne |
|---|---|---|---:|
| TCGA-A2-A0D0 | pcam | gcn | 0.0383 |
| TCGA-A2-A0D0 | pcam | mlp | 0.0790 |
| TCGA-A2-A0D0 | rgb | gcn | 0.0382 |
| TCGA-A2-A0D0 | rgb | mlp | 0.1139 |
| TCGA-A2-A0D2 | pcam | gcn | 0.1004 |
| TCGA-A2-A0D2 | pcam | mlp | 0.0604 |
| TCGA-A2-A0D2 | rgb | gcn | 0.2294 |
| TCGA-A2-A0D2 | rgb | mlp | 0.2930 |

## Portee et sources

PCam preentraine MobileNetV3-Small : poids officiels TIACentre, empreinte verifiee,
3 833 181 octets. RGB /255, patches 96x96 au pas physique demande de 1 micrometre.
Le backbone reste gele. MLP et GCN utilisent les memes representations.
Reference constante : moyenne par patient des fractions tumorales du train.
La normalisation est apprise uniquement sur les patients du train.

Poids et licence CC0 documentes par [TIAToolbox](https://tia-toolbox.readthedocs.io/en/stable/pretrained.html).
Lymph nodes PCam et tissu mammaire BCSS sont des domaines distincts.
Les patches de 96 pixels et la reference constante ponderee par patient different
de certains essais precedents : la comparaison juste est celle de ce tableau.

Deux nouveaux patients ne suffisent pas a une validation populationnelle.
C'est une prediction directe de concept, pas une preuve de son utilisation diagnostique.
Aucun test de significativite, choix du meilleur seed ou validation GCIA ne sont revendiques.

Reproduction : scripts/run_encoder_study.py ; refus decrasement des resultats existants.
Synthese sans reentrainement : scripts/report_encoder_study.py.
Les checkpoints, normalisations, embeddings, protocole et empreintes sont conserves ici.
