---
noteId: "c156f820b3b911f180a4c967c0077061"
tags: []

---

# Preuve de faisabilite centree sur la proportion tumorale

## Conclusion

L'hypothese selon laquelle ces descripteurs de texture amelioreraient la prediction
ne se confirme pas sur les six patients BCSS disponibles. Aucun modele ne bat la
reference constante pour l'estimation moyenne par region. La preuve de faisabilite
reste donc partielle ; la contribution diagnostique GCIA n'est pas validee.

## Protocole fixe avant calcul

Aucun telechargement supplementaire. Six patients BCSS, un patient reserve a chaque
tour ; les cinq autres servent a l'apprentissage et a la normalisation.
Patches non chevauchants de 128 pixels, sur RGB acquis a environ 1 micrometre/pixel.
Les bordures incompletes sont ecartees et au moins 80 % du patch doit etre annote.
Masques redimensionnes au plus proche voisin a la taille du RGB.

Un GCN a deux couches predit directement la proportion tumorale de chaque patch.
Il s'agit d'un controle positif de prediction de concept, pas d'un audit d'utilisation
du concept dans un diagnostic independant. Graines 11, 23, 37, 120 epochs,
hyperparametres fixes, perte donnant le meme poids a chaque patient d'apprentissage.

Deux representations comparees : six statistiques RGB et ces statistiques enrichies
de quantiles, histogramme et differences spatiales d'intensite. Ces caracteristiques
ne decrivent pas des noyaux segmentes. La reference constante est la moyenne des
fractions tumorales des patches valides d'apprentissage, sans acces au patient reserve.

## Prediction hors patient

Moyennes sur six patients et trois graines, chaque patient ayant le meme poids :

| Representation | MSE par patch | Erreur absolue par region |
|---|---:|---:|
| Reference constante | 0,1388 | 0,1391 |
| RGB | 0,1075 | 0,1463 |
| RGB et texture | 0,1597 | 0,2088 |

L'erreur par region porte sur la fraction moyenne des patches valides uniquement,
pas sur la totalite de la lame. Le RGB ameliore l'erreur par patch mais pas celle par region.
Les descripteurs de texture choisis n'apportent pas l'amelioration esperee.
Cela ne prouve pas que toute representation de texture serait inutile.

## Intervention et controles

Dans chaque patient reserve, un patch contenant au moins 80 % de tumeur est remplace
par un patch stromal du meme patient ; le temoin le remplace par un autre patch tumoral.
Maximum trois receveurs par patient, choisis par ordre spatial, jamais selon la prediction.
Le donneur stromal minimise la difference RGB ; le temoin apparie l'amplitude de ce changement.
Les descripteurs etant independants par patch, remplacer leur ligne par celle du donneur
est exactement equivalent a recalculer les descripteurs apres transplantation du patch.
Cette equivalence fait l'objet d'un test logiciel.

Controles fixes : ecart relatif d'amplitude RGB au plus 20 %, modification des autres
tissus au plus 5 points, et derive de chaque fraction tissulaire du temoin au plus 5 points.
Le stroma augmente deliberement : il ne s'agit pas d'une variation isolee de la tumeur.
Les controles portent sur des fractions calculees depuis les masques, pas sur une
nouvelle lecture de l'image composee par un pathologiste.

12 candidats chez quatre patients ; sept candidats chez trois patients passent ces
controles. Deux patients n'offrent pas les donneurs necessaires. Tous les cas et leurs
exclusions sont conserves dans coverage.csv et substitutions.csv.

Pour les sept cas passant les controles numeriques, moyennes sur trois graines :

| Representation | Changement predit, cible | Changement predit, temoin |
|---|---:|---:|
| RGB | -0,844 point | -0,034 point |
| RGB et texture | -1,061 point | -0,136 point |

Le changement moyen de fraction tumorale calcule depuis les masques est de -1,465 point
pour la cible et +0,014 point pour le temoin. Les variations predites vont en moyenne
dans le sens attendu. Ces sept cas ne sont pas sept patients independants.
Aucune significativite statistique ou causalite biologique n'est revendiquee.

## Decision scientifique

Ne pas etendre les conclusions a la classification diagnostique, ne pas annoncer une
validation de l'abstract initial. L'etape suivante depend de la qualification des
representations et de la validite histologique des transplantations. Accumuler des
perturbations ou ajuster les seuils jusqu'a obtenir un effet ne resout pas ces limites.
Le protocole et les resultats negatifs sont conserves pour guider cette decision.

## Reproduction

Depuis GCIA : `.\.venv\Scripts\python.exe scripts\run_focused_study.py`.
36 ajustements, checkpoints et parametres de normalisation conserves localement.
Voir configs/focused_protocol.json, prediction.csv, coverage.csv, substitutions.csv et
summary.json (empreintes des sources). Le script ne telecharge rien.
