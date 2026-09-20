# Pilote BRACS : normal versus carcinome invasif

40 RoI de 40 patients distincts : 20 train, 10 validation et 10 test interne.
Chaque groupe comporte autant de regions normales que de regions invasives.
Les trois graines sont assemblees par moyenne des probabilites ; seuil fixe a 0,5.

## Selection sur validation

| Modele | Balanced accuracy | AUC |
|---|---:|---:|
| mean_mlp | 0.700 | 0.800 |
| gcn | 0.700 | 0.880 |

Modele retenu : **mean_mlp**. En cas d egalite, preference fixee au modele simple.
Selection sauvegardee avant decodage des images de test.

## Evaluation unique du modele retenu sur le test interne

- Patients : 10.
- Accuracy : 70.0%.
- Balanced accuracy : 70.0%.
- AUC : 0.880.
- Reference constante : balanced accuracy 50 %.

Matrice : lignes = classes reelles, colonnes = classes predites, ordre N puis IC.
`[[4, 1], [2, 3]]`

## Portee

Les patients sont distincts entre groupes. Le test est interne, issu de la
partition officielle train : ce n est ni le test officiel BRACS ni BACH.
La selection des petites images introduit un biais ; dix sujets de test donnent
une estimation tres imprecise. Un changement de prediction change l accuracy de 10 points.
La distinction de ces deux extremes ne couvre pas les lesions intermediaires.
Aucun resultat clinique ni validation interventionnelle GCIA ne sont revendiques.

PCam est gele, mais son domaine et son echelle different des RoI BRACS.
Chaque image est redimensionnee a 768x768 puis divisee en 64 patches 96x96.
L echelle physique des RoI BRACS n est pas etablie ; cette limitation est conservee.

## Tracabilite

Protocoles : configs/bracs_binary_protocol.json, selection.json, test_opened.json.
Predictions individuelles : predictions.csv. Poids et normalisations : *.pt.
Acquisition : data/bracs_binary_manifest.csv et rapports dans reports/bracs_binary_plan.
Le script run_bracs_binary.py refuse de rouvrir le test apres cette execution.
Ce test est maintenant examine et ne doit plus servir a optimiser la methode.
Regenerer ce rapport sans entrainement : scripts/report_bracs_binary.py.
