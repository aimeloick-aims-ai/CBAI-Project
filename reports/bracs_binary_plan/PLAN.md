# Faisabilite : normal versus carcinome invasif sur BRACS

Inventaire local du FTP observe le 18 septembre 2026, recoupe avec BRACS.xlsx.
Aucun nouveau telechargement ni entrainement effectue par cette etape.

## Disponibilite observee

| Partition officielle | Classe de RoI | Images | Patients |
|---|---|---:|---:|
| Train | Normal | 357 | 46 |
| Train | Invasif | 521 | 46 |
| Validation | Normal | 46 | 8 |
| Validation | Invasif | 47 | 4 |

Les nombres de patients par classe ne s'additionnent pas necessairement : un patient
peut avoir des regions de classes differentes. Aucun chevauchement de patients entre
les partitions officielles des RoI inventoriees. Le test officiel n'est pas inventorie.

## Proposition equilibree

| Usage propose | Normal | Invasif | Patients distincts |
|---|---:|---:|---:|
| Apprentissage | 10 | 10 | 20 |
| Validation interne | 5 | 5 | 10 |
| Test interne reserve | 5 | 5 | 10 |

40 images de 40 patients distincts. Une image par patient, sans patient commun
entre groupes. Les patients de validation et de test n'ont aucune RoI deja acquise
localement ; ils n'ont donc pas ete inclus dans nos experiences precedentes.

Telechargement supplementaire estime : **233 921 626 octets, soit 233,9 Mo / 223,1 Mio**.
Ce plan depasse le repere exploratoire de 150 Mio choisi pour l'inventaire.
Ce repere n'est pas une autorisation de depense ni une contrainte imposee par l'utilisateur.
Chaque fichier selectionne est inferieur ou egal a 16 Mio.

Tout est selectionne dans la partition officielle train : validation et test sont
des repartitions internes proposees, pas les partitions officielles BRACS. Le test
officiel et BACH restent en dehors de cette proposition.

## Regles et limites

Patients ordonnes par hash fixe, test reserve en premier ; classe normale puis invasive.
Chez chaque patient eligible, la plus petite image de la classe est retenue.
Ce choix economise la bande passante mais biaise potentiellement taille et morphologie.
Un premier calcul avec plafond de 8 Mio ne permettait pas la repartition souhaitee ;
le plafond de 16 Mio et le quota de 10 patients d'apprentissage par classe sont fixes
uniquement a partir des metadonnees, sans scores de modele.

Les labels sont ceux des regions, pas des diagnostics attribues au patient entier.
Le projet porterait sur la distinction de RoI normales et invasives, pas sur le
diagnostic complet des lesions mammaires. Cet effectif est un pilote de faisabilite,
pas un calcul de puissance et pas une garantie de resultats positifs.

## Suite experimentale proposee

1. Acquerir le manifeste fixe avec verification des empreintes et du volume.
2. Garder les images du test interne fermees pendant les choix de modele.
3. Comparer encodeur gele + modele simple et le meme encodeur + GNN.
4. Choisir sur validation selon un critere fixe puis evaluer une fois sur test.
5. Examiner ensuite les explications, sans reutiliser ce test pour optimiser GCIA.

Selection exacte : proposed_manifest.csv. Inventaire et empreintes : summary.json.
Recalcul hors reseau : scripts/plan_bracs_binary.py.
