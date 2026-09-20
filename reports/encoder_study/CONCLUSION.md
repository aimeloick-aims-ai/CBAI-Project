# Interpretation de la comparaison fixee

Un signal positif exploratoire apparait pour la representation PCam gelee.
Sur les deux nouveaux patients, le MLP donne une MSE patch de 0,0697 contre
0,1338 pour la reference constante et 0,2035 pour le MLP avec RGB seul.
La MAE de proportion regionale est de 0,0608 contre 0,0993 pour la constante.

Le GCN avec PCam obtient 0,0694 de MSE patch et 0,1160 de MAE regionale.
Il ne demontre donc pas de gain clair par rapport au MLP, et sa MAE regionale
reste superieure a celle de la reference constante.

Ce resultat soutient la piste d'une representation preentrainee mieux adaptee
que les statistiques RGB sur cet echantillon. Il ne prouve pas la superiorite
generale de PCam, l'utilite du graphe ou la fidelite interventionnelle GCIA.
Les resultats varient entre les deux patients ; toutes les conditions sont
conservees dans RESULTS.md et prediction.csv, y compris celles moins bonnes.

Les deux patients reserves ont maintenant ete examines : ne pas les reutiliser
comme test intact apres de nouveaux choix. Une confirmation independante et
la validation histologique des interventions restent necessaires.

Audit visuel : aucun decalage grossier observe dans la planche reduite alignment.png.
Cette inspection technique ne garantit pas l'exactitude de chaque contour.
Poids et nouveaux RGB/masques : environ 20,1 Mo, plus quelques metadonnees.
