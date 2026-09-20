---
noteId: "e51ecd50b4fe11f1ac794713d9e3a04f"
tags: []

---

# Revue des cinq clusters candidats

Objectif : déterminer si chaque groupe correspond à une notion histologique cohérente,
puis proposer un nom et une définition observable. Il est acceptable de ne pas nommer
un groupe ou de proposer une fusion/scission : cinq clusters ne garantissent pas cinq concepts.

1. Ouvrir PLANCHE.pdf dans chacun des cinq dossiers.
2. Examiner les images originales et leur contexte, notamment les contre-exemples.
3. Remplir ANNOTATIONS_EXEMPLES.csv par image puis NOMMER_LES_GROUPES.csv pour la synthèse.
4. Préciser les ambiguïtés, la qualité et la confiance (1 faible, 5 forte).

Les dossiers proviennent du K-means existant à cinq groupes, sur les représentations
du GCN de **patches**. Ce ne sont pas les clusters des nouveaux GNN cellulaires.
Aucun nouveau clustering ni sélection fondée sur le diagnostic n'a été effectué.
Seuls les patients d'apprentissage sont utilisés. Les diagnostics et prédictions ne
sont pas inclus dans ce dossier afin de limiter leur influence sur l'interprétation.

Par groupe : jusqu'à 12 prototypes proches du centroïde, avec un patient distinct
par prototype, puis jusqu'à 4 exemples supplémentaires aléatoires d'autres patients.
Ces exemples sont sélectionnés pour la revue ; ils ne permettent pas d'estimer sans
biais la pureté globale du cluster. Les PNG patch sont les vues exactes du modèle ;
les PNG original conservent les pixels natifs correspondants ; le contexte indique
la zone en jaune. Les mesures physiques ne sont pas certifiées.

Après cette première revue, figer les définitions et les vérifier sur un ensemble
distinct annoté sans ajuster les clusters. Ne pas confondre nommage, pureté mesurée
et preuve d'utilisation : GCIA devra ensuite tester les interventions associées aux
concepts validés, leur sélectivité et leur plausibilité.

Les CSV s'ouvrent dans un tableur. Conserver les identifiants d'exemples et les IDs
des groupes. Pour deux spécialistes, travailler sur deux copies séparées avant consensus.
