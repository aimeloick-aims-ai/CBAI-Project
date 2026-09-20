# Passage du pilote à une étude complète

## Avancement de cette étape

- Un spécialiste disponible, déclaré par l'utilisateur. Aucun retour d'annotation reçu à ce stade.
- Lot de découverte indépendant préparé : `../expert_round_1_blinded.zip`, 68 exemples de patients d'apprentissage, noms de groupes et propositions IA retirés. Correspondance conservée séparément dans `../expert_round_1_private_mapping.json` : ne pas transmettre ce fichier au spécialiste.
- Les exemples restent issus d'une sélection de prototypes : ils servent à définir les concepts, pas à mesurer sans biais leur pureté. Une exposition antérieure du spécialiste aux noms proposés doit être déclarée.
- Inventaire de 30 patients candidats dans le test officiel, absents des manifestes BRACS locaux. Ce n'est pas encore un test verrouillé ni une preuve qu'ils n'ont jamais été inspectés ailleurs. Voir `test_candidate_inventory.json`.
- Règles conservatrices ajoutées dans `src/concepts/audit_decisions.py` avec trois tests : pas de conclusion clinique sans annotations indépendantes, probe tenu à l'écart, intervention sélective/plausible et témoins valides ; pas de conclusion d'effet négligeable sans marge préspécifiée. Ce module interprète les preuves fournies, il ne les mesure pas et ne valide pas automatiquement leurs indicateurs.

## Ordre de travail et conditions de passage

1. **Découverte experte** : recevoir ANNOTATIONS.csv et EVALUATEUR.json ; conserver les désaccords avec les noms IA. Ne pas imposer cinq concepts. Avec un évaluateur, pas d'accord inter-évaluateur calculable.
2. **Définition** : fixer les critères observables, l'unité (cellule, patch, interface), les cas indéterminés et une grille de présence indépendante. Faire annoter un nouveau lot aléatoire sans noms de clusters ; les proportions de prototypes ne seront pas publiées comme pureté populationnelle.
3. **Cohorte** : réconcilier inventaire RoI, classes ADH/DCIS/IC et patients officiels. Conserver les patients historiques dans le développement. Vérifier une nouvelle fois les recouvrements avant de figer les listes et leurs empreintes. Le test candidat ne sera pas utilisé pour choisir des seuils ou des concepts. Les images candidates ne sont pas encore acquises dans cette étape.
4. **Représentations cellulaires** : vérifier segmentation, échelle et correspondance des cellules/compartiments avec les annotations. Aucun nom d'un cluster de patches ne devient automatiquement un type cellulaire.
5. **Probing** : probes linéaires, séparation patient stricte, normalisation sur apprentissage, contrôles de labels permutés au niveau patient et de modèles randomisés. Morphologie directement fournie en entrée conservée comme contrôle positif. Comparer plusieurs couches disponibles ; ajouter des couches nécessite un protocole d'entraînement explicite.
6. **Interventions** : phénotype, population et relations spatiales ; contrôles sur norme/distribution, nombre/degré/localisation, longueurs/régions. Mesurer le concept ciblé et les changements collatéraux avec des mesures indépendantes du seul clustering. Les fractions K-means sont compositionnelles et ne prouvent pas cette sélectivité.
7. **Statistiques** : fixer les marges d'effet et de sélectivité sur développement, avant le nouveau test. Rapporter tous les échecs et les effectifs. Un intervalle large qui contient zéro est indéterminé. Même un excès cible-témoin négligeable ne prouve pas une absence universelle d'utilisation du concept.
8. **Externe et papier** : choisir un endpoint BACH compatible, figer les transformations puis appliquer le modèle sans ajustement sur le test externe. Vérifier la bibliographie et les baselines annoncées. Aucun résultat hypothétique du texte transmis n'est ajouté aux tableaux.

## Ce qui ne doit pas être confondu

L'ajout d'un protocole ou d'une règle de décision n'est pas l'exécution des expériences correspondantes. Les huit étapes ne sont pas achevées. L'étude clinique complète dépend notamment du retour réel du spécialiste et des données annotées indépendantes. L'archive LaTeX précédente reste un manuscrit exploratoire ; aucune validation clinique n'y est ajoutée par cette préparation.
