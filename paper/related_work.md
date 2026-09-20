# Travaux connexes et décision de positionnement

Recherche ciblée effectuée le 18 septembre 2026. Fenêtre principale : 2020–2026, avec TCAV (2018) et ACE (2019) comme antécédents requis. La matrice distingue les textes consultés des notices et résumés. « Non établi » signifie information non vérifiée, et non absence de cette propriété dans l'article. Cette revue n'est pas une revue systématique exhaustive.

## Ce que GCIA ne peut pas revendiquer

La distinction entre présence d'information dans les représentations et utilisation par le modèle n'est pas nouvelle. [Amnesic Probing](https://doi.org/10.1162/tacl_a_00359) étudie déjà le changement de comportement après retrait d'information. [LEACE](https://arxiv.org/abs/2306.03819) formalise un effacement minimal des informations linéairement accessibles. Ces travaux justifient une baseline d'effacement latent et interdisent de présenter la question générale comme une découverte de GCIA.

[TCAV](https://proceedings.mlr.press/v80/kim18d.html) relie des directions conceptuelles à la sensibilité du classifieur. [ACE](https://arxiv.org/abs/1902.03129) découvre des concepts visuels automatiquement. Les [Concept Bottleneck Models](https://proceedings.mlr.press/v119/koh20a.html) autorisent la correction de valeurs conceptuelles avant la décision. Leurs extensions sur la [procédure d'intervention](https://proceedings.mlr.press/v202/shin23a.html) et l'[apprentissage des interventions](https://proceedings.mlr.press/v235/steinmann24a.html) doivent être considérées lorsque l'on définit les comparateurs. GCIA vise ici un audit post hoc de modèles gelés.

## Antécédents directs sur graphes et images

[GCExplainer](https://arxiv.org/abs/2107.11889) fournit des concepts à partir des représentations de graphes. [Digital Histopathology with Graph Neural Networks](https://arxiv.org/html/2312.02225v1) combine segmentation HoVer-Net, classification GCN, concepts et explications logiques dans le sein. Une réutilisation de cette chaîne ne constitue donc pas à elle seule une contribution méthodologique.

[Concept Backpropagation](https://arxiv.org/abs/2307.12601) modifie l'entrée en suivant une sonde conceptuelle. Cette méthode motive l'intervention guidée par sonde, mais GCIA doit vérifier séparément le changement du concept, les autres concepts et la décision. [CF-GNNExplainer](https://arxiv.org/abs/2102.03322) recherche des suppressions d'arêtes changeant la prédiction. Un tel changement n'établit pas à lui seul une transformation histologique valide.

## Publications récentes modifiant le positionnement

Le préprint [PICASSO](https://repository.cshl.edu/id/eprint/42227/) décrit une décomposition conceptuelle des embeddings de modèles de fondation en pathologie, leur modulation et un retour à l'image par Emb2Img. Les sections accessibles [sur PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13307983/) décrivent aussi l'audit de prédictions et les artefacts. Cela recouvre fortement une revendication générale d'audit interventionnel en histopathologie. Différence de travail proposée, à vérifier expérimentalement : GNN cellulaires gelés, comparaison multicouche et contrôles conjoints de spécificité et de validité, puis transfert BRACS–BACH.

[Cellina](https://arxiv.org/abs/2606.08493) étudie les contrefactuels sur graphes de tissus pour l'expression cellulaire. Son objet n'est pas identique au diagnostic H&E, mais les interventions sur voisinage ne sont pas une nouveauté générale. [MoPaDi](https://aacrjournals.org/cancerres/article/doi/10.1158/0008-5472.CAN-25-5645/786406/Counterfactual-Diffusion-Models-Provide) traite les contrefactuels de pathologie par diffusion ; ses détails de validité restent à examiner dans le texte intégral.

## Décision et précisions méthodologiques

Retenir la formulation prudente de `contribution.md`. Ne pas utiliser « first », ni revendiquer une causalité biologique. Le critère documentaire de phase 0 est satisfait pour ce positionnement prospectif : il s'appuie sur des différences explicites et non sur une absence supposée de travaux. La matrice ne certifie pas une priorité scientifique.

Avant les expériences, préciser les points suivants dans le protocole statistique :

- Une fraction cellulaire continue exige des métriques de régression ; AUROC et F1 ne conviennent qu'après définition préalable d'une cible binaire ou catégorielle.
- Optimiser une sonde puis utiliser cette même sonde comme unique preuve de succès crée un risque circulaire. Ajouter une mesure de concept indépendante ou une annotation avant/après.
- Un effet non significatif n'est pas une preuve de non-utilisation. Définir une marge d'équivalence et une puissance suffisante ; sinon conclure « indéterminé ».
- L'effet mesure la sensibilité à une famille d'interventions valides. Il ne prouve ni une nécessité universelle du concept, ni un mécanisme biologique.
- Les annotations de NuCLS/BCSS ne deviennent pas automatiquement des vérités terrain BRACS/BACH. Leur transfert nécessite sa propre validation.
- Une intervention d'arêtes avec coordonnées fixes peut contredire la règle kNN/rayon. Définir explicitement le support admissible.
- Ne pas inventer de groupes patients BACH : si les identifiants manquent, documenter l'unité disponible et limiter l'inférence.

## Journal de recherche

Familles de requêtes : noms exacts des sept méthodes requises ; « graph concept interventions pathology concept decodability use counterfactual 2025 2026 » ; « concept erasure LEACE probing information use » ; « Amnesic Probing » ; « Dissecting and directing pathology foundation models ». Sources primaires : arXiv, PMLR, TACL/MIT Press, NeurIPS, dépôts des auteurs, éditeurs et dépôt institutionnel CSHL. Le texte HTML de Digital Histopathology a été consulté. L'ouverture directe de PMC a rencontré un contrôle navigateur ; les sections indexées et la notice institutionnelle ont été utilisées. La tentative HTML de Concept Backpropagation a échoué ; son résumé a été utilisé.

À refaire avant soumission : recherche de citations des antécédents, lecture intégrale des concurrents récents, archivage des versions et comparaison des suppléments. Aucune valeur de performance de ces articles n'a été importée comme résultat GCIA.
