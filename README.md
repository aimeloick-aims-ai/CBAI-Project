> **Etat actuel : [STATUS.md](STATUS.md).** Les bilans dates ci-dessous sont historiques. Les demonstrations synthetiques ne constituent pas une validation BACH ou clinique.

# GCIA — audit interventionnel de concepts histopathologiques

Projet de Mahugnon Aime Loick Gohouede, préparé à partir de `Plan_scientifique_GCIA_histopathologie.docx`. Les pipelines pilotes sont implémentés et ont produit des résultats exploratoires. Le protocole clinique complet reste à valider ; consulter [STATUS.md](STATUS.md) pour l'état actuel.

## Bilan du 18 septembre 2026

**Mise à jour : données réelles acquises et code exécuté.** Le projet contient maintenant 7 RoI BRACS d'entraînement (une par classe), `BRACS.xlsx` et 2 régions BCSS avec leurs masques. Les 18 fichiers téléchargés, métadonnées comprises, représentent 50 066 957 octets et leurs empreintes sont vérifiées. Sept tests logiciels passent. Ce petit lot technique ne constitue pas une cohorte complète.

Depuis le dossier `GCIA`, relancer les contrôles avec :

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_data_smoke.ps1
```

Ce programme a été exécuté avec succès. Il contrôle les métadonnées BRACS, les annotations BCSS et l'intégrité des téléchargements ; **il n'entraîne aucun modèle**. Voir [le guide actuel](reports/data_quickstart.md), [l'audit des fichiers](reports/acquisition_checks.json) et [l'audit BCSS](reports/data_audit.html).

La revue bibliographique et le positionnement sont rédigés. L'environnement Windows contient 215 dépendances installées et compatibles, avec versions et empreintes verrouillées. **Sept tests passent** : quatre contrôles CPU et trois tests du traitement des annotations. Les contrôles de formatage passent également. Rapports : `reports/environment_check.json` et `reports/software_tests.xml`. Aucune GPU CUDA n'est détectée. La reproduction depuis un clone neuf et la lecture d'une vraie WSI restent à vérifier.

L'accès FTP BRACS est débloqué et testé. Les expériences attendent encore les cohortes complètes, NuCLS et l'implémentation des étapes de segmentation, graphes et entraînement. Un chevauchement patient train/validation a été repéré dans le résumé des WSI, absent parmi les lames ayant des RoI déclarées ; il reste à contrôler sur l'inventaire complet. Aucune performance ni validation clinique n'est revendiquée.

## Livrables actuels

- [Matrice de nouveauté](paper/novelty_matrix.csv) : 14 méthodes, sources et niveau de vérification.
- [Analyse bibliographique](paper/related_work.md) et [contribution révisée](paper/contribution.md).
- [Registre des acquisitions](data/registry.yaml) et [guide actuel des données](reports/data_quickstart.md).
- [État d'avancement](reports/progress.md), [journal](CHANGELOG.md) et configurations.
- Scripts de vérification de l'environnement et tests d'intégration CPU.

Les observations cliniques, identifiants patients, partitions, checkpoints et performances restent absents tant que leurs étapes n'ont pas été exécutées sur les données autorisées.

## Environnement Windows

Python isolé 3.11.15 ; `uv` doit être disponible. Le fichier `environment/requirements.in` exprime les dépendances demandées. Le lock est produit par résolution, jamais rempli avec des versions devinées. Les composants GPU et l'environnement Linux nécessiteront une résolution séparée.

Depuis ce dossier, après présence du lock validé :

```powershell
.\scripts\bootstrap.ps1
```

Pour relancer uniquement les vérifications dans l'environnement préparé :

```powershell
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check scripts tests
.\.venv\Scripts\python.exe -m ruff format --check scripts tests
```

Les petits graphes et le TIFF générés par les tests sont des fixtures logicielles explicites, sans identifiant patient et sans usage dans les résultats scientifiques. Le test vérifie les gradients vers les attributs nodaux pour GCN, GraphSAGE et GATv2, puis le décodage de pixels par OpenSlide. Il ne valide pas HoVer-Net, les poids préentraînés, les formats des cohortes ni les interventions.

Pour tester ultérieurement une vraie lame autorisée, hors test externe :

```powershell
.\.venv\Scripts\python.exe scripts/check_environment.py --wsi 'CHEMIN_VERS_UNE_LAME_DE_DEVELOPPEMENT'
```

Le script écrit un rapport JSON et retourne un code d'échec si un import ou un test échoue. L'absence de WSI réelle est explicitement signalée ; elle n'est pas assimilée à une validation clinique.

## Expérience BRACS élargie (66 patients)

Le téléchargement plafonné à 1 Go est terminé : 49 images supplémentaires, dont 22 patients nouveaux. L'expérience binaire N/IC utilise 34 patients d'apprentissage, 10 de validation et 22 de test interne distinct. Le GCN choisi sur validation obtient 20/22 classifications correctes, une balanced accuracy de 90,2 % et une AUC de 0,955. Ce petit échantillon biaisé vers les petits fichiers ne valide pas encore l'abstract GCIA ni la fidélité des explications.

Résultats et limites : [rapport de l'expérience](reports/bracs_expanded/RESULTS.md). Les prédictions, poids et traces de sélection sont dans `reports/bracs_expanded/`. Le test a été ouvert ; ne pas relancer l'entraînement pour ajuster ses résultats. Pour régénérer uniquement le rapport :

```powershell
.\.venv\Scripts\python.exe scripts/report_bracs_expanded.py
```

## Audits XAI du GCN BRACS élargi

[Rapport des trois audits](reports/bracs_three_xai/RESULTS.md) : GNNExplainer officiel (66 explications), adaptation automatique de GCExplainer et pilote GCIA sur trois propriétés visuelles proxy. À 10 % de nœuds remplacés, GNNExplainer réduit la confiance de 10,60 points contre 1,33 pour les contrôles aléatoires. Le surrogate inspiré de GCExplainer reproduit 22/22 décisions du GCN, sans validation sémantique des groupes. Le pilote GCIA ne valide pas les concepts pathologiques ni l'abstract complet. Les poids et les prédictions initiales du classifieur sont inchangés.

Les résultats concernent le test interne déjà consulté ; aucune nouvelle validation externe n'est revendiquée. Pour régénérer le rapport et les figures sans relancer l'audit :

```powershell
.\.venv\Scripts\python.exe scripts/report_bracs_three_xai.py
```

## Audit GCIA avec annotations tissulaires

L'[audit annoté BCSS](reports/annotated_gcia/RESULTS.md) utilise maintenant les fractions annotées de tumeur, stroma et infiltrat lymphocytaire, avec probes à trois niveaux et validation par exclusion d'un patient sur huit. Il inclut des substitutions de pixels réencodés avec contrôles et des suppressions de connexions avec contrôles de longueur et détection de déconnexion. Le GCN BRACS est gelé. Ce test inter-cohortes est exploratoire : il ne mesure pas la performance diagnostique sur BCSS et ne valide pas la plausibilité des interventions.

Les probes de tumeur et d'infiltrat battent la référence constante en MSE moyenne ; celle de stroma ne la bat pas. Les effets des substitutions sont proches du contrôle, et les suppressions ciblées d'arêtes ont un effet plus faible que les contrôles. La contribution GCIA complète reste donc non démontrée. Régénération du rapport :

```powershell
.\.venv\Scripts\python.exe scripts/report_annotated_gcia.py
```

## Pipeline de graphes cellulaires candidats

Le [guide des graphes cellulaires](reports/cell_graph_quickstart.md) décrit la nouvelle
commande `scripts/build_cell_graph.py`. Elle accepte des cartes d'instances nucléaires,
extrait la morphologie, construit les connexions spatiales et calcule les métriques
de segmentation si une annotation indépendante est fournie. La démonstration locale
utilise une baseline watershed non validée et une image d'apprentissage ; elle ne
constitue pas une validation de segmentation ou de diagnostic.

## Évaluation nucléaire et construction de cohorte en cours

La chaîne lancée évalue HoVer-Net et la baseline watershed sur les 14 images complètes
du test officiel MoNuSeg, puis construit 66 graphes BRACS candidats. Consulter le
[suivi local](reports/segmentation_progress.html), les
[scores par image](reports/segmentation_evaluation/metrics.csv) et, après calcul,
le [rapport complet](reports/segmentation_evaluation/RESULTS.md).
Les graphes et leur manifeste sont produits dans `reports/cell_cohort`.
La performance de segmentation sur BRACS et l'indépendance par rapport au
préentraînement PanNuke ne sont pas établies par cette évaluation MoNuSeg.

## XAI des nouveaux modèles cellulaires : exécuté

Le [rapport XAI cellulaire](reports/cell_xai/RESULTS.md) présente 198 explications
GNNExplainer sur 22 patients et trois architectures gelées, les contrôles aléatoires,
les probes morphologiques et les interventions GCIA d'aire nucléaire (±5 %, axes
ajustés conjointement). Un contrôle supplémentaire utilise des perturbations
aléatoires cohérentes entre noyaux. Les données brutes, poids des probes et figures
sont dans `reports/cell_xai`. Ces résultats sont exploratoires et ne valident pas la
plausibilité histologique des interventions. Le code est `scripts/audit_cell_xai.py` ;
pour régénérer seulement le rapport : `scripts/report_cell_xai.py`.

## Reprise scientifique

Le pilote BRACS/BCSS est acquis, enregistré et audité. Suivre `reports/data_quickstart.md` pour étendre l'acquisition. Il reste à réconcilier l'inventaire complet avec les métadonnées et à préparer NuCLS avant la segmentation et les graphes. BACH reste réservé à l'évaluation externe verrouillée.

Le plan exige une seule phase active et interdit d'avancer si le critère précédent échoue. `reports/progress.md` distingue les livrables préparés des phases effectivement acceptées. La phase de recherche complète et le manuscrit ne sont pas terminés.

## Mise a jour pipeline technique

Une commande de bout en bout est maintenant disponible :

```powershell
.\.venv\Scripts\python.exe scripts/run_phase2_smoke.py
```

Elle charge les 7 RoI BRACS autorisees, verifie l'absence de fuite patient dans ce pilote, construit 16 noeuds par image a partir de patchs, cree des graphes kNN, execute GCN/GraphSAGE/GATv2 et teste une edition bornee de feature. Le rapport est [phase2_smoke.json](reports/phase2_smoke.json). Ce rapport est un controle d'integration logiciel, pas une performance scientifique.
