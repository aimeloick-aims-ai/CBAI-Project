# Comprendre et continuer GCIA

**But : distinguer ce que le GNN représente de ce qu'il utilise pour prédire.**
Une *probe* essaie de retrouver un concept dans ses représentations. Une intervention modifie le graphe ; on compare son effet à celui de témoins appariés. Un concept décodable n'est pas forcément utilisé.

## 1. La chaîne à comprendre

`Image H&E → segmentation des noyaux → graphe → GNN → prédiction → audit`

Un graphe contient `x` (caractéristiques des nœuds), `pos` (positions), `edge_index` (connexions) et `y` (diagnostic). `batch` indique à quel graphe appartient chaque nœud quand plusieurs graphes sont traités ensemble.

**Deux chaînes existent :** les anciens graphes de **patches** et les graphes **cellulaires**, où chaque noyau segmenté devient un nœud. Les cinq clusters nommés provisoirement viennent des patches : leurs noms ne sont pas des types cellulaires validés.

## 2. Lire ces fichiers dans cet ordre

| Étape | Fichier | Question à comprendre |
|---|---|---|
| État actuel | [STATUS.md](STATUS.md) | Qu'est-ce qui est exécuté, bloqué ou manquant ? |
| Données | [manifeste cellulaire](reports/cell_cohort/manifest.csv) | Quel patient, quel graphe, quelle partition ? |
| Segmentation | [hovernet.py](src/segmentation/hovernet.py), [nuclei.py](src/segmentation/nuclei.py) | Comment extraire et évaluer les noyaux ? |
| Graphe | [cells.py](src/graphs/cells.py) | Comment calculer les caractéristiques et relier les noyaux ? |
| Modèle | [gnn.py](src/models/gnn.py) | Comment les voisins contribuent-ils à la prédiction ? |
| Entraînement | [train_cell_gnns.py](scripts/train_cell_gnns.py) | Comment normaliser, entraîner et évaluer sans mélanger les patients ? |
| Audit exécuté | [audit_cell_xai.py](scripts/audit_cell_xai.py) | Comment expliquer les modèles gelés et tester la morphologie ? |
| Concepts | [probes.py](src/concepts/probes.py), [audit_decisions.py](src/concepts/audit_decisions.py) | Comment mesurer le décodage et limiter les conclusions ? |
| Structure | [interventions.py](src/graphs/interventions.py), [audit_cluster_structure.py](scripts/audit_cluster_structure.py) | Comment retirer des nœuds/arêtes et construire les témoins ? |

Dans un script, commence par `main()`, puis suis les fonctions appelées. Lis ensuite son test correspondant dans `tests/`.

## 3. Se repérer dans les dossiers

- `src/` : fonctions et modèles réutilisables ; `scripts/` : expériences qui les assemblent.
- `configs/` : paramètres et protocoles ; `data/` : images et manifestes.
- `tests/` : vérifications logicielles ; `reports/` : résultats, protocoles, empreintes et figures.
- `paper/` : manuscrit et archives ; `examples/toy_demo/` : exemples artificiels, sans valeur de validation clinique.

Les fichiers `.pt` contiennent notamment modèles ou graphes ; `.npz`, des tableaux ; `.csv`, les observations ; `.json`, les paramètres et synthèses. Un SHA256 permet de vérifier qu'un fichier n'a pas changé.

## 4. Exécuter sans perdre les résultats

Dans PowerShell, **depuis le dossier GCIA** :

```powershell
# Vérifier le logiciel
.\.venv\Scripts\python.exe -m pytest -q

# Tester seulement les règles de conclusion
.\.venv\Scripts\python.exe -m pytest tests/test_audit_decisions.py -q

# Consulter les options d'entraînement, sans lancer l'entraînement
.\.venv\Scripts\python.exe scripts/train_cell_gnns.py --help
```

Avant de lancer une expérience, vérifie ses chemins de sortie dans `main()`. Plusieurs scripts refusent d'écraser leurs résultats : conserve cette protection. Travaille dans une nouvelle copie ou un nouveau dossier de sortie pour une nouvelle expérience. Les audits supposent des données et checkpoints déjà préparés.

**Ne lance pas `run_cell_gcia.py` comme pipeline complet** : ce démonstrateur est bloqué. `acquire_bach.py` ne télécharge pas BACH. Voir [les corrections](reports/CODE_REVIEW_REMEDIATION.md).

## 5. Lire les résultats

- [Classification cellulaire](reports/cell_gnn_comparison/results.json) : accuracy = proportion correcte ; balanced accuracy = moyenne des rappels par classe ; AUC = capacité à classer les cas selon leur score.
- [Audit cellulaire](reports/cell_xai/summary.json) : R² des probes et effets des perturbations. Décoder l'aire, déjà fournie en entrée, ne valide pas un concept clinique indépendant.
- [Audit des clusters](reports/candidate_cluster_audit/RESULTS.md) et [audit structurel](reports/cluster_structure_audit/RESULTS.md) : différence cible–témoin, effectifs et intervalles d'incertitude. Figures `effects.png` dans ces dossiers.

Un effet cible supérieur au témoin indique une sensibilité sous cette intervention. Il ne prouve une dépendance au concept nommé que si les annotations, la sélectivité et la plausibilité sont établies. Un intervalle contenant zéro signifie souvent **incertain**, pas « inutilisé ». Les unités indépendantes sont les patients, pas les graines ni les répétitions.

## 6. Continuer toi-même

1. Choisir **une question et un protocole** avant de modifier le modèle : par exemple, le GNN exploite-t-il les relations au-delà des caractéristiques des cellules ?
2. Garder les patients séparés ; ajuster normalisation et paramètres sur l'apprentissage/validation. Le test historique est déjà consulté.
3. Ajouter une modification, son test utile, puis une expérience avec sortie distincte. Consigner paramètres, contrôles et résultats négatifs.
4. Poursuivre les étapes du [plan complet](reports/full_study_preparation/PLAN_EXECUTION.md) : concepts indépendants, typage/échelle physique, interventions sélectives et vraie validation externe.

**Le pilote fonctionne ; le protocole clinique complet reste à terminer.** La confirmation des annotations IA par un spécialiste est rapportée par l'utilisateur ; elle n'est pas une revue indépendante en aveugle. Un test logiciel réussi ne démontre pas une hypothèse scientifique.

Dans VS Code : **Ctrl+Shift+V** pour afficher ce guide.
