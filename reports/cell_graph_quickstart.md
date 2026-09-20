---
noteId: "2ed56510b4c511f1ac794713d9e3a04f"
tags: []

---

# Pipeline de graphes cellulaires candidats

## Commande complète

```powershell
.\scripts\run_cell_pipeline.ps1
```

Cette commande génère automatiquement la cohorte, son manifeste et l'entraînement
comparatif. Attendre la fin de l'acquisition déjà active avant de la lancer pour éviter
deux téléchargements du même fichier. Le calcul peut être long sur CPU. Les sorties
sont dans `reports/cell_pipeline`, dont `training/results.json`. Une erreur arrête
le pipeline ; aucun patient n'est exclu silencieusement. Cette exécution sur cohorte
réelle reste à réaliser ; les graphes sont candidats tant que la segmentation n'est
pas évaluée sur annotations indépendantes.

## HoVer-Net préentraîné et entraînement comparatif

Acquisition officielle avec révision fixée et vérification SHA256 fournie par Hugging Face :

```powershell
.\.venv\Scripts\python.exe scripts/acquire_hovernet.py
.\.venv\Scripts\python.exe scripts/build_cell_graph.py --segmenter hovernet --weights data/pretrained/hovernet_fast-pannuke.pth --crop-size 256 --output reports/cell_graph_hovernet
```

Le modèle est `hovernet_fast-pannuke` de TIACentre, utilisé via TIAToolbox installé.
Les entrées 256×256 produisent des cartes centrales 164×164 assemblées avant un
post-traitement global. La résolution cible documentée dans la configuration du
modèle est 0,25 µm/pixel. Aucune résolution BRACS inconnue n'est inventée, aucun
redimensionnement implicite n'est réalisé. Les performances aux joints des tuiles
et sur le domaine cible restent à évaluer avec des annotations.

L'état de l'essai automatique en cours se lit dans `reports/hovernet_job.json`,
son journal dans `reports/hovernet_smoke.log`. Un état `complete` signifie que
l'intégration s'est exécutée, pas que la segmentation est scientifiquement validée.

Après construction et contrôle de qualité de la cohorte, créer un manifeste CSV
avec les colonnes `patient_id,split,label,graph_path,sha256` : un graphe par patient,
`split` parmi `train,validation,test`, `label` parmi `0,1`, chemin du `graph.pt`
relatif au manifeste et empreinte SHA256 du fichier. Les trois partitions doivent
contenir les deux classes. Ne pas appeler « test indépendant » une cohorte déjà explorée.

```powershell
.\.venv\Scripts\python.exe scripts/train_cell_gnns.py --manifest data/cell_graph_manifest.csv --output reports/cell_gnn_comparison
```

Cette commande entraîne GCN, GraphSAGE et GATv2, chacun sur les graines 11/23/37,
avec 120 époques fixes. La normalisation est calculée uniquement sur l'apprentissage.
Les prédictions moyennes des graines sont évaluées à seuil 0,5. Les poids, paramètres,
prédictions et métriques sont sauvegardés ; les résultats existants sont protégés
contre l'écrasement. Le manifeste de cohorte réel reste à produire après validation
de segmentation ; le test d'intégration emploie uniquement des fixtures synthétiques.

Le code accepte une image RGB et une carte d'instances nucléaires, puis extrait un
nœud par instance retenue. Il n'utilise pas les poids du classifieur de patches :
les nouveaux attributs nécessiteront un entraînement distinct.

## Démonstration locale

Depuis `GCIA`, avec l'environnement existant :

```powershell
.\.venv\Scripts\python.exe scripts/build_cell_graph.py
```

Sans argument, une image **d'apprentissage** du manifeste BRACS est sélectionnée,
son empreinte est vérifiée, et un recadrage central de 768 pixels maximum est utilisé
sans redimensionnement. La segmentation watershed sur hématoxyline est une baseline
technique non validée, pas HoVer-Net et pas une annotation clinique.

Les sorties sont dans `reports/cell_graph_smoke` :

- `overlay.png` : image, contours des instances candidates et graphe.
- `instances.npy` : identifiants des instances, zéro pour le fond.
- `graph.pt` : dictionnaire des tenseurs PyTorch, identifiants et noms des attributs.
- `quality.json` : provenance, recadrage, unités, nœuds, connexions, nœuds isolés et métriques disponibles.

Le script refuse d'écraser les résultats. Utiliser `--output reports/cell_graph_v2`
pour une exécution distincte ; ne pas changer les paramètres pour choisir de beaux résultats.

La première exécution produit 1 492 instances candidates et aucun nœud isolé.
L'inspection de `overlay.png` révèle de nombreux petits fragments et des objets
découpés en plusieurs instances. Ce nombre ne doit donc pas être interprété comme
un dénombrement fiable des noyaux. Cette baseline est impropre à une conclusion
scientifique sur l'organisation cellulaire ; elle valide seulement la circulation
des données dans le pipeline. La remplacer et l'évaluer avant tout entraînement
présenté comme une expérience sur graphes cellulaires.

## Segmentation prédite et annotations indépendantes

```powershell
.\.venv\Scripts\python.exe scripts/build_cell_graph.py `
  --image chemin/image.png `
  --instances chemin/prediction_instances.npy `
  --reference chemin/annotation_instances.npy `
  --mpp 0.5 --radius 50 --crop-size 0 `
  --output reports/cell_graph_annotated
```

`0.5` est seulement un exemple : renseigner la résolution physique documentée de
l'image. Sans `--mpp`, distances et aires restent en pixels et pixels carrés.
Les cartes doivent être des matrices entières de mêmes dimensions que l'image,
avec un identifiant positif distinct par noyau. **Un masque sémantique BCSS de
tumeur/stroma n'est pas une carte d'instances nucléaires.** Aucun redimensionnement
ou alignement implicite des annotations n'est effectué.

Mesures : Dice du premier plan, détection F1, qualité de segmentation et qualité
panoptique, avec appariement un-à-un à IoU strictement supérieur à 0,5. Les cas sans
instances sont explicitement signalés par `null` lorsque le score est indéfini.
Les métriques s'appliquent au recadrage si celui-ci est activé ; les noyaux tronqués
aux frontières sont conservés. Aucune acceptation scientifique automatique n'est déduite des scores.

## Attributs et connexions

Aire, excentricité, solidité, axes majeur/mineur et couleur moyenne RGB ; positions
aux centroïdes. Jusqu'à huit voisins par nœud sont recherchés, puis les connexions
sont symétrisées et limitées par le rayon. La symétrisation peut donner plus de huit
connexions à un nœud. Les nœuds isolés sont conservés et comptés. Aucun type cellulaire
n'est attribué à partir de la couleur ou de la forme.

Les tests vérifient les métriques, les fusions, les unités physiques, les connexions
et la rétropropagation pour les trois architectures existantes. Ils ne valident pas
la précision biologique de la baseline. Restent à intégrer : modèle de segmentation
validé, évaluation sur annotations nucléaires et entraînement des GNN cellulaires
avec séparation des patients, avant les audits de concepts.
