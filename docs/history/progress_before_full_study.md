---
noteId: "6a3fd190b5c811f19aa73b13fcac3a6b"
tags: []

---

> **Etat actuel : [STATUS.md](../STATUS.md).** Les bilans dates ci-dessous sont historiques. Les demonstrations synthetiques ne constituent pas une validation BACH ou clinique.

# Exécution du plan — 18 septembre 2026

> Pilote BRACS binaire execute : [resultats](bracs_binary/RESULTS.md).
> 40 patients distincts (20 train, 10 validation, 10 test interne). MLP et GCN a egalite
> sur validation (balanced accuracy 70 %) ; MLP retenu selon la regle fixee avant calcul.
> Test interne du MLP : accuracy/balanced accuracy 70 %, AUC 0,88, soit 7/10 corrects.
> Le GCN n'a pas ete teste sur ce test reserve. Ce pilote n'est pas une validation clinique
> ni une demonstration de la contribution XAI GCIA. Test desormais examine et ferme au tuning.

> Baseline existante executee : [GNNExplainer](gnnexplainer_baseline/RESULTS.md),
> 24 explications, 1512 perturbations, huit patients avec checkpoints hors apprentissage.
> Comparaison a vingt controles aleatoires par budget. Sensibilite non systematiquement
> superieure au hasard ; adaptation a la regression, sans revendication GCIA.

> Encodeur gele : [comparaison complete](encoder_study/RESULTS.md) et
> [interpretation](encoder_study/CONCLUSION.md). Six patients de developpement,
> deux nouveaux patients reserves. PCam+MLP : MSE patch 0,0697 contre constante 0,1338
> sur les deux nouveaux patients. Gain exploratoire de representation ; aucun gain GNN
> ou resultat GCIA final revendique. Audit visuel dans encoder_study/alignment.png.

> Preuve de faisabilite resserree : [proportion tumorale et transplantations controlees](focused_study/RESULTS.md).
> 36 ajustements sur six patients BCSS ; l'ajout de texture ne donne pas l'amelioration attendue.
> Sept substitutions chez trois patients passent les controles numeriques ; realisme histologique
> non valide. Il s'agit d'un controle positif de prediction de concept, pas de diagnostic GCIA.

> Extension legere : six patients BCSS, trois concepts, modeles geles et baseline gradients.
> Voir [resultats de l'extension](extended_xai/RESULTS.md). Quatre nouveaux patients :
> 12,86 Mo de RGB/masques. Ce resultat ne clot pas les exigences scientifiques de GCIA.

> Actualisation du 19 septembre : experiences exploratoires multi-couches sur BRACS/BCSS
> executees. Voir [resultats et limites](real_xai/RESULTS.md) : 15 entrainements,
> 90 evaluations de sondes, 990 perturbations. Aucun telechargement additionnel.
> Les passages ci-dessous decrivent l'historique ; la validation scientifique GCIA reste ouverte.

## Phase 0 — terminée pour le positionnement prospectif

Livrables : `paper/novelty_matrix.csv`, `paper/related_work.md`, `paper/contribution.md`.

14 méthodes comparées, dont les sept familles explicitement demandées. La revue a identifié des antécédents directs à la distinction encodage/utilisation et à l'audit conceptuel en pathologie. La contribution a été restreinte en conséquence. La recherche ne prétend pas être exhaustive ; les limites de lecture sont indiquées par ligne.

Critère : satisfait pour la formulation prospective retenue, qui ne dépend pas d'une revendication d'absence de travaux. Ce statut n'est ni une validation par les pairs ni une garantie d'originalité absolue. Vérifications de structure : `document_checks.json`.

## Phase 1 — vérifiée localement ; acceptation complète encore ouverte

Arborescence et dépôt Git créés. Python 3.11.15 isolé. 215 dépendances installées, compatibles selon `uv pip check`, avec empreintes dans `environment/requirements-win-py311.lock`. Hooks pre-commit installés ; Ruff et formatage réussis.

Tests : **4 réussis**, couvrant les sorties et gradients de GCN, GraphSAGE et GATv2 ainsi que la lecture des pixels d'un TIFF synthétique par OpenSlide. Rapport JUnit : `software_tests.xml`. Deux avertissements de dépréciation internes à PyTorch/PyG, sans échec. La première fixture TIFF RGB non compressée a échoué ; son encodage a été remplacé par un TIFF RGBA compressé, puis les quatre tests ont été relancés avec succès. Aucune donnée clinique n'est impliquée.

Limites : CPU uniquement, aucune GPU CUDA détectée. Le script de reconstruction existe, mais aucun clone neuf séparé n'a été testé. La lecture d'une vraie lame reste à vérifier après acquisition. Le critère strict de reproductibilité du plan reste donc ouvert ; aucune phase expérimentale suivante n'est déclarée acceptée.

## Phase 2 — active, acquisition pilote réalisée

À la demande de l'utilisateur, acquisition pilote effectuée : 7 RoI BRACS train, résumé BRACS et 2 paires RGB/masque BCSS. 18 fichiers avec reçus contrôlés, 50 066 957 octets. Le script `run_data_smoke.ps1` a terminé sans erreur. Sept tests logiciels passent. Guide de reprise : `data_quickstart.md`.

Le résumé BRACS révèle un patient commun entre train/val pour toutes les WSI, mais aucun parmi les lames avec RoI déclarées. Il reste à réconcilier la cohorte complète. NuCLS et BACH ne sont pas acquis. Le critère de phase 2 pour les quatre jeux complets n'est pas atteint. La vérification d'un clone neuf de phase 1 reste ouverte ; cette acquisition bornée suit la demande explicite de l'utilisateur et ne vaut pas validation des phases expérimentales.

## Phases 3 à 14 — non commencées

Partitions, segmentation, graphes, entraînements, concepts, sondes, interventions, audit, ablations, statistiques, validation externe et rédaction finale restent à exécuter après leurs prérequis. Les répertoires vides ne valent pas implémentation ou validation de ces phases.

## Résultats

Aucune performance prédictive calculée. Les seules mesures sur données réelles sont des contrôles d'intégrité et des fractions de surfaces annotées BCSS. Aucun patient inventé, aucun modèle entraîné. BACH n'a pas été ouvert. Le manuscrit final ne peut pas être produit comme un article de résultats.

## Mise a jour technique

`scripts/run_phase2_smoke.py` construit maintenant des graphes kNN de patchs sur les 7 RoI BRACS pilote et execute GCN, GraphSAGE et GATv2 avec gradients et intervention bornee de feature. Rapport : `phase2_smoke.json`.

Cette execution valide le cablage logiciel des modules `src/data`, `src/segmentation`, `src/graphs`, `src/models`, `src/interventions` et `src/evaluation`. Elle ne remplace pas HoVer-Net, NuCLS, les concepts independants, les entrainements multi-seed ni l'evaluation BACH.

## Mise a jour XAI

Les briques XAI sont maintenant implementees et executees sur une fixture synthetique : sonde lineaire sur representations gelees, evaluation hors echantillon, rejet de fuite patient, intervention, effet sur probabilite et score de specificite face a des controles apparies. Commande : `./.venv/Scripts/python.exe scripts/run_xai_smoke.py`. Le rapport `xai_smoke.json` porte explicitement le statut technique et ne contient aucun resultat histopathologique.

Le resultat XAI scientifique reste indisponible : il exige des annotations de concepts independantes des labels de diagnostic, une segmentation HoVer-Net validee sur NuCLS, un modele entraine sur la cohorte complete et des interventions dont le realisme est evalue independamment.

## Resultat exploratoire BRACS

Un pilote reel sur 21 RoI BRACS (14 train, 7 validation, sept classes, cinq graines) a ete execute avec des graphes de grille RGB, et non avec une segmentation cellulaire. Les moyennes de precision validation sont GCN 25,7 %, GraphSAGE 28,6 % et GATv2 25,7 %. Rapport : `bracs_pilot_results.json`. Ces chiffres ne sont pas des resultats scientifiques finaux : chaque classe de validation ne contient qu'une image, les concepts ne sont pas annotes independamment, HoVer-Net et NuCLS ne sont pas valides et BACH reste ferme.
