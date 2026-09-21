---
noteId: "ada319b0b5c811f19aa73b13fcac3a6b"
tags: []

---

# Vérifier et reproduire la nouvelle étude

Depuis GCIA, avec l'environnement Python préparé :

```powershell
.\.venv\Scripts\python.exe scripts/run_all_checks.py
```

Preuves : `reports/software_verification/junit.xml`, sorties Ruff et `verification.json`. Celui-ci conserve le HEAD de base, les modifications locales et les SHA256 des sources effectivement testées. Ne pas présenter une exécution sur fichiers non commités comme une vérification du seul HEAD. Le workflow GitHub est configuré, mais son exécution distante n'est pas attestée ici.

## Benchmark synthétique

```powershell
.\.venv\Scripts\python.exe experiments/full_study/run_synthetic.py
```

Ce script refuse d'écraser `reports/full_study/synthetic_v1`. Pour reproduire, utiliser une copie de travail sans ce dossier en conservant les originaux. Configuration : `experiments/full_study/config.json`. Les données sont générées localement ; aucun téléchargement nécessaire. Résultats : [RESULTS.md](reports/full_study/synthetic_v1/RESULTS.md).

Les poids entraînés et tableaux de représentations sont enregistrés localement. Les fichiers `.pt` sont ignorés par Git : un clone doit relancer l'expérience pour les recréer. Les graines, paramètres, protocoles et empreintes sont conservés. Les tableaux de récupération sont des diagnostics descriptifs sur un espace factoriel simple, pas une validation clinique ni un test d'équivalence.

## Pilote historique

Ne pas recalculer ni optimiser ses anciens tests pour la nouvelle étude. `experiments/full_study/pilot_frozen_hashes.json` recense les empreintes des sorties XAI historiques et des archives existantes à l'ouverture de cette phase.

Le protocole complet et les conditions non satisfaites sont dans [FULL_STUDY_PROTOCOL.md](FULL_STUDY_PROTOCOL.md). Aucun test BRACS nouveau ou BACH n'est ouvert par les commandes ci-dessus.
