---
noteId: "fd67b740b37e11f18317010d52b875d5"
tags: []

---

# Données acquises et exécution

## Disponible localement

| Données | Contenu | Contrôles exécutés |
|---|---|---|
| BRACS | `data/raw/BRACS/BRACS.xlsx` et 7 RoI train, une par classe | PNG, taille FTP, empreintes, rattachement patient/lame |
| BCSS | `data/raw/BCSS/smoke_v1` : 2 RGB à 1 µm/pixel et masques natifs | Codes, dimensions, empreintes, fractions de surfaces annotées |

18 fichiers avec reçus, métadonnées comprises, soit 50 066 957 octets. Voir `data/checksums.csv`, `data/bracs_smoke_manifest.csv` et `data/bcss_smoke_manifest.csv`. Les identifiants patients sont tirés des sources. Le lot est choisi pour un test technique, sans représentativité statistique.

## Exécuter les contrôles

Depuis `GCIA` :

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_data_smoke.ps1
.\.venv\Scripts\python.exe -m pytest -q
```

La première commande a été exécutée avec succès. Elle fonctionne hors ligne et régénère les audits. Sept tests logiciels passent. Aucun modèle n'est entraîné par ces commandes.

## Télécharger davantage de BRACS

Le serveur officiel fourni par le portail connecté est `histoimage.na.icar.cnr.it`. L'accès FTP est testé. Usage limité à la recherche non commerciale d'après les conditions fournies par l'utilisateur. Aucun identifiant FTP n'est enregistré dans les fichiers du projet.

```powershell
.\.venv\Scripts\python.exe scripts/bracs_data.py --download --per-class 2
```

Le script demande le nom d'utilisateur puis le mot de passe masqué. Il accepte aussi les variables de processus `BRACS_FTP_USER` et `BRACS_FTP_PASSWORD`. Les copies existantes sont vérifiées. Le pilote sélectionne les premiers noms de fichiers sous un plafond de 16 Mio par classe ; `--max-file-mib` permet jusqu'à 64 Mio. Les choix et les fichiers trop volumineux sont consignés dans `reports/bracs_acquisition_plan.json`. Ce sont des choix d'acquisition technique, pas des exclusions scientifiques. Seuls train et val sont acceptés.

Pour acquérir les RoI complètes avec votre client FTP, prendre `/BRACS_RoI/latest_version/`, en conservant `train`, `val`, `test` et les classes, puis `/BRACS.xlsx`. Éviter le lien symbolique récursif `BRACS_RoI` et le dossier `previous_versions`. Conserver le test fermé pour la sélection des modèles. Le script pilote ne télécharge pas la cohorte complète ; les grandes WSI ne sont pas nécessaires au premier prototype sur RoI.

Le résumé mentionne 547 lames, 189 patients et 4 539 RoI. Le patient 67 apparaît dans train et val parmi les WSI ; aucun chevauchement n'apparaît parmi les lames avec RoI déclarées. Vérifier l'inventaire complet avant entraînement. La réussite de l'acquisition n'est pas une validation des partitions.

## Télécharger davantage de BCSS

Les [auteurs](https://github.com/PathologyDataScience/BCSS) publient les données sous CC0. Le script utilise leur serveur public et leurs liens de masques, sans clé. Les métadonnées sont fixées au commit enregistré dans la configuration.

```powershell
.\.venv\Scripts\python.exe scripts/acquire_bcss.py --limit 2
.\.venv\Scripts\python.exe scripts/audit_bcss.py
```

La limite peut être augmentée progressivement jusqu'à 151 régions. Le réglage actuel à 1 µm/pixel est réservé au contrôle d'acquisition et de tissu, pas à la validation des noyaux. Pour une autre résolution, copier la configuration et choisir un nouveau `raw_directory`, puis passer `--config` aux deux scripts. Les masques restent natifs : ne pas les superposer directement aux RGB réduits sans transformation explicite. Les codes 0 et 7 sont ignorés dans le dénominateur des fractions selon la configuration.

## NuCLS et BACH restent à acquérir

- **NuCLS** : [page single-rater officielle](https://sites.google.com/view/nucls/single-rater), rubrique **Corrected single-rater dataset**, vers le [dossier QC](https://drive.google.com/drive/folders/1eGlF9Dgu3WMEik4fqj0wJ13LKVufsfZ0). Conserver les RGB, annotations, masques disponibles, documentation et identifiants TCGA. Tous les noyaux n'ont pas nécessairement de contour. Données CC0. Stocker dans `data/raw/NuCLS/`.
- **BACH** : [dépôt officiel Zenodo](https://zenodo.org/records/3632035). Archive principale annoncée : environ 10,4 Go ; archive du test du challenge : environ 3 Go. Licence indiquée : CC BY-NC-ND, à prendre en compte pour l'utilisation et la diffusion. BACH reste fermé et n'est pas nécessaire à ces tests.

## Limites actuelles

Le pipeline complet — HoVer-Net validé, features, graphes, entraînement, sondes et interventions — reste à implémenter et valider. Aucun résultat de classification n'est disponible. La reproduction sur clone neuf et la lecture d'une vraie WSI restent également à vérifier ; des RoI PNG réelles ont été décodées.

Un en-tête de note a été ajouté au README BCSS téléchargé. Cette copie locale et son reçu initial sont préservés dans `reports/local_notes/`. Le document source exact a été récupéré comme `.txt`, puis ses empreintes ont été vérifiées. Les images et masques n'ont pas été modifiés.
