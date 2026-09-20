# Journal des changements

## 2026-09-18 — acquisition et exécution sur données réelles

- À la demande de l'utilisateur : accès FTP BRACS testé sans persister les identifiants ; résumé et 7 RoI train acquis. Sélection pilote lexicale avec plafond de 16 Mio après rencontre d'une RoI dépassant 64 Mio ; paramètres CLI et choix enregistrés dans le rapport d'acquisition.
- Deux régions BCSS et masques natifs acquis depuis les sources des auteurs, RGB à 1 µm/pixel. Configuration dans `configs/bcss_acquisition.yaml`. Codes 0 et 7 ignorés pour les fractions de surfaces du contrôle qualité.
- 18 fichiers et 50 066 957 octets vérifiés. Manifestes et reçus générés. Sept tests passent ; exécution du pilote via `scripts/run_data_smoke.ps1` réussie.
- Chevauchement patient 67 entre train/val dans les métadonnées WSI BRACS ; aucun parmi les lames avec RoI déclarées. Aucun entraînement lancé avant réconciliation complète.
- Copie annotée localement du README BCSS préservée dans `reports/local_notes/`; source exacte récupérée comme texte après détection par empreinte. Aucun masque ni image modifié.
- Phase active portée à l'acquisition pilote suivant la demande utilisateur. Clone neuf, cohortes complètes et pipeline expérimental restent non validés.

## 2026-09-18 — initialisation 0.1.0

- Lecture intégrale du DOCX ; extraction et empreinte SHA-256 conservées dans reports/.
- Phase 0 : matrice de 14 méthodes, références et contribution prospective révisée.
- Ajout des antécédents Amnesic Probing, LEACE, PICASSO, Cellina et MoPaDi.
- Abandon des revendications générales de priorité sur la distinction encodage/utilisation et sur les interventions conceptuelles en histopathologie ; décision dans configs/project.yaml.
- Ordre retenu : phase 0 puis phase 1 ; les indications contradictoires du document sur les premières commandes sont résolues en faveur des critères d'acceptation phase par phase.
- Préparation de l'environnement CPU isolé avec Python 3.11 pour éviter de dépendre de l'installation globale Python 3.14. Aucun entraînement de recherche lancé.
- Absence de données confirmée par l'utilisateur. BACH reste fermé. Registre préparatoire distinct d'une acquisition validée.
- Anomalie documentaire : le portail BRACS annonce 4 539 RoI, alors que le plan mentionne environ 4 537. Aucun de ces nombres n'est une observation locale.
- Environnement installé : 215 distributions compatibles selon `uv pip check`. TIAToolbox 2.1.3 fixé ; dépendances transitives résolues et verrouillées avec empreintes. Hooks pre-commit installés et contrôles Ruff réussis.
- Vérification locale : quatre tests CPU réussis. Fixture TIFF passée en RGBA compressé après échec de lecture de la version RGB non compressée. Aucun changement de données ou de protocole scientifique. Reproduction sur clone neuf et vraie WSI non vérifiées.
