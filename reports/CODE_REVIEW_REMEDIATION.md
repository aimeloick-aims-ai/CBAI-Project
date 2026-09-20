---
noteId: "1416e2c0b53e11f1b1f059eca98e70d3"
tags: []

---

# Consolidation après les deux revues de code

## Corrections effectuées

- La prétendue acquisition BACH échoue explicitement sans produire de données. Le générateur label-codé reste un outil de test nommé synthétique.
- Les scripts master et cell-graph demo ne peuvent plus être utilisés comme expériences sur modèles entraînés. Sources historiques conservées sous examples/toy_demo avec extension .disabled.txt.
- CellGNNExplainer délègue à GNNExplainer officiel PyG, qui applique les masques d'arêtes. Les paramètres du modèle sont gelés pendant l'explication puis leur état est restauré. La perte finale, non exposée par PyG, vaut None plutôt qu'un chiffre inventé.
- Le formulaire de revue montrant prédictions et scores préremplis est bloqué. Aucun nouveau dossier favorable artificiel ne peut en sortir. Le lot d'images aveugle existant reste utilisable pour la découverte ; il ne remplace pas une revue des interventions sur images.
- NullProbeEvaluator refuse l'évaluation par défaut sur l'apprentissage, exige des identifiants patients disjoints et une seule représentation par patient. Les permutations portent donc sur les patients. Les petits nombres de permutations des tests logiciels ne sont pas un protocole scientifique confirmatoire.
- La stabilité des probes conserve le signe du cosinus et refuse de conclure avec un seul probe.
- Le rewiring arbitraire est désactivé plutôt que présenté comme une intervention spatiale valide. Un nouvel algorithme contraint reste à implémenter et valider.
- STATUS.md fournit l'inventaire actualisé ; les anciens bilans sont marqués historiques, le verrou externe non réalisé est corrigé dans project.yaml.

## Résultats et provenance préservés

Les checkpoints, chiffres d'audits et anciennes archives du papier ne sont pas recalculés ou réécrits. L'autorisation antérieure de noter la confirmation du spécialiste reste respectée : elle est une déclaration de l'utilisateur, pas une attestation indépendante inventée. La suggestion de l'effacer n'est pas suivie.

L'autorisation de chevauchement cible/témoin des contrôles d'arêtes historiques est explicite. Modifier ce protocole après résultats exigerait un nouvel audit séparé. Une suppression de tous les éléments d'une strate peut ne laisser aucun témoin non-cible apparié ; une nouvelle étude devra signaler ces cas plutôt que relâcher silencieusement les contraintes.

## Restant

Les propositions de benchmark à vérité connue, DeepSets, réseau multicouche, typage/MPP, TCAV/LEACE, contrôle de sélectivité indépendant, nouvelle cohorte et véritable BACH constituent une étude supplémentaire, non accomplie par cette correction logicielle. Les revendications bibliographiques des textes transmis ne sont pas reprises comme faits vérifiés. Aucune réussite clinique nouvelle n'est revendiquée.
