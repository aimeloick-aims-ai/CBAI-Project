# Acquisition — prérequis non satisfaits

> Note historique de la préparation initiale. L'accès BRACS est maintenant débloqué et un pilote BRACS/BCSS est téléchargé. Consulter le guide actuel [data_quickstart.md](data_quickstart.md) et les manifestes pour l'état réel.

L'utilisateur confirme ne pas disposer des données le 18 septembre 2026. Aucun fichier clinique n'a été téléchargé ou traité. Les champs observés du registre restent nuls. Aucun checksums.csv clinique ni rapport d'intégrité positif ne peut encore être produit.

1. **BRACS** : s'inscrire sur le [portail des producteurs](https://www.bracs.icar.cnr.it/), consulter les conditions de la version obtenue et obtenir l'accès. Conserver les métadonnées patient/lame/RoI et les partitions proposées. Ne jamais communiquer de mot de passe dans le projet.
2. **BACH** : consulter le [site du challenge](https://iciar2018-challenge.grand-challenge.org/). Le navigateur de recherche a reçu une erreur 403 ; ne pas en déduire une indisponibilité générale. Vérifier la licence du dépôt officiel lié et les identifiants de regroupement. Garder le test externe fermé.
3. **NuCLS** : suivre les liens de données du [dépôt des auteurs](https://github.com/PathologyDataScience/NuCLS). Le README distingue la licence CC0 des données de la licence MIT du code. Choisir explicitement le sous-ensemble et les annotations exploitables ; conserver les identifiants TCGA.
4. **BCSS** : suivre le téléchargement du [dépôt des auteurs](https://github.com/PathologyDataScience/BCSS). Données CC0, code MIT. Conserver RGB, masques, table de classes et résolution. Le code zéro est à ignorer dans la perte, pas à apprendre comme classe de tissu.

Les conditions CC0 de NuCLS et BCSS sont identifiées ; elles ne nécessitent pas d'inscription institutionnelle sur BRACS. Pour la cohorte principale, l'inscription doit contenir les coordonnées et l'affiliation exactes du demandeur, puis les conditions du téléchargement doivent être vérifiées. Ces informations et cet accès ne sont pas disponibles ici. Une fois les données obtenues, leurs chemins permettront d'enregistrer les versions et de calculer les empreintes.

L'entraînement complet exigera une mesure des besoins mémoire et temps sur un petit lot. Un GPU n'est pas nécessaire aux tests logiciels CPU ; la faisabilité du traitement de cohorte n'est pas encore établie.

## Mise a jour

Le pilote peut aussi verifier le cablage des graphes et modeles avec :

```powershell
.\.venv\Scripts\python.exe scripts/run_phase2_smoke.py
```

Le rapport correspondant est `reports/phase2_smoke.json`. Il ne debloque pas les cohortes completes, NuCLS ou BACH.
