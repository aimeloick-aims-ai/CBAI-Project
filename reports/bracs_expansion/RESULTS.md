# Extension BRACS terminee

- Nouvelles images : 49 (normal : 29, invasif : 20).
- Volume verifie de cette extension : 999,972,025 octets.
- Patients couverts : 47, dont 22 absents des RoI locales anterieures.
- Total local BRACS : 109 RoI de 72 patients.

Plafond autorise : 1 000 000 000 octets. Controle taille serveur, empreinte SHA256
locale et verification PNG pour chaque fichier. Les empreintes locales ne sont
pas des empreintes publiees par le fournisseur. Identifiants FTP non enregistres.

Selection : ordre hash fixe, parcours par patient, filtrage par budget,
sans choix selon les scores. La contrainte de taille peut introduire un biais.

Les classes concernent les regions, pas necessairement le diagnostic du patient.
Les nouvelles images de patients deja examines ne sont pas de nouveaux patients
de test independants. Aucun entrainement ni nouveau decoupage effectue ici.
Le pilote historique a 40 patients et ses predictions restent conserves.

Manifest : data/bracs_expansion_manifest.csv. Reçus : fichiers *.receipt.json.
Audit et comptages : completed_summary.json.
