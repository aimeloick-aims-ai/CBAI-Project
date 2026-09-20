---
noteId: "001f89d0b3b711f180a4c967c0077061"
tags: []

---

# GCIA : experiences exploratoires du 19 septembre 2026

Ces experiences sont terminees localement, sans telechargement supplementaire.
Elles ne constituent pas la validation de toutes les affirmations de l'abstract.

## Protocole execute

- BRACS : 14 RoI train, 7 validation, partitions sans patient commun.
- BCSS : deux patients, une region annotee par patient ; 64 noeuds utilisables chacun.
- Graphes de patches RGB 8x8, aretes kNN symetriques dedupliquees, et non graphes cellulaires.
- GCN, GraphSAGE, GATv2 : deux couches, 60 epochs, cinq graines fixes.
- Sondes ridge sur entree, couche 1 et couche 2 des GNN geles.
- Cible des sondes : fraction tumorale du masque BCSS (code 1) parmi les pixels valides.
- Ajustement de chaque sonde sur un patient et evaluation sur l'autre, puis inversion.
- Controles : moyenne du patient d'ajustement et labels melanges sur ce patient.
- Perturbations : remplacement des attributs des huit noeuds les plus tumoraux par la moyenne,
  occultation des pixels tumoraux par la couleur moyenne et suppression des aretes incidentes
  aux noeuds selectionnes. Dix controles aleatoires de meme budget pour chaque intervention.
- Classe suivie : classe predite avant intervention, fixee pour toutes ses variantes.

Les poids (.pt), sorties par graine (.csv), empreintes des sources et protocole (summary.json)
sont conserves dans ce dossier. Les poids servent a reproduire ce pilote uniquement.

## Classification BRACS

| Architecture | Accuracy validation moyenne (5 graines) |
|---|---:|
| GCN | 34,29 % |
| GraphSAGE | 31,43 % |
| GATv2 | 31,43 % |

Ces chiffres concernent la nouvelle configuration a deux couches et grille 8x8.
Ils ne remplacent pas silencieusement le precedent pilote a une couche et grille 4x4.
La validation BRACS a deja ete examinee ; ce n'est pas un test final intact.

## Decodage de la fraction tumorale BCSS

Erreur quadratique moyenne, moyenne descriptive des deux directions et cinq graines.
Plus faible est meilleur. La baseline constante donne 0,1286.

| Architecture | Couche 1 | Couche 2 |
|---|---:|---:|
| GCN | 0,1006 | 0,2590 |
| GraphSAGE | 0,1158 | 0,2678 |
| GATv2 | 0,2911 | 0,2105 |

Les sondes de couche 1 GCN et GraphSAGE ont une erreur moyenne inferieure a la
baseline constante. Ce n'est pas une preuve statistique de generalisation.
La couche 2 est moins bonne que cette baseline pour les trois architectures.
Les erreurs des controles a labels melanges sont dans probes.csv.

## Sensibilite aux perturbations

Variation moyenne de probabilite de la classe initiale, pour les interventions ciblees,
moyennee sur deux patients, trois architectures et cinq graines :

| Niveau | Cible (points de pourcentage) | Controles aleatoires (points) |
|---|---:|---:|
| Attributs | +0,267 | +0,054 |
| Pixels | -6,181 | -0,469 |
| Aretes | -0,707 | -0,062 |

Les controles aleatoires sont mesures, pas fixes artificiellement ; toutes les observations
sont dans interventions.csv (control=-1 : cible ; control=0..9 : controles).
Une variation nulle ou faible ne prouve pas l'absence d'utilisation du concept.
Une variation importante ne prouve pas sa specificite.

## Limites et experiences non executees

Deux patients BCSS ne permettent pas une inference de population fiable. Les noeuds,
graines et perturbations ne sont pas des patients independants ; aucun intervalle de confiance
de population ni test de significativite n'est revendique.

Les masques sont redimensionnes au plus proche voisin pour suivre la grille RGB.
Les interventions ne produisent pas de nouvelles annotations histologiques valides.
Les controles apparient le nombre de pixels/noeuds/aretes modifies, pas leur structure
spatiale, leur amplitude ou tous les concepts non cibles. BCSS est hors domaine diagnostique
du classifieur BRACS ; ses probabilites ne sont pas des diagnostics BCSS.

Restent non executes : segmentation cellulaire validee sur NuCLS, concepts nucleaires
annotes, interventions histologiquement plausibles et validees independamment,
baselines XAI specialisees, audit confirme encodage/utilisation, evaluation BACH externe.
Leur absence empeche de presenter l'abstract comme une etude achevee.

## Reproduction

Depuis GCIA : `.\.venv\Scripts\python.exe scripts\run_real_xai.py`.
Les executions ecrasent les sorties de ce protocole dans reports/real_xai.
