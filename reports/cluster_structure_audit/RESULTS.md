# Audit structurel exploratoire des clusters de patches

Trois GCN gelés, 22 patients internes déjà examinés. Concepts nommés par IA, non validés. Aucun type cellulaire n'est attribué.

Suppression de nœuds : retrait du groupe et de ses arêtes incidentes. Suppression de frontière : retrait des arêtes reliant le groupe aux autres ; les nœuds restent identiques. Les témoins sont appariés selon protocol.json. Aucune intervention ne garantit la plausibilité biologique.

Différences en points de probabilité de la classe initialement prédite. IC bootstrap patients descriptifs, sans correction des dix comparaisons. Effectifs variables : les lignes ne forment pas un classement.

| Intervention | Cluster | Patients | Ciblé | Témoin | Différence [IC 95 %] |
|---|---:|---:|---:|---:|---:|
| node_removal | 0 | 16 | 57.94 | 7.46 | 50.48 [32.78, 66.76] |
| node_removal | 1 | 8 | -1.03 | 1.22 | -2.25 [-11.56, 4.27] |
| node_removal | 2 | 13 | -2.11 | 0.84 | -2.94 [-8.45, 3.19] |
| node_removal | 3 | 6 | 9.08 | 0.18 | 8.90 [-0.22, 18.17] |
| node_removal | 4 | 18 | -3.53 | -0.13 | -3.40 [-6.33, -1.22] |
| boundary_removal | 0 | 16 | 0.81 | 0.19 | 0.62 [0.05, 1.29] |
| boundary_removal | 1 | 8 | 1.28 | 0.34 | 0.94 [-0.19, 2.73] |
| boundary_removal | 2 | 13 | 0.92 | 0.01 | 0.91 [0.07, 1.81] |
| boundary_removal | 3 | 6 | 1.36 | 0.29 | 1.07 [-0.53, 2.88] |
| boundary_removal | 4 | 18 | 0.32 | 0.23 | 0.09 [-0.28, 0.55] |
