# Reference experiment before GCIA

This project now includes a runnable adaptation of GNNExplainer to the existing
PCam-encoder/patch-GCN pipeline. It is not an exact reproduction of a published
cell-graph breast-cancer benchmark.

GNNExplainer learns masks over graph structure and input features to explain a
trained GNN's predictions. We use the implementation provided by PyTorch Geometric,
which supports the regression task used here. Sources:
[Ying et al., GNNExplainer (2019)](https://arxiv.org/abs/1903.03894) and
[PyG documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.explain.algorithm.GNNExplainer.html).

Our task is prediction of the mean tumor fraction on valid BCSS patches, not a
diagnostic class. The pretrained PCam encoder and trained two-layer GCN remain
frozen. GNNExplainer optimizes node and edge masks jointly for 100 epochs, using
the installed library's default regularization and learning rate 0.01. We rank
nodes by their learned scalar masks and replace their normalized features with
zero (the training mean) at fixed 5%, 10%, and 20% budgets. Twenty random selections
of the same size provide a baseline at every budget. Original topology and output
aggregation are retained. Edge masks are saved but not separately evaluated.

All eight locally available patients and three model seeds are included, each
with a checkpoint that excludes that patient from model training. The two latest
patients have already been examined in the encoder study: this analysis does not
constitute a new untouched test. Signed and absolute prediction changes are saved.
Large absolute changes measure perturbation sensitivity, not necessarily a
plausible or concept-specific explanation. Joint-mask explanations evaluated by
node-feature deletion are also only a partial assessment of GNNExplainer.

This reference establishes an executable comparison for future GCIA experiments.
It does not yet establish an advantage of GCIA. That comparison will require
concept-specific interventions, independent validity assessment, and a common
evaluation protocol. Current outputs are in reports/gnnexplainer_baseline.
