"""Real BACH acquisition is not implemented; synthetic substitution is forbidden."""


def prepare_bach_cohort(*args, **kwargs):
    raise NotImplementedError(
        "No real BACH acquisition implemented. Synthetic fixtures are in examples/toy_demo/generate_synthetic_graphs.py; they are not external validation."
    )


def generate_synthetic_bach_cell_graph(*args, **kwargs):
    raise RuntimeError(
        "Synthetic graphs must not be named BACH. Use the explicitly named toy fixture generator."
    )


if __name__ == "__main__":
    prepare_bach_cohort()
