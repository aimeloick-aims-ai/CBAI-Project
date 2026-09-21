"""Train-only linear concept probes."""

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_probe(embeddings, labels, seed, *, continuous=False, config=None):
    config = config or {"ridge_alpha": 1, "logistic_c": 1, "max_iter": 1000}
    estimator = (
        Ridge(alpha=config["ridge_alpha"])
        if continuous
        else LogisticRegression(
            C=config["logistic_c"], max_iter=config["max_iter"], random_state=seed
        )
    )
    probe = make_pipeline(StandardScaler(), estimator)
    return probe.fit(embeddings, labels)
