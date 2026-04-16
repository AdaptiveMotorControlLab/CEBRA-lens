import inspect

from cebra.integrations.sklearn.cebra import CEBRA
from cebra.solver.base import Solver


def transform(model, data, label, **transform_kwargs):
    if isinstance(model, Solver):
        transform_signature = inspect.signature(model.transform)
        if "labels" in transform_signature.parameters:
            embedding = model.transform(inputs=data, labels=label, **transform_kwargs)
        else:
            embedding = model.transform(inputs=data, **transform_kwargs)
    elif isinstance(model, CEBRA):
        embedding = model.transform(inputs=data, **transform_kwargs)
    else:
        raise TypeError(
            "Model must be an instance of cebra.solver.base.Solver "
            "or cebra.integrations.sklearn.cebra.CEBRA, "
            f"got {type(model)} instead.",
        )
    return embedding
