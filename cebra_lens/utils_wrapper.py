import cebra
from cebra.integrations.sklearn.cebra import CEBRA 
from cebra.solver.base import Solver


def transform(model, data, label, **transform_kwargs):
    if isinstance(model, Solver):
        print(data[0].shape)
        print(label[0].shape)
        embedding = model.transform(data, label, **transform_kwargs)
    elif isinstance(model, CEBRA):
        embedding = model.transform(data, **transform_kwargs)
    else:
        raise TypeError(
            "Model must be an instance of cebra.solver.UnifiedSolver",
            f"or cebra.integrations.sklearn.cebra.CEBRA, got {type(model)} instead.",
        )
    return embedding


# def load_model():

#         loaded_model = cebra.CEBRA.load(
#             file, backend=backend, map_location=torch.device("cpu")
#         ).to("cpu")

#     else:
#         checkpoint = torch.load(file, map_location=device, weights_only = False)
#         solver.load_state_dict(checkpoint, strict=True)
