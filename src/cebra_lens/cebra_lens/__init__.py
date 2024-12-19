# example of structure so that you can directly use the functions get_layer_activations instead of having to do CEBRA_Lens.activations.get_layer_activations
from .activations import *
from .transform import *
from .quantification import *
from .plotting import *
from .utils_allen import *

# selects what files can be imported when doing from CEBRA_Lens import * --> keep env clean
# __all__ = ['get_layer_activations']
