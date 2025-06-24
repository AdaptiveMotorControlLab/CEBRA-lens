# example of structure so that you can directly use the functions get_layer_activations instead of having to do CEBRA_Lens.activations.get_layer_activations
from .activations import *
from .quantification.cka_metric import *
from .quantification.decoder import *
from .quantification.distance import *
from .quantification.rdm_metric import *
from .quantification.tsne import *
from .utils import *
from .utils_allen import *
from .utils_hpc import *
from .utils_plot import *
from .utils_wrapper import *

# selects what files can be imported when doing from CEBRA_Lens import * --> keep env clean
# __all__ = ['get_layer_activations']
