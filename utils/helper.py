import pickle
from utils.attribute_list import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    
def load_candidate(experiment_name, n_attr):
    experiment_name = experiment_name.rsplit('_', 1)[0]
    return attr_dict[experiment_name][:n_attr]