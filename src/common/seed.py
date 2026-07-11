import numpy as np
import random


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)