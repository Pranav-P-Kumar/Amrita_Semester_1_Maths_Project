import numpy as np
from src.connectivity.static_fc import calculate_static_fc
from src.connectivity.dynamic_fc import sliding_window_fc

def test_static_properties():
    C = calculate_static_fc(np.random.default_rng(0).normal(size=(100,4)))
    assert C.shape == (4,4) and np.allclose(C,C.T) and np.allclose(np.diag(C),1)

def test_dynamic_shape():
    mats, meta = sliding_window_fc(np.random.default_rng(1).normal(size=(300,10)))
    assert mats.shape == (26,10,10) and len(meta) == 26
