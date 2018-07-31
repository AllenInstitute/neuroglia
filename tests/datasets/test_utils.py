from neuroglia.datasets.utils import get_neuroglia_data_home
import os

HOME = os.path.join(
    os.path.expanduser("~"),
    'data',
)

def test_get_neuroglia_data_home():
    EXPECTED = os.path.join(HOME,'neuroglia')
    data_home = get_neuroglia_data_home(data_home=HOME)
    assert data_home==EXPECTED
