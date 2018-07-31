import os
import hashlib
from sklearn.datasets.base import get_data_home


def get_neuroglia_data_home(data_home=None):
    # return get_data_home(data_home=data_home)
    return os.path.join(get_data_home(data_home=data_home),'neuroglia')


def _md5(filename):
    with open(filename) as f:
        data = f.read()
        md5_returned = hashlib.md5(data).hexdigest()
    return md5_returned
