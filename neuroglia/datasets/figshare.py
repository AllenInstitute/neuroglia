import os
import numpy as np
import pandas as pd
from collections import namedtuple
from scipy.io import loadmat
from six.moves.urllib.request import urlretrieve
from sklearn.datasets.base import _pkl_filepath
from sklearn.utils import Bunch
from .utils import _md5, get_neuroglia_data_home


FigshareFileMetadata = namedtuple(
    'FigshareFileMetadata',
    ['url', 'filename', 'md5'],
)


def fetch_10k_neurons(data_home=None,download_if_missing=True):

    data_home = get_neuroglia_data_home(data_home=data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    # check if local files exist. if so, load, otherwise download raw

    calcium_path = _pkl_filepath(data_home, 'tenk_neurons.pkl')
    behavior_path = _pkl_filepath(data_home, 'tenk_behavior.pkl')

    if (not os.path.exists(calcium_path)) or (not os.path.exists(behavior_path)):
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        mat_path = os.path.join(data_home, 'figshare_10k.mat')

        figshare = FigshareFileMetadata(
            url="https://ndownloader.figshare.com/files/11152646",
            filename="spont_M161025_MP030_2016-11-20.mat",
            md5="54ffdc817e82710952dbee2655d4d1d2",
        )

        _fetch_figshare_datafile(
            figshare=figshare,
            local_filename=mat_path,
        )

        calcium, behavior = load_10k(mat_path)

        calcium.to_pickle(calcium_path)
        behavior.to_pickle(behavior_path)

        os.remove(mat_path)
    else:
        calcium = pd.read_pickle(calcium_path)
        behavior = pd.read_pickle(behavior_path)

    return Bunch(
        calcium=calcium,
        behavior=behavior,
    )


def load_10k(mat_path):

    # read in the data and return neuroglia-friendly dataframes
    data = loadmat(mat_path)

    tres = 0.25
    time = np.arange(0, data['Fsp'].shape[1] * tres, tres)

    calcium_events = pd.DataFrame(
        data['Fsp'].T,
        index=time,
        columns=['neuron_{}'.format(ii) for ii in range(data['Fsp'].shape[0])],
    )

    behavior = pd.DataFrame(
        {
            'speed': np.squeeze(data['beh']['runSpeed'][0][0]),
            'pupil_size': np.squeeze(data['beh']['pupil'][0][0]['area'][0][0]),
        },
        index=time,
    )

    return calcium_events, behavior


def _fetch_figshare_datafile(figshare, local_filename=None, chunk_size=1024):

    if local_filename is None:
        local_filename = figshare.filename

    if os.path.exists(local_filename):
        checksum = _md5(local_filename)
        if figshare.checksum == checksum:
            return local_filename

    urlretrieve(figshare.url, local_filename)

    checksum = _md5(local_filename)

    if figshare.checksum != checksum:
        raise IOError("{} has an md5 checksum ({}) "
                      "differing from expected ({}), "
                      "file may be corrupted.".format(local_filename, checksum,
                                                      figshare.checksum))
    return local_filename
