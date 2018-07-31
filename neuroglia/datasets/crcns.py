# from os.path import exists
# from os import makedirs, remove
import os
import tarfile
from collections import namedtuple
import requests
import pandas as pd
import numpy as np
from sklearn.datasets.base import _sha256, _pkl_filepath
from sklearn.utils import Bunch
from .utils import get_neuroglia_data_home

try:
    from itertools import izip as zip
except ImportError:  # must be python3
    pass

URL = 'https://portal.nersc.gov/project/crcns/download/index.php'


def get_environ_username():
    return os.environ['CRCNS_USER']


def get_environ_password():
    return os.environ['CRCNS_PASSWORD']


Payload = namedtuple('Payload',['username','password','fn','submit'])


def _create_payload(username,password,path,filename):
    datafile = "{}/{}".format(path,filename)
    return dict(
        username=username,
        password=password,
        fn=datafile,
        submit='Login'
    )


def _create_local_filename(dest,datafile):
    if dest is None:
        dest = os.cwd()
    return os.path.join(
        dest,
        datafile.split('/')[-1],
    )


def crcns_retrieve(request_payload,local_filename):
    with requests.Session() as s:
        r = s.post(URL,data=request_payload,stream=True)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    return local_filename


def _fetch_crcns_datafile(crcns,local_filename=None,username=None,password=None,chunk_size=1024):

    if local_filename is None:
        local_filename = crcns.filename

    if os.path.exists(local_filename):
        checksum = _sha256(local_filename)
        if crcns.checksum == checksum:
            return local_filename

    if username is None:
        username = get_environ_username()
    if password is None:
        password = get_environ_password()

    request_payload = _create_payload(
        username,
        password,
        crcns.path,
        crcns.filename,
    )

    crcns_retrieve(request_payload,local_filename)

    checksum = _sha256(local_filename)

    if crcns.checksum != checksum:
        raise IOError("{} has an SHA256 checksum ({}) "
                      "differing from expected ({}), "
                      "file may be corrupted.".format(local_filename, checksum,
                                                      crcns.checksum))
    return local_filename

CRCNSFileMetadata = namedtuple(
    'CRCNSFileMetadata',
    ['filename', 'path', 'checksum'],
)

def read_spikes_from_tar(f):

    SPIKES_HZ = 20000

    timestamp_files = (
        'crcns/hc2/ec014.333/ec014.333.res.1',
        'crcns/hc2/ec014.333/ec014.333.res.2',
        'crcns/hc2/ec014.333/ec014.333.res.3',
        'crcns/hc2/ec014.333/ec014.333.res.4',
        'crcns/hc2/ec014.333/ec014.333.res.5',
        'crcns/hc2/ec014.333/ec014.333.res.6',
        'crcns/hc2/ec014.333/ec014.333.res.7',
        'crcns/hc2/ec014.333/ec014.333.res.8',
    )

    cluster_files = (
        'crcns/hc2/ec014.333/ec014.333.clu.1',
        'crcns/hc2/ec014.333/ec014.333.clu.2',
        'crcns/hc2/ec014.333/ec014.333.clu.3',
        'crcns/hc2/ec014.333/ec014.333.clu.4',
        'crcns/hc2/ec014.333/ec014.333.clu.5',
        'crcns/hc2/ec014.333/ec014.333.clu.6',
        'crcns/hc2/ec014.333/ec014.333.clu.7',
        'crcns/hc2/ec014.333/ec014.333.clu.8',
    )

    spikes = []

    for timestamps,clusters in zip(timestamp_files,cluster_files):
        shank = int(timestamps[-1])
        #print timestamps,clusters
        time = 0

        ts = f.extractfile(timestamps)
        clu = f.extractfile(clusters)
        for frame,cluster in zip(ts.readlines(),clu.readlines()):
            if int(cluster)>1:
                spike = dict(
                    time=float(frame) / SPIKES_HZ,
                    neuron='{}-{:02d}'.format(shank,int(cluster)),
#                     shank=shank,
                )
                spikes.append(spike)

    spikes = pd.DataFrame(spikes)
    return spikes

def read_location_from_tar(f):

    LOCATION_HZ = 39.06

    location_file = 'crcns/hc2/ec014.333/ec014.333.whl'
    loc = pd.read_csv(
        f.extractfile(location_file),
        sep='\t',
        header=0,
        names=['x','y','x2','y2'],
    )
    loc = loc.replace(-1.0,np.nan)
    loc['time'] = loc.index / LOCATION_HZ
    loc = loc.dropna()
    return loc




def load_hc2(tar_path):

    with tarfile.open(mode="r:gz", name=tar_path) as f:
        spikes = read_spikes_from_tar(f)
        location = read_location_from_tar(f)

    # truncate neuronal data to time when mouse is exploring
    min_time = location['time'].min()
    max_time = location['time'].max()

    spikes = spikes[
        (spikes['time'] >= min_time)
        & (spikes['time'] <= max_time)
    ]

    # set approx center of arena to zero in x & y
    x0 = np.mean([location['x2'].max(),location['x2'].min()])
    y0 = np.mean([location['y2'].max(),location['y2'].min()])

    location['x'] -= x0
    location['x2'] -= x0
    location['y'] -= y0
    location['y2'] -= y0

    return spikes, location


def fetch_rat_hippocampus_foraging(data_home=None,username=None,password=None,download_if_missing=True):
    """Loader for experiment ec014.333 from the HC-2 dataset on crcns.org

    More info on this dataset: https://crcns.org/data-sets/hc/hc-2/about-hc-2

    To download this data, you must have a CRCNS account. Request an account
    at https://crcns.org/request-account/

    Warning! The first time you run this function, it will download a 3.3GB file.

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.
    username : optional, default: None
        CRCNS username. All CRCNS datasets need a username to login. If `None`
        (default), the `CRCNS_USERNAME` environment variable is used.
    password : optional, default: None
        CRCNS username & password. All CRCNS datasets need a username to login. If `None`
        (default), the `CRCNS_USERNAME` environment variable is used.
    download_if_missing : optional, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    dataset : dict-like object with the following attributes:
    dataset.spikes : dataframe, shape [20640, 2]
        Each row is a single spike at `time` elicited from neuron `neuron`
    dataset.location : dataframe, shape (20640,)
        Each row is a sample of the rat's position, with the location of the
        head designated by (x,y) and the location of the back designated by
        (x2, y2)

    Notes
    ------
    This dataset consists of 58 simultaneously recorded neurons from the rat
    hippocampus along with coordinates of its position while it forages in an
    open arena (180cm x 180cm) for 92 minutes.

    References
    ----------

    Mizuseki K, Sirota A, Pastalkova E, Buzsaki G. (2009): Multi-unit recordings
    from the rat hippocampus made during open field foraging
    http://dx.doi.org/10.6080/K0Z60KZ9

    """


    data_home = get_neuroglia_data_home(data_home=data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    # check if local files exist. if so, load, otherwise download raw

    spikes_path = _pkl_filepath(data_home, 'crcns_hc2_spikes.pkl')
    location_path = _pkl_filepath(data_home, 'crcns_hc2_location.pkl')


    if (not os.path.exists(spikes_path)) or (not os.path.exists(location_path)):
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        tar_path = os.path.join(data_home,'crcns_hc2.tar.gz')

        crcns = CRCNSFileMetadata(
            path = "hc-2/ec014.333",
            filename = "ec014.333.tar.gz",
            checksum = '819d9060bcdd439a2024ee44cfb3e7be45056632af052e524e0e23f139c6a260',
        )

        local_filename = _fetch_crcns_datafile(
            crcns=crcns,
            local_filename=tar_path,
            username=username,
            password=password,
        )

        spikes, location = load_hc2(tar_path)

        spikes.to_pickle(spikes_path)
        location.to_pickle(location_path)

        os.remove(tar_path)
    else:
        spikes = pd.read_pickle(spikes_path)
        location = pd.read_pickle(location_path)

    return Bunch(
        spikes=spikes,
        location=location
    )
