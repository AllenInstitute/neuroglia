import neuroglia as ng
import pandas as pd
import pandas.util.testing as pdt


def test_spike_tablizer():
    compare = ng.nwb.SpikeTablizer().transform({'a':[1.1,2,3], 'b':[1,2.5,6]})
    base = pd.DataFrame({'neuron': ['b', 'a', 'a', 'b', 'a', 'b'],
                         'time': [1.0, 1.1, 2.0, 2.5, 3.0, 6.0]}).set_index('time')
    pdt.assert_frame_equal(base, compare)

if __name__ == "__main__":
    test_spike_tablizer()
