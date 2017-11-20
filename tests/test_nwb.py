import neuroglia as ng
import pandas as pd
import pandas.util.testing as pdt


def test_spike_tablizer():
    spike_dict = {
        'a': [1.1,2,3],
        'b': [1,2.5,6],
        }

    compare_list = [
        ng.nwb.SpikeTablizer().transform(spike_dict).reset_index(drop=True),
        ng.nwb.SpikeTablizer().fit_transform(spike_dict).reset_index(drop=True),
        ]

    base = pd.DataFrame({'neuron': ['b', 'a', 'a', 'b', 'a', 'b'],
                         'time': [1.0, 1.1, 2.0, 2.5, 3.0, 6.0]})
    for compare in compare_list:
        print(base)
        print(compare)
        pdt.assert_frame_equal(base, compare)

if __name__ == "__main__":
    test_spike_tablizer()
