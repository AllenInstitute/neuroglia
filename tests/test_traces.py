import numpy as np
import numpy.testing as npt

import pandas as pd
import pandas.testing as pdt

from neuroglia.trace import EdgeDetector, WhenTrueFinder

X = np.array(
    [[0, 0, 1],
     [1, 1, 0],
     [0, 1, 1]
    ])

RISING = np.array(
    [[0, 0, 0],
     [1, 1, 0],
     [0, 0, 1],
    ]
)

FALLING = np.array(
    [[0, 0, 0],
     [0, 0, 1],
     [1, 0, 0],
    ]
)

def test_EdgeDetector():
    detector = EdgeDetector()
    output = detector.fit_transform(X)
    npt.assert_array_equal(output,RISING)

    detector = EdgeDetector(falling=True)
    output = detector.fit_transform(X)
    npt.assert_array_equal(output,FALLING)

DF = pd.DataFrame(
    data=RISING,
    index=[0.1,0.2,0.3],
    columns=['n1','n2','n3'],
)

WHENTRUE = pd.DataFrame(dict(
    neuron=['n1','n2','n3'],
    time=[0.2,0.2,0.3],
)).set_index('time')

def test_WhenTrueFinder():
    finder = WhenTrueFinder()
    output = finder.fit_transform(DF)
    output = output.sort_values(['time','neuron']).set_index('time')
    print(output)
    print(DF)
    pdt.assert_frame_equal(output,WHENTRUE)
