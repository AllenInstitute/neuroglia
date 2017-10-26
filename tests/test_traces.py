import numpy as np
import numpy.testing as npt

import pandas as pd
import pandas.testing as pdt

from neuroglia.trace import EdgeDetector, WhenTrueFinder, Binarizer

X = np.array(
    [[0, 0, 1],
     [1, 1, 0],
     [0, 1, 1]
    ])

XDF = pd.DataFrame(
    data=X,
    index=[0.1,0.2,0.3],
    columns=['n1','n2','n3'],
)

RISING = np.array(
    [[0, 0, 0],
     [1, 1, 0],
     [0, 0, 1],
    ]
)

DF = pd.DataFrame(
    data=RISING,
    index=[0.1,0.2,0.3],
    columns=['n1','n2','n3'],
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

    detector = EdgeDetector()
    output = detector.fit_transform(XDF)
    npt.assert_array_equal(output.values,RISING)
    print(output)
    print(DF)
    pdt.assert_frame_equal(output,DF)


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

X2 = np.array(
    [[0, 0, 1],
     [1, 2, 0],
     [0, 1, 10]
    ])

X2DF = pd.DataFrame(
    data=X,
    index=[0.1,0.2,0.3],
    columns=['n1','n2','n3'],
)

def test_Binarizer():
    binarizer = Binarizer()
    output = binarizer.fit_transform(X2)
    npt.assert_array_equal(output,X)

    binarizer = Binarizer()
    output = binarizer.fit_transform(X2DF)
    pdt.assert_frame_equal(output,XDF)
