import rustlib
import matplotlib.pyplot as plt

# import numpy as np
#
# print(isinstance(np.array([1]), np.ndarray))
#
# exit()
def draw_2d(model, X, Y):
    max_x = max(x[0] for x in X)
    max_y = max(x[1] for x in X)

    for x, y in zip(X, Y):
        print(f"{x} => Expected {y}, get {model.predict(x)}")

    # DISPLAY
    XToPredict = [
        [i / 100.0, j / 100.0]
        for i in range(0, int(max_x) * 100 + 1) for j in range(0, int(max_y) * 100 + 1)
    ]
    YPredicted = [
        model.predict(x) for x in XToPredict
    ]

    XToPlotUnHappy = []
    XToPlotHappy = []
    for i, val in enumerate(YPredicted):
        if val >= 0:
            XToPlotHappy.append(XToPredict[i])
        else:
            XToPlotUnHappy.append(XToPredict[i])

    def get(i, l):
        return [z[i] for z in l]

    plt.scatter(
        get(0, XToPlotUnHappy),
        get(1, XToPlotUnHappy),
        color="yellow"
    )
    plt.scatter(
        get(0, XToPlotHappy),
        get(1, XToPlotHappy),
        color="violet"
    )
    plt.scatter(
        get(0, [x for j, x in enumerate(X) if Y[j] == -1]),
        get(1, [x for j, x in enumerate(X) if Y[j] == -1]),
        color='red'
    )
    plt.scatter(
        get(0, [x for j, x in enumerate(X) if Y[j] != -1]),
        get(1, [x for j, x in enumerate(X) if Y[j] != -1]),
        color='blue'
    )
    plt.show()
    plt.clf()


X = [
    [0, 0],
    [0, 1],
    [0.7, 0.7],
    # [1, 3],
    [0.5, 1],
    [1, 0.3],
    [0.7, 0.3],
    [0.7, 0.5],
    [0.4, 0.2],
]
Y = [
    -1,
    -1,
    -1,
    # 1,
    -1,
    1,
    1,
    1,
    1,
]

model = rustlib.Linear(
    alpha=0.1,
    x_train=X,
    y_train=Y,
)

model.fit(
    epochs=1000,
)

draw_2d(model, X, Y)
