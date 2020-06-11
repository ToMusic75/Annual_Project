import rustlib
import matplotlib.pyplot as plt
from pprint import pprint
from time import sleep
import numpy as np

X = np.array([
    [1],
    [2]
])

Y = np.array([
    2,
    3
])

model = rustlib.Mlp(
    alpha=0.001,
    x_train=X,
    y_train=Y,
    npl=[1],
    type_=rustlib.REGRESSION
)
model.fit(
    epochs=10000,
)

for x, y in zip(X.tolist(), Y.tolist()):
    plt.scatter(
        x,
        y,
        color='red'
    )

for i in range(300):
    i /= 100
    plt.scatter(i, model.predict([i]), marker=',', color='b', s=1)

plt.show()
plt.clf()
