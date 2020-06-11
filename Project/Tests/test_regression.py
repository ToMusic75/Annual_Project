import rustlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np


X = np.array([
    [1],
    [2]
])

Y = np.array([
    2,
    3
])


model = rustlib.Linear(
    x_train=X,
    y_train=Y,
    type_=rustlib.REGRESSION,
)
model.fit()

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
