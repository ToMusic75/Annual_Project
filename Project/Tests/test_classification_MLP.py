import matplotlib.pyplot as plt
from pprint import pprint
from time import sleep

# X = [
#     [0, 0],
#     [0, 1],
#     [0.7, 0.7],
#     [1, 1],
#     [0.5, 1],
#     [1, 0.3],
#     [0.7, 0.3],
#     [0.7, 0.5],
#     [0.4, 0.2],
# ]
# Y = [
#     -1,
#     -1,
#     -1,
#     1,
#     -1,
#     1,
#     1,
#     1,
#     1,
# ]
X = [
    [0, 0],  # Bleu
    [1, 0],  # Rouge
    [0, 1],  # Rouge
    [1, 1],  # Bleu
]
Y = [
   1,# [1],
   -1,# [-1],
   -1,# [-1],
   1,# [1]
]

model = rustymachine.Mlp(
    alpha=0.01,
    x_train=X,
    y_train=Y,
    npl=[2, 1]
)

model.init_w()
model.fit(
    epochs=5000,
)

# pprint(model.model)
print("\n" * 3)
for x, y in zip(X, Y):
    print(f"{x} => Expected {y}, get {model.predict(x)}")

# DISPLAY
XToPredict = [
    [i / 100.0, j / 100.0]
    for i in range(-200, 200) for j in range(-200, 200)
]
YPredicted = [
    model.predict(x) for x in XToPredict
]

XToPlotUnHappy = []
XToPlotHappy = []
for i, val in enumerate(YPredicted):
    if val[0] >= 0:
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

