import ctypes
import numpy as np
import matplotlib.pyplot as plt

USE_RUST = True

if __name__ == "__main__":

    path_to_dll = "../Lib/target/debug/lib.d"
    my_lib = ctypes.CDLL(path_to_dll)

    my_lib.linear_create_model.argtypes = [ctypes.c_int]
    my_lib.linear_create_model.restype = ctypes.c_void_p

    if not USE_RUST:
        my_lib.linear_dispose_model.argtypes = [ctypes.c_void_p]
    else:
        my_lib.linear_dispose_model.argtypes = [ctypes.c_void_p, ctypes.c_int]
    my_lib.linear_dispose_model.restype = None

    my_lib.linear_train_model_classification.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int
    ]
    my_lib.linear_train_model_classification.restype = None

    my_lib.linear_predict_model_classification.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int
    ]
    my_lib.linear_predict_model_classification.restype = ctypes.c_double

    # for _ in range(10000000000):
    model = my_lib.linear_create_model(ctypes.c_int(2))

    # my_lib.linear_dispose_model(model)

    # exit(0)

    X = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ], dtype='float64')
    Y = np.array([
        1,
        -1,
        -1
    ], dtype='float64')

    flattened_X = X.flatten()

    print("Before Training")
    for inputs_k in X:
        print(my_lib.linear_predict_model_classification(
            model,
            inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(inputs_k)
        ))

    my_lib.linear_train_model_classification(
        model,
        flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        X.shape[0],
        X.shape[1],
        0.01,
        1000
    )

    print("After Training")
    for inputs_k in X:
        print(my_lib.linear_predict_model_classification(
            model,
            inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(inputs_k)
        ))

    plt.scatter(X[0, 0], X[0, 1], color='blue')
    plt.scatter(X[1:3, 0], X[1:3, 1], color='red')
    plt.show()
    plt.clf()

    test_points = np.array([[i, j] for i in range(50) for j in range(50)], dtype='float64') / 50.0 * 2.0 + 1.0

    test_points_predicted = np.zeros(len(test_points))
    red_points = []
    blue_points = []
    for k, test_input_k in enumerate(test_points):
        predicted_value = my_lib.linear_predict_model_classification(
                model,
                test_input_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                len(test_input_k))
        if predicted_value == 1.0:
            blue_points.append(test_input_k)
        else:
            red_points.append(test_input_k)

    red_points = np.array(red_points)
    blue_points = np.array(blue_points)

    if len(red_points) > 0:
        plt.scatter(red_points[:, 0], red_points[:, 1], color='red', alpha=0.5, s=2)
    if len(blue_points) > 0:
        plt.scatter(blue_points[:, 0], blue_points[:, 1], color='blue', alpha=0.5, s=2)
    plt.scatter(X[0, 0], X[0, 1], color='blue', s=10)
    plt.scatter(X[1:3, 0], X[1:3, 1], color='red', s=10)
    plt.show()
    plt.clf()

    if not USE_RUST:
        my_lib.linear_dispose_model(model)
    else:
        my_lib.linear_dispose_model(model, len(X[0]))
