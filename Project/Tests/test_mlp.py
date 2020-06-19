import ctypes
import numpy as np
import matplotlib.pyplot as plt
from sys import platform

USE_RUST = True

if __name__ == "__main__":
    path_to_dll = "../Lib/target/debug/liblib.dylib"
    if platform == "linux" or platform == "linux2":
        #To do for linux
        print("to do for linux")
    elif platform == "darwin":
        path_to_dll = "../Lib/target/debug/liblib.dylib"
    elif platform == "win32":
        path_to_dll = "../Lib/target/debug/liblib.so"

    my_lib = ctypes.CDLL(path_to_dll)

    my_lib.mlp_create_model.argtypes = [ctypes.c_int]
    my_lib.mlp_create_model.restype = ctypes.c_void_p

    if not USE_RUST:
        my_lib.mlp_dispose_model.argtypes = [ctypes.c_void_p]
    else:
        my_lib.mlp_dispose_model.argtypes = [ctypes.c_void_p, ctypes.c_int]
    my_lib.linear_dispose_model.restype = None

    my_lib.mlp_train_model_classification.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int
    ]
    my_lib.mlp_train_model_classification.restype = None

    my_lib.mlp_predict_classification.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int
    ]
    my_lib.mlp_predict_classification.restype = ctypes.c_double

    my_lib.linear_train_model_regression.argtypes = [ctypes.c_void_p,
                                               ctypes.POINTER(ctypes.c_double),
                                               ctypes.POINTER(ctypes.c_double),
                                               ctypes.c_int,
                                               ctypes.c_int]

    my_lib.linear_train_regression.restype = None

    my_lib.linear_predict_regression.argtypes = [ctypes.c_void_p,
                                                 ctypes.POINTER(ctypes.c_double),
                                                 ctypes.c_int]
    my_lib.linear_predict_regression.restype = ctypes.c_double

    # for _ in range(10000000000):
    model = my_lib.mlp_create_model(ctypes.c_int(2))

    # my_lib.linear_dispose_model(model)

    # exit(0)

    x_train = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ]

    y_train = [
        [-1],
        [1],
        [1],
        [-1],
    ]

    my_mlp = my_lib.mlp_create([2, 2, 1])

    my_lib.mlp_train_classification(my_mlp, x_train, y_train, 100000, 0.1)

    print(my_lib.mlp_predict_classification(my_mlp, x_train[0]))
    print(my_lib.mlp_predict_classification(my_mlp, x_train[1]))
    print(my_lib.mlp_predict_classification(my_mlp, x_train[2]))
    print(my_lib.mlp_predict_classification(my_mlp, x_train[3]))

    y_train = [
        [-3],
        [2],
        [8],
        [-5],
    ]

    my_mlp = my_lib.mlp_create([2, 5, 1])

    my_lib.mlp_train_regression(my_mlp, x_train, y_train, 100000, 0.1)

    print(my_lib.mlp_predict_regression(my_mlp, x_train[0]))
    print(my_lib.mlp_predict_regression(my_mlp, x_train[1]))
    print(my_lib.mlp_predict_regression(my_mlp, x_train[2]))
    print(my_lib.mlp_predict_regression(my_mlp, x_train[3]))

    y_train = [
        [1, -1, -1],
        [-1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ]

    my_mlp = my_lib.mlp_create([2, 5, 3])

    my_lib.mlp_train_classification(my_mlp, x_train, y_train, 100000, 0.1)

    print(my_lib.mlp_predict_classification(my_mlp, x_train[0]))
    print(my_lib.mlp_predict_classification(my_mlp, x_train[1]))
    print(my_lib.mlp_predict_classification(my_mlp, x_train[2]))
    print(my_lib.mlp_predict_classification(my_mlp, x_train[3]))

    y_train = [
        [1, -1, 3],
        [2, 2, 4],
        [8, -1.5, -1],
        [-5, -1, 2],
    ]

    my_mlp = my_lib.mlp_create([2, 5, 3])

    my_lib.mlp_train_regression(my_mlp, x_train, y_train, 100000, 0.1)

    print(my_lib.mlp_predict_regression(my_mlp, x_train[0]))
    print(my_lib.mlp_predict_regression(my_mlp, x_train[1]))
    print(my_lib.mlp_predict_regression(my_mlp, x_train[2]))
    print(my_lib.mlp_predict_regression(my_mlp, x_train[3]))

    exit(0)
