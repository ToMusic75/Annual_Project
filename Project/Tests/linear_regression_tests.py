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
        #to do for windows
        print("to do for windows")

    my_lib = ctypes.CDLL(path_to_dll)

    my_lib.linear_create_model.argtypes = [ctypes.c_int]
    my_lib.linear_create_model.restype = ctypes.c_void_p

    if not USE_RUST:
        my_lib.linear_dispose_model.argtypes = [ctypes.c_void_p]
    else:
        my_lib.linear_dispose_model.argtypes = [ctypes.c_void_p, ctypes.c_int]

    my_lib.linear_dispose_model.restype = None

    my_lib.linear_train_regression.argtypes = [ctypes.c_void_p,
                                               ctypes.POINTER(ctypes.c_double),
                                               ctypes.POINTER(ctypes.c_double),
                                               ctypes.c_int,
                                               ctypes.c_int]

    my_lib.linear_train_regression.restype = None

    my_lib.linear_predict_regression.argtypes = [ctypes.c_void_p,
                                                 ctypes.POINTER(ctypes.c_double),
                                                 ctypes.c_int]
    my_lib.linear_predict_regression.restype = ctypes.c_double

    # Test Regression
    X = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ], dtype="float64")
    Y = np.array([
        4,
        -5,
        -10
    ], dtype="float64")

    model = my_lib.linear_create_model(len(X[0]))

    flattened_X = X.flatten()

    print("Valeurs prédites avant entrainement : ")
    for inputs_k in X:
        print(my_lib.linear_predict_regression(model,
                                               inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                               len(inputs_k)
                                               ))

    my_lib.linear_train_regression(
        model,
        flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(X),
        len(X[0])
    )

    print("Valeurs prédites après entrainement : ")
    for inputs_k in X:
        print(my_lib.linear_predict_regression(model,
                                               inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                               len(inputs_k)
                                               ))

    my_lib.linear_dispose_model(model, len(X[0]))

