from ctypes import *
print("Chargement de la DLL et definition des signatures des méthodes")
my_dll_path = "C:/Users/Damien/Desktop/HETIC/dev/Annual_Project/Project/Lib/target/debug/lib.d "

ml_lib = CDLL(my_dll_path)

