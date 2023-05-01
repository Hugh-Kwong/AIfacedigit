import sys
import numpy as np
import os

# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)
file = open("facedatatrain.txt", "r")

faces = np.array([[1, 2, 3,], [4, 5, 6], [7, 8, 9]])
print(faces)
# print(file.read())