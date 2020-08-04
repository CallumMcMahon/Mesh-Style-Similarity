import numpy as np
import trimesh
import os
import warnings
warnings.simplefilter("ignore")

trimesh.util.attach_to_log(level=60)

# mesh objects can be created from existing faces and vertex data
mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                       faces=[[0, 1, 2]])

# mesh objects can be loaded from a file name or from a buffer
# you can pass any of the kwargs for the `Trimesh` constructor
# to `trimesh.load`, including `process=False` if you would like
# to preserve the original loaded data without merging vertices
# STL files will be a soup of disconnected triangles without
# merging vertices however and will not register as watertight
#mesh = trimesh.load('../data/style_data/building/train/asian_01.obj', )
file_loc = '../data/style/Building/simplified/'
watertight = 0
not_watertight = 0
mean_edges = 0
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for filename in os.listdir(file_loc):
        try:
            mesh = trimesh.load_mesh(file_loc+filename, "obj")
            if mesh.is_watertight:
                watertight+=1
                print(filename, mesh.is_watertight, mesh.edges.shape)
            else:
                not_watertight+=1

        except:
            pass

# is the current mesh watertight?
print("water", watertight, "not-water", not_watertight)