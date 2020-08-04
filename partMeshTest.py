import torch
from models.layers.mesh import Mesh, PartMesh
import utils
import numpy as np
from p2m_options import Options
import time
import os

mesh = Mesh("../data/style/Building/simplified/Asian_1.obj", device=torch.device("cpu"), hold_history=True)
part_mesh = PartMesh(mesh, num_parts=8, bfs_depth=0)
part_mesh.export("partMesh.obj")