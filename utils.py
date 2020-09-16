import torch
import numpy as np
import os
import uuid
from p2m_options import MANIFOLD_DIR
import glob


def manifold_upsample(mesh, save_path, Mesh, num_faces=2000, res=3000, simplify=True):
    # export before upsample
    fname = os.path.join(save_path, 'recon_{}.obj'.format(len(mesh.faces)))
    mesh.export(fname)

    temp_file = os.path.join(save_path, random_file_name('obj'))
    opts = ' ' + str(res) if res is not None else ''

    manifold_script_path = os.path.join(MANIFOLD_DIR, 'manifold')
    if not os.path.exists(manifold_script_path):
        raise FileNotFoundError(f'{manifold_script_path} not found')
    cmd = "{} {} {}".format(manifold_script_path, fname, temp_file + opts)
    os.system(cmd)

    if simplify:
        cmd = "{} -i {} -o {} -f {}".format(os.path.join(MANIFOLD_DIR, 'simplify'), temp_file,
                                            temp_file, num_faces)
        os.system(cmd)

    m_out = Mesh(temp_file, hold_history=True, device=mesh.device)
    fname = os.path.join(save_path, 'recon_{}_after.obj'.format(len(m_out.faces)))
    m_out.export(fname)
    [os.remove(_) for _ in list(glob.glob(os.path.splitext(temp_file)[0] + '*'))]
    return m_out


def read_pts(pts_file):
    '''
    :param pts_file: file path of a plain text list of points
    such that a particular line has 6 float values: x, y, z, nx, ny, nz
    which is typical for (plaintext) .ply or .xyz
    :return: xyz, normals
    '''
    xyz, normals = [], []
    with open(pts_file, 'r') as f:
        # line = f.readline()
        spt = f.read().split('\n')
        # while line:
        for line in spt:
            parts = line.strip().split(' ')
            try:
                x = np.array(parts, dtype=np.float32)
                xyz.append(x[:3])
                normals.append(x[3:])
            except:
                pass
    return np.array(xyz, dtype=np.float32), np.array(normals, dtype=np.float32)


def load_obj(file):
    vs, faces = [], []
    f = open(file)
    for line in f:
        line = line.strip()
        splitted_line = line.split()
        if not splitted_line:
            continue
        elif splitted_line[0] == 'v':
            vs.append([float(v) for v in splitted_line[1:4]])
        elif splitted_line[0] == 'f':
            face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
            assert len(face_vertex_ids) == 3
            face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                               for ind in face_vertex_ids]
            faces.append(face_vertex_ids)
    f.close()
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return vs, faces


RGBs = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 102, 255], [255, 128, 0], [127, 0, 255], [238, 130, 238],
            [255, 99, 71], [255, 255, 0], [0, 255, 255], [255, 0, 255], [200, 121, 0]]


def export(file, vs, faces, edges=None, ve=None, v_color=None, e_color=None):
    # exports mesh with potential vertex or edge coloring
    # edge coloring idea from https://stackoverflow.com/questions/44278650/coloring-mesh-edges-in-meshlab
    with open(file, 'w+') as f:
        for vi, v in enumerate(vs):
            if v_color is not None:
                f.write("v {} {} {} {} {} {}\n".format(v[0], v[1], v[2], *v_color[vi]))
            elif e_color is not None and ve is not None:
                mean_color = e_color[ve[vi]].mean(0)
                f.write("v {} {} {} {} {} {}\n".format(v[0], v[1], v[2], *mean_color))
            else:
                f.write("v {} {} {}\n".format(v[0], v[1], v[2]))

        if e_color is not None:
            index = vs.shape[0]
            extra_faces = []
            for edge in edges:
                # midpoint
                m = vs[[edge[0], edge[1]]].mean(0)
                f.write("v {} {} {} {} {} {}\n".format(m[0], m[1], m[2], *e_color[vi]))
                index += 1
                extra_faces.append("f {} {} {}\n".format(edge[0]+1, index, edge[1]+1))

        for face in faces:
            f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))
        if e_color is not None:
            for face in extra_faces:
                f.write(face)

def convert_to_grayscale(im_as_arr):
    # adapted from https://github.com/utkuozbulak/pytorch-cnn-visualizations
    """
        Converts edge feature gradients to grayscale
    Args:
        im_as_arr (numpy arr): features per edge with shape (C, E)
    returns:
        grayscale_im (numpy_arr): Grayscale labels with shape (E)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    return grayscale_im


class act_hook():
    def __init__(self):
        self.activations = dict()

    def __call__(self, m, input, output):
        #grab output to layer and store in dict with key as layer name
        print(m, m.__class__.__name__, ": ", input[0].size(), input[1].edges_count)
        self.activations[m] = output.detach().to("cpu")

def register_hooks(net, layers, is_dataParallel):
    obj = net.module if is_dataParallel else net
    act = act_hook()
    for layer in layers:
        getattr(obj, layer).register_forward_hook(act)


def random_file_name(ext, prefix='temp'):
    return f'{prefix}{uuid.uuid4()}.{ext}'


def get_mesh_path(file: str, about: str = "", fileType: str = ".pkl"):
    load_file = file.parent / "cache" / (file.stem + "_" + about + fileType)
    if not os.path.isdir(file.parent / "cache"):
        os.makedirs(file.parent / "cache", exist_ok=True)
    return load_file
