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


colorRGB = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 102, 255], [255, 128, 0], [127, 0, 255], [238, 130, 238],
            [255, 99, 71], [255, 255, 0], [0, 255, 255], [255, 0, 255], [200, 121, 0]]


def export(file, vs, faces, vn=None, color=None):
    with open(file, 'w+') as f:
        for vi, v in enumerate(vs):
            if color is None:
                f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            else:
                f.write("v %f %f %f %f %f %f\n" % (v[0], v[1], v[2], colorRGB[color[vi]][0],
                                                   colorRGB[color[vi]][1], colorRGB[color[vi]][2]))
            if vn is not None:
                f.write("vn %f %f %f\n" % (vn[vi, 0], vn[vi, 1], vn[vi, 2]))
        for face in faces:
            f.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))


def export_segments(file, segments):
    if not self.export_folder:
        return
    cur_segments = segments
    for i in range(self.pool_count + 1):
        filename, file_extension = os.path.splitext(self.filename)
        file = '%s/%s_%d%s' % (self.export_folder, filename, i, file_extension)
        fh, abs_path = mkstemp()
        edge_key = 0
        with os.fdopen(fh, 'w') as new_file:
            with open(file) as old_file:
                for line in old_file:
                    if line[0] == 'e':
                        new_file.write('%s %d' % (line.strip(), cur_segments[edge_key]))
                        if edge_key < len(cur_segments):
                            edge_key += 1
                            new_file.write('\n')
                    else:
                        new_file.write(line)
        os.remove(file)
        move(abs_path, file)
        if i < len(self.history_data['edges_mask']):
            cur_segments = segments[:len(self.history_data['edges_mask'][i])]
            cur_segments = cur_segments[self.history_data['edges_mask'][i]]


def random_file_name(ext, prefix='temp'):
    return f'{prefix}{uuid.uuid4()}.{ext}'


def get_mesh_path(file: str, about: str = "", fileType: str = ".pkl"):
    load_file = file.parent / "cache" / (file.stem + "_" + about + fileType)
    if not os.path.isdir(file.parent / "cache"):
        os.makedirs(file.parent / "cache", exist_ok=True)
    return load_file
