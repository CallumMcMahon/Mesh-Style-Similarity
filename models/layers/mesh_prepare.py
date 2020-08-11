import numpy as np
import torch
import os
import ntpath

# Data augmentation methods
def augmentation(mesh, opt, faces=None):
    if hasattr(opt, 'scale_verts') and opt.scale_verts:
        scale_verts(mesh)
    if hasattr(opt, 'flip_edges') and opt.flip_edges:
        faces = flip_edges(mesh, opt.flip_edges, faces)
    return faces


def post_augmentation(mesh, opt):
    if hasattr(opt, 'slide_verts') and opt.slide_verts:
        slide_verts(mesh, opt.slide_verts)


def slide_verts(mesh, prct):
    edge_points = get_edge_points(mesh)
    dihedral = dihedral_angle(mesh, edge_points).squeeze() #todo make fixed_division epsilon=0
    thr = np.mean(dihedral) + np.std(dihedral)
    vids = np.random.permutation(len(mesh.ve))
    target = int(prct * len(vids))
    shifted = 0
    for vi in vids:
        if shifted < target:
            edges = mesh.ve[vi]
            if len(dihedral[edges]) > 0 and min(dihedral[edges]) > 2.65:
                edge = mesh.edges[np.random.choice(edges)]
                vi_t = edge[1] if vi == edge[0] else edge[0]
                nv = mesh.vs[vi] + np.random.uniform(0.2, 0.5) * (mesh.vs[vi_t] - mesh.vs[vi])
                mesh.vs[vi] = nv
                shifted += 1
        else:
            break
    mesh.shifted = shifted / len(mesh.ve)


def scale_verts(mesh, mean=1, var=0.1):
    for i in range(mesh.vs.shape[1]):
        mesh.vs[:, i] = mesh.vs[:, i] * np.random.normal(mean, var)


def angles_from_faces(mesh, edge_faces, faces):
    normals = [None, None]
    for i in range(2):
        edge_a = mesh.vs[faces[edge_faces[:, i], 2]] - mesh.vs[faces[edge_faces[:, i], 1]]
        edge_b = mesh.vs[faces[edge_faces[:, i], 1]] - mesh.vs[faces[edge_faces[:, i], 0]]
        normals[i] = np.cross(edge_a, edge_b)
        div = fixed_division(torch.norm(normals[i], dim=1), epsilon=0)
        normals[i] /= div[:, np.newaxis]
    dot = np.sum(normals[0] * normals[1], axis=1).clip(-1, 1)
    angles = np.pi - np.arccos(dot)
    return angles


def flip_edges(mesh, prct, faces):
    edge_count, edge_faces, edges_dict = get_edge_faces(faces)
    dihedral = angles_from_faces(mesh, edge_faces[:, 2:], faces)
    edges2flip = np.random.permutation(edge_count)
    # print(dihedral.min())
    # print(dihedral.max())
    target = int(prct * edge_count)
    flipped = 0
    for edge_key in edges2flip:
        if flipped == target:
            break
        if dihedral[edge_key] > 2.7:
            edge_info = edge_faces[edge_key]
            if edge_info[3] == -1:
                continue
            new_edge = tuple(sorted(list(set(faces[edge_info[2]]) ^ set(faces[edge_info[3]]))))
            if new_edge in edges_dict:
                continue
            new_faces = np.array(
                [[edge_info[1], new_edge[0], new_edge[1]], [edge_info[0], new_edge[0], new_edge[1]]])
            if check_area(mesh, new_faces):
                del edges_dict[(edge_info[0], edge_info[1])]
                edge_info[:2] = [new_edge[0], new_edge[1]]
                edges_dict[new_edge] = edge_key
                rebuild_face(faces[edge_info[2]], new_faces[0])
                rebuild_face(faces[edge_info[3]], new_faces[1])
                for i, face_id in enumerate([edge_info[2], edge_info[3]]):
                    cur_face = faces[face_id]
                    for j in range(3):
                        cur_edge = tuple(sorted((cur_face[j], cur_face[(j + 1) % 3])))
                        if cur_edge != new_edge:
                            cur_edge_key = edges_dict[cur_edge]
                            for idx, face_nb in enumerate(
                                    [edge_faces[cur_edge_key, 2], edge_faces[cur_edge_key, 3]]):
                                if face_nb == edge_info[2 + (i + 1) % 2]:
                                    edge_faces[cur_edge_key, 2 + idx] = face_id
                flipped += 1
    return faces


def rebuild_face(face, new_face):
    new_point = list(set(new_face) - set(face))[0]
    for i in range(3):
        if face[i] not in new_face:
            face[i] = new_point
            break
    return face

def check_area(mesh, faces):
    face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    face_areas *= 0.5
    return face_areas[0] > 0 and face_areas[1] > 0


def get_edge_faces(faces):
    edge_count = 0
    edge_faces = []
    edge2keys = dict()
    for face_id, face in enumerate(faces):
        for i in range(3):
            cur_edge = tuple(sorted((face[i], face[(i + 1) % 3])))
            if cur_edge not in edge2keys:
                edge2keys[cur_edge] = edge_count
                edge_count += 1
                edge_faces.append(np.array([cur_edge[0], cur_edge[1], -1, -1]))
            edge_key = edge2keys[cur_edge]
            if edge_faces[edge_key][2] == -1:
                edge_faces[edge_key][2] = face_id
            else:
                edge_faces[edge_key][3] = face_id
    return edge_count, np.array(edge_faces), edge2keys


def set_edge_lengths(mesh, edge_points=None):
    if edge_points is not None:
        edge_points = get_edge_points(mesh)
    edge_lengths = torch.norm(mesh.vs[edge_points[:, 0]] - mesh.vs[edge_points[:, 1]], dim=1)
    mesh.edge_lengths = edge_lengths


def extract_features(mesh):
    features = []
    edge_points = get_edge_points(mesh)
    set_edge_lengths(mesh, edge_points)
    with np.errstate(divide='raise'):
        try:
            for extractor in [dihedral_angle, symmetric_opposite_angles, symmetric_ratios]:
                feature = extractor(mesh, edge_points)
                features.append(feature)
            mesh.features = torch.cat(features, dim=0)
        except Exception as e:
            print(e)
            raise ValueError(mesh.filename, 'bad features')


def dihedral_angle(mesh, edge_points):
    normals_a = get_normals(mesh, edge_points, 0)
    normals_b = get_normals(mesh, edge_points, 3)
    dot = torch.sum(normals_a * normals_b, dim=1).clamp(-1, 1)
    angles = (np.pi - np.arccos(dot)).unsqueeze(0)
    return angles


def symmetric_opposite_angles(mesh, edge_points):
    """ computes two angles: one for each face shared between the edge
        the angle is in each face opposite the edge
        sort handles order ambiguity
    """
    angles_a = get_opposite_angles(mesh, edge_points, 0)
    angles_b = get_opposite_angles(mesh, edge_points, 3)
    angles = torch.stack([angles_a, angles_b])
    angles = torch.sort(angles, dim=0)[0]
    return angles


def symmetric_ratios(mesh, edge_points):
    """ computes two ratios: one for each face shared between the edge
        the ratio is between the height / base (edge) of each triangle
        sort handles order ambiguity
    """
    ratios_a = get_ratios(mesh, edge_points, 0)
    ratios_b = get_ratios(mesh, edge_points, 3)
    ratios = torch.stack([ratios_a, ratios_b])
    return torch.sort(ratios, dim=0)[0]


def get_edge_points(mesh):
    """ returns: edge_points (#E x 4) tensor, with four vertex ids per edge
        for example: edge_points[edge_id, 0] and edge_points[edge_id, 1] are the two vertices which define edge_id 
        each adjacent face to edge_id has another vertex, which is edge_points[edge_id, 2] or edge_points[edge_id, 3]
    """
    edge_points = np.zeros([mesh.edges_count, 4], dtype=np.int32)
    for edge_id, edge in enumerate(mesh.edges):
        edge_points[edge_id] = get_side_points(mesh, edge_id)
        # edge_points[edge_id, 3:] = mesh.get_side_points(edge_id, 2)
    return edge_points


def get_side_points(mesh, edge_id):
    """ if a boundary is detected, use the available triangle twice
    This should lead to 0 dihedral angle, and a copy of the face-specific features to the missing face """
    edge_a = mesh.edges[edge_id]

    # if a side is unavailable, copy the available side's edges
    if mesh.gemm_edges[edge_id, 0] == -1:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    else:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    if mesh.gemm_edges[edge_id, 2] == -1:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    else:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 3]]

    #
    first_vertex = 0
    third_vertex = 0
    forth_vertex = 0
    if edge_a[1] in edge_b:
        first_vertex = 1
    if edge_b[1] in edge_c:
        third_vertex = 1
    if edge_d[1] in edge_e:
        forth_vertex = 1
    return [edge_a[first_vertex], edge_a[1 - first_vertex], edge_b[third_vertex], edge_d[forth_vertex]]


def get_normals(mesh, edge_points, side):
    edge_a = mesh.vs[edge_points[:, side // 2 + 2]] - mesh.vs[edge_points[:, side // 2]]
    edge_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2]]
    normals = torch.cross(edge_a, edge_b)
    normals /= fixed_division(torch.norm(normals, dim=1), epsilon=0.1).unsqueeze(1)
    return normals

def get_opposite_angles(mesh, edge_points, side):
    edges_a = mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]
    edges_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]

    edges_a /= fixed_division(torch.norm(edges_a, dim=1), epsilon=0.1).unsqueeze(1)
    edges_b /= fixed_division(torch.norm(edges_b, dim=1), epsilon=0.1).unsqueeze(1)
    dot = torch.sum(edges_a * edges_b, dim=1).clamp(-1, 1)
    return np.arccos(dot)


def get_ratios(mesh, edge_points, side):
    edges_lengths = torch.norm(mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, 1 - side // 2]],
                                dim=1)
    point_o = mesh.vs[edge_points[:, side // 2 + 2]]
    point_a = mesh.vs[edge_points[:, side // 2]]
    point_b = mesh.vs[edge_points[:, 1 - side // 2]]
    line_ab = point_b - point_a
    projection_length = torch.sum(line_ab * (point_o - point_a), dim=1) / fixed_division(
        torch.norm(line_ab, dim=1), epsilon=0.1)
    closest_point = point_a + (projection_length / edges_lengths)[:, np.newaxis] * line_ab
    d = torch.norm(point_o - closest_point, dim=1)
    return d / edges_lengths

def fixed_division(to_div, epsilon):
    if epsilon == 0:
        to_div[to_div == 0] = 0.1
    else:
        to_div += epsilon
    return to_div
