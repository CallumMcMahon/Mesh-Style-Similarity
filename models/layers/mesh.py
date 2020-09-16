import torch
import numpy as np
from queue import Queue
from utils import load_obj, export, convert_to_grayscale, get_mesh_path
from models.layers.mesh_prepare import extract_features
import copy
from pathlib import Path
import pickle
import os
#from pytorch3d.ops.knn import knn_gather, knn_points


class Mesh:

    def __init__(self, file, hold_history=False, vs=None, faces=None, device='cpu', gfmm=False):
        if file is None:
            return
        self.filename = Path(file)
        self.vs = self.v_mask = self.edge_areas = self.v_color = self.e_color = None
        self.edges = self.gemm_edges = self.sides = None
        self.device = device
        self.create_connectivity(vs, faces)
        if type(self.vs) is np.ndarray:
            self.vs = torch.from_numpy(self.vs)
        if type(self.faces) is np.ndarray:
            self.faces = torch.from_numpy(self.faces)
        extract_features(self)
        self.history_data = None
        if hold_history:
            self.init_history()

        # self.vs = self.vs.to(self.device)
        # self.faces = self.faces.to(self.device).long()
        # self.area, self.normals = self.face_areas_normals(self.vs, self.faces)

    def create_connectivity(self, vs, faces):
        if vs is not None and faces is not None:
            self.vs, self.faces = vs.cpu().numpy(), faces.cpu().numpy()
            self.scale, self.translations = 1.0, np.zeros(3,)
        else:
            self.vs, self.faces = load_obj(self.filename)
            self.normalize_unit_bb()
        self.vs_in = copy.deepcopy(self.vs)
        self.v_mask = np.ones(len(self.vs), dtype=bool)
        self.faces, self.face_areas = self.remove_non_manifolds(self.faces)
        self.build_gemm()

    def build_gemm(self):
        """
        gemm_edges: array (#E x 4) of the 4 one-ring neighbors for each edge
        sides: array (#E x 4) indices (values of: 0,1,2,3) indicating where an edge is in the gemm_edge entry of the 4 neighboring edges
        for example edge i -> gemm_edges[gemm_edges[i], sides[i]] == [i, i, i, i]
        """
        # ve is every edge that connects to each vertex
        self.ve = [[] for _ in self.vs]
        # vei is whether the vertex is the lower or higher (0 or 1) index of edges ve connected
        self.vei = [[] for _ in self.vs]
        # edge_nb is "fake image" edge neighbourhood
        edge_nb = []
        sides = []
        # translate edges to an index
        edge2key = dict()
        edges = []
        edges_count = 0
        nb_count = []
        for face_id, face in enumerate(self.faces):
            # list of sorted edges belonging to face. i.e. [(v1, v2), ...]
            faces_edges = [tuple(sorted([face[i], face[(i + 1) % 3]])) for i in range(3)]
            for idx, edge in enumerate(faces_edges):
                if edge not in edge2key:
                    # assign an index to each edge
                    edge2key[edge] = edges_count
                    edges.append(list(edge))
                    edge_nb.append([-1, -1, -1, -1])
                    sides.append([-1, -1, -1, -1])
                    # add edge index to list of edges coming from vertex v
                    self.ve[edge[0]].append(edges_count)
                    self.ve[edge[1]].append(edges_count)
                    self.vei[edge[0]].append(0)
                    self.vei[edge[1]].append(1)
                    nb_count.append(0)
                    edges_count += 1
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                # add two other edges' indexes on face to neighbourhood ring
                edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
                edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
                # remember if it's the first or second touching face that we've gotten edges from
                nb_count[edge_key] += 2
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                # inverse function of edge neighbourhood, how to get from neighbours back to original edge
                # i.e. for each edge, which index in the other edges' 4-ring neighbourhood does the edge appear
                # edge_nb[edge_nb[edge_key, i], sides[edge_nb[edge_key, i]] = edge_key
                sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
                sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
        self.edges = np.array(edges, dtype=np.int32)
        self.gemm_edges = np.array(edge_nb, dtype=np.int64)
        self.sides = np.array(sides, dtype=np.int64)
        self.edges_count = edges_count
        self.edge2key = edge2key

        # lots of DS for loss
        self.nvs, self.nvsi, self.nvsin = [], [], []
        for i, e in enumerate(self.ve):
            self.nvs.append(len(e))
            self.nvsi.append(len(e) * [i])
            self.nvsin.append(list(range(len(e))))
        self.vei = torch.from_numpy(np.concatenate(np.array(self.vei)).ravel()).to(self.device).long()
        self.nvsi = torch.Tensor(np.concatenate(np.array(self.nvsi)).ravel()).to(self.device).long()
        self.nvsin = torch.from_numpy(np.concatenate(np.array(self.nvsin)).ravel()).to(self.device).long()
        ve_in = copy.deepcopy(self.ve)
        self.ve_in = torch.from_numpy(np.concatenate(np.array(ve_in)).ravel()).to(self.device).long()
        self.max_nvs = max(self.nvs)
        self.nvs = torch.Tensor(self.nvs).to(self.device).float()
        for i in range(self.gemm_edges.shape[0]):
            for idx, j in enumerate(self.gemm_edges[self.gemm_edges[i], self.sides[i]]):
                assert j == i or self.gemm_edges[i, idx] == -1

    def build_ef(self):
        edge_faces = dict()
        if type(self.faces) == torch.Tensor:
            faces = self.faces.cpu().numpy()
        else:
            faces = self.faces
        for face_id, face in enumerate(faces):
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                if edge not in edge_faces:
                    edge_faces[edge] = []
                edge_faces[edge].append(face_id)
        for k in edge_faces.keys():
            if len(edge_faces[k]) < 2:
                edge_faces[k].append(edge_faces[k][0])
        return edge_faces

    def build_gfmm(self):
        edge_faces = self.build_ef()
        gfmm = []
        if type(self.faces) == torch.Tensor:
            faces = self.faces.cpu().numpy()
        else:
            faces = self.faces
        for face_id, face in enumerate(faces):
            neighbors = [face_id]
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                neighbors.extend(list(set(edge_faces[edge]) - set([face_id])))
            gfmm.append(neighbors)
        self.gfmm = torch.Tensor(gfmm).long().to(self.device)

    def normalize_unit_bb(self):
        """
        normalizes to unit bounding box and translates to center
        :param verts: new verts
        """
        cache_norm_file = get_mesh_path(self.filename, 'normalize', ".npz")
        if not cache_norm_file.exists():
            scale = max([self.vs[:, i].max() - self.vs[:, i].min() for i in range(3)])
            scaled_vs = self.vs / scale
            target_mins = [(scaled_vs[:, i].max() - scaled_vs[:, i].min()) / -2.0 for i in range(3)]
            translations = [(target_mins[i] - scaled_vs[:, i].min()) for i in range(3)]
            np.savez_compressed(cache_norm_file, scale=scale, translations=translations)
        # load from the cache
        cached_data = np.load(cache_norm_file, encoding='latin1', allow_pickle=True)
        self.scale, self.translations = cached_data['scale'], cached_data['translations']
        self.vs /= self.scale
        self.vs += self.translations[None, :]

    def discrete_project(self, pc: torch.Tensor, thres=0.9, cpu=False):
        with torch.no_grad():
            device = torch.device('cpu') if cpu else self.device
            pc = pc.double()
            if isinstance(self, Mesh):
                mid_points = self.vs[self.faces].mean(dim=1)
                normals = self.normals
            else:
                mid_points = self[:, :3]
                normals = self[:, 3:]
            pk12 = knn_points(mid_points[:, :3].unsqueeze(0), pc[:, :, :3], K=3).idx[0]
            pk21 = knn_points(pc[:, :, :3], mid_points[:, :3].unsqueeze(0), K=3).idx[0]
            loop = pk21[pk12].view(pk12.shape[0], -1)
            knn_mask = (loop == torch.arange(0, pk12.shape[0], device=self.device)[:, None]).sum(dim=1) > 0
            mid_points = mid_points.to(device)
            pc = pc[0].to(device)
            normals = normals.to(device)[~ knn_mask, :]
            masked_mid_points = mid_points[~ knn_mask, :]
            displacement = masked_mid_points[:, None, :] - pc[:, :3]
            torch.cuda.empty_cache()
            distance = displacement.norm(dim=-1)
            mask = (torch.abs(torch.sum((displacement / distance[:, :, None]) *
                                        normals[:, None, :], dim=-1)) > thres)
            if pc.shape[-1] == 6:
                pc_normals = pc[:, 3:]
                normals_correlation = torch.sum(normals[:, None, :] * pc_normals, dim=-1)
                mask = mask * (normals_correlation > 0)
            torch.cuda.empty_cache()
            distance[~ mask] += float('inf')
            min, argmin = distance.min(dim=-1)

            pc_per_face_masked = pc[argmin, :].clone()
            pc_per_face_masked[min == float('inf'), :] = float('nan')
            pc_per_face = torch.zeros(mid_points.shape[0], 6).\
                type(pc_per_face_masked.dtype).to(pc_per_face_masked.device)
            pc_per_face[~ knn_mask, :pc.shape[-1]] = pc_per_face_masked
            pc_per_face[knn_mask, :] = float('nan')

            # clean up
            del knn_mask
        return pc_per_face.to(self.device), (pc_per_face[:, 0] == pc_per_face[:, 0]).to(device)

    def remove_non_manifolds(self, faces):
        ''' Removes faces which form T-junctions (more than 2 faces with same edge). border faces are untouched
        :param mesh:
        :param faces:
        :return: subset of faces which do not break the 1-ring assumption
        '''
        edges_set = set()
        # True values in mask are manifold and are kept
        mask = np.ones(len(faces), dtype=bool)
        face_areas, _ = self.face_areas_normals(self.vs, self.faces)
        for face_id, face in enumerate(self.faces):
            if face_areas[face_id] == 0:
                mask[face_id] = False
                continue
            faces_edges = []
            is_manifold = True
            for i in range(3):
                cur_edge = (face[i], face[(i + 1) % 3])
                # each edge added twice, as (a, b) and (b, a). stop edge from being added third time
                if cur_edge in edges_set:
                    is_manifold = False
                    break
                else:
                    faces_edges.append(cur_edge)
            if not is_manifold:
                mask[face_id] = False
            else:
                for idx, edge in enumerate(faces_edges):
                    edges_set.add(edge)
        return faces[mask], face_areas[mask]

    @staticmethod
    def face_areas_normals(vs, faces):
        if type(vs) is not torch.Tensor:
            vs = torch.from_numpy(vs)
        if type(faces) is not torch.Tensor:
            faces = torch.from_numpy(faces)
        face_normals = torch.cross(vs[faces[:, 1]] - vs[faces[:, 0]],
                                   vs[faces[:, 2]] - vs[faces[:, 1]])

        face_areas = torch.norm(face_normals, dim=1)
        face_normals = face_normals / face_areas[:, None]
        face_areas = 0.5 * face_areas
        face_areas = 0.5 * face_areas
        return face_areas, face_normals

    def update_verts(self, verts):
        """
        update verts positions only, same connectivity
        :param verts: new verts
        """
        self.vs = verts

    def deep_copy(self): #TODO see if can do this better
        new_mesh = Mesh(file=None)
        types = [np.ndarray, torch.Tensor,  dict, list, str, int, bool, float]
        for attr in self.__dir__():
            if attr == '__dict__':
                continue

            val = getattr(self, attr)
            if type(val) == types[0]:
                new_mesh.__setattr__(attr, val.copy())
            elif type(val) == types[1]:
                new_mesh.__setattr__(attr, val.clone())
            elif type(val) in types[2:4]:
                new_mesh.__setattr__(attr, pickle.loads(pickle.dumps(val, -1)))
            elif type(val) in types[4:]:
                new_mesh.__setattr__(attr, val)

        return new_mesh

    def merge_vertices(self, edge_id):
        self.remove_edge(edge_id)
        edge = self.edges[edge_id]
        # v_a = self.vs[edge[0]]
        # v_b = self.vs[edge[1]]
        # update pA
        # move first vertex to midpoint of edge
        self.vs[edge[0]] = self.vs[edge].mean(0)
        # v_a.__iadd__(v_b)
        # v_a.__itruediv__(2)
        self.v_mask[edge[1]] = False
        # replace any references to deleted vertex v_b with v_a
        mask = self.edges == edge[1]
        self.ve[edge[0]].extend(self.ve[edge[1]])
        self.edges[mask] = edge[0]


    def remove_vertex(self, v):
        # when 3 touching triangles collapse into 1, "invalid" case in mesh_pool, used by "remove_triplete"
        self.v_mask[v] = False

    def remove_edge(self, edge_id):
        vs = self.edges[edge_id]
        for v in vs:
            if edge_id not in self.ve[v]:
                print(self.ve[v])
                print(self.filename)
            self.ve[v].remove(edge_id)

    def clean(self, edges_mask, groups):
        edges_mask = edges_mask.astype(bool)
        torch_mask = torch.from_numpy(edges_mask.copy())
        # only keep edges and gemm/sides of edges remaining
        # all vertices kept (with a vs_mask), filtered edge list references original vs indices
        self.gemm_edges = self.gemm_edges[edges_mask]
        self.edges = self.edges[edges_mask]
        self.sides = self.sides[edges_mask]

        new_ve = []
        edges_mask = np.concatenate([edges_mask, [False]])
        new_indices = np.zeros(edges_mask.shape[0], dtype=np.int32)
        new_indices[-1] = -1
        # update indices in gemm to reference smaller set up filtered edges
        new_indices[edges_mask] = np.arange(0, np.ma.where(edges_mask)[0].shape[0])
        self.gemm_edges[:, :] = new_indices[self.gemm_edges[:, :]]
        # todo: check if we need to filter ve by vs_mask
        for v_index, ve in enumerate(self.ve):
            update_ve = []
            for e in ve:
                update_ve.append(new_indices[e])
            new_ve.append(update_ve)
        self.ve = new_ve
        self.__clean_history(groups, torch_mask)

    def export(self, file, history=0, v_color=None, e_color=None):
        # if rgb values haven't been specified already
        if e_color.ndim != 3:
            # int per class
            if e_color.dtype == np.int32:
                RGBs = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 102, 255], [255, 128, 0], [127, 0, 255],
                                 [238, 130, 238], [255, 99, 71], [255, 255, 0], [0, 255, 255], [255, 0, 255], [200, 121, 0]])
                e_color = RGBs[e_color]
            # float values to be mapped to intensity
            else:
                # rescale values
                e_color = convert_to_grayscale(e_color)
                e_color = np.stack([e_color * 255, (1-e_color) * 255, np.zeros_like(e_color)]).T
        vs = self.vs.cpu().clone().numpy() if history is None else self.history_data['vs'][history]
        vs -= self.translations[None, :]
        vs *= self.scale
        if history is None:
            export(file, vs, self.faces, self.edges, self.ve, v_color=v_color, e_color=e_color)
        else:
            v_mask = self.history_data['v_mask'][history]
            vs = vs[v_mask]
            new_indices = np.zeros(v_mask.shape[0], dtype=np.int32)
            new_indices[v_mask] = np.arange(0, np.ma.where(v_mask)[0].shape[0])
            gemm = self.history_data['gemm_edges'][history]
            sides = self.history_data['sides'][history]
            edges = self.history_data["edges"][history]
            ve = self.history_data["ve"][history]
            faces = []
            assert e_color.shape[0] == gemm.shape[0]
            for i in range(self.gemm_edges.shape[0]):
                for idx, j in enumerate(self.gemm_edges[self.gemm_edges[i], self.sides[i]]):
                    assert j == i or self.gemm_edges[i, idx] == -1
            for edge_index in range(len(gemm)):
                cycles = self.__get_cycle2(sides, gemm, edge_index)
                for cycle in cycles:
                    faces.append(self.__cycle_to_face(edges, cycle, new_indices))
            # deduplicate_faces = []
            # for face in faces:
            #     if face not in deduplicate_faces:
            #         deduplicate_faces.append(face)
            # faces = deduplicate_faces
            edges = new_indices[edges]
            export(file, vs, faces, edges, ve, v_color=v_color, e_color=e_color)

    @staticmethod
    def __get_cycle2(sides, gemm, edge_id):
        cycles = []
        # each edge has two faces connected to it
        for j in range(2):
            start_point = j * 2
            # if edge is on boundary, skip
            if gemm[edge_id, start_point] == -1:
                continue
            cycles.append([edge_id, gemm[edge_id, start_point], gemm[edge_id, start_point+1]])
        return cycles

    @staticmethod
    def __get_cycle(sides, gemm, edge_id):
        cycles = []
        # each edge has two faces connected to it
        for j in range(2):
            next_side = start_point = j * 2
            next_key = edge_id
            # if edge is on boundary, skip
            if gemm[edge_id, start_point] == -1:
                continue
            cycles.append([])
            for i in range(3):
                # view face from perspective of all 3 edges
                tmp_next_key = gemm[next_key, next_side]
                tmp_next_side = sides[next_key, next_side]
                # make sure next edge to focus on is not current or past visited edge
                tmp_next_side = tmp_next_side + 1 - 2 * (tmp_next_side % 2)
                # set edges to -1 so if face comes up again, know not to add it. Not convinced it's needed...
                gemm[next_key, next_side] = -1
                gemm[next_key, next_side + 1 - 2 * (next_side % 2)] = -1
                # hop to next edge in face
                next_key = tmp_next_key
                next_side = tmp_next_side
                cycles[-1].append(next_key)
        return cycles

    @staticmethod
    def __cycle_to_face2(edges, cycle, v_indices):
        face = list(np.unique(edges[cycle]))
        assert len(face) == 3
        return face

    @staticmethod
    def __cycle_to_face(edges, cycle, v_indices):
        # turns 3 edges of face in correspoding 3 vertices in semantic order
        face = []
        for i in range(3):
            v = list(set(edges[cycle[i]]) & set(edges[cycle[(i + 1) % 3]]))
            assert len(v)>0
            face.append(v_indices[v[0]])
        return face

    # TODO add export segmentation

    def init_history(self):
        self.history_data = {
            'groups': [],
            'gemm_edges': [self.gemm_edges.copy()],
            'occurrences': [],
            'edges_count': [self.edges_count],
            # export info
            'vs': [self.vs.cpu().clone().numpy()],
            'sides': [self.sides.copy()],
            'v_mask': [torch.ones(self.vs.shape[0], dtype=torch.bool)],
            'edges': [np.copy(self.edges)],
            've': [np.copy(self.ve)],
            }


    def get_groups(self):
        return self.history_data['groups'].pop()

    def get_occurrences(self):
        return self.history_data['occurrences'].pop()
    
    def __clean_history(self, groups, pool_mask):
        if self.history_data is not None:
            #edges and ve get overwritten with subset each pooling, vs gets masked (and mutated in pool)
            self.history_data['occurrences'].append(groups.get_occurrences())
            self.history_data['groups'].append(groups.get_groups(pool_mask))
            self.history_data['gemm_edges'].append(self.gemm_edges.copy())
            self.history_data['edges_count'].append(self.edges_count)
            # export info
            self.history_data['vs'].append(self.vs.cpu().clone().numpy())
            self.history_data['sides'].append(self.sides.copy())
            self.history_data['v_mask'].append(np.copy(self.v_mask))
            self.history_data['edges'].append(np.copy(self.edges))
            self.history_data['ve'].append(np.copy(self.ve))

    
    def unroll_gemm(self):
        self.history_data['gemm_edges'].pop()
        self.gemm_edges = self.history_data['gemm_edges'][-1]
        self.history_data['edges_count'].pop()
        self.edges_count = self.history_data['edges_count'][-1]

    @staticmethod
    def from_tensor(mesh, vs, faces, gfmm=False):
        return Mesh(file=mesh.filename, vs=vs, faces=faces, device=mesh.device, hold_history=True, gfmm=gfmm)

    def submesh(self, vs_index):
        return PartMesh.create_submesh(vs_index, self)


class PartMesh:
    """
    Divides a mesh into submeshes
    """
    def __init__(self, main_mesh: Mesh, vs_groups=None, num_parts=1, bfs_depth=0, n=-1):
        """
        Part Mesh constructor
        :param main_mesh: main mesh to pick the submeshes from
        :param vs_groups: tensor the size of vs that contains the submesh index from 0 upto number_of_sub_meshes - 1
        :param num_parts: number of parts to seperate the main_mesh into
        """
        self.main_mesh = main_mesh
        self.bfs_depth = bfs_depth
        # create cached data for each submesh
        self.vs_groups = PartMesh.segment_shape(self.main_mesh.vs, seg_num=num_parts)
        torch.unique(self.vs_groups)
        self.n_submeshes = torch.max(self.vs_groups).item() + 1
        self.sub_mesh_index = []
        self.sub_mesh = []
        self.init_verts = []
        tmp_vs_groups = self.vs_groups.clone()
        for i in range(self.n_submeshes):
            vs_index = (self.vs_groups == i).nonzero().squeeze(1)
            # remove empty groups and shift other group indexing down
            if vs_index.size()[0] == 0:
                tmp_vs_groups[self.vs_groups > i] -= 1
                continue
            vs_index = torch.sort(vs_index, dim=0)[0]
            vs_index = torch.tensor(self.vs_bfs(vs_index.tolist(), self.main_mesh.faces.tolist(), self.bfs_depth),
                                    dtype=vs_index.dtype).to(vs_index.device)

            m, vs_index = self.create_submesh(vs_index, self.main_mesh)
            self.sub_mesh.append(m)

            self.sub_mesh_index.append(vs_index)
            self.init_verts.append(m.vs.clone().detach())

        self.vs_groups = tmp_vs_groups
        self.n_submeshes = torch.max(self.vs_groups).item() + 1

        vse = self.vs_e_dict(self.main_mesh.edges)
        self.sub_mesh_edge_index = []
        for i in range(self.n_submeshes):
            mask = torch.zeros(self.main_mesh.edges.shape[0]).long()
            for face in self.sub_mesh[i].faces:
                face = self.sub_mesh_index[i][face].to(face.device).long()
                for j in range(3):
                    e = tuple(sorted([face[j].item(), face[(j + 1) % 3].item()]))
                    mask[vse[e]] = 1
            self.sub_mesh_edge_index.append(self.mask_to_index(mask))

    def update_verts(self, new_vs: torch.Tensor, index: int):
        m = self.sub_mesh[index]
        m.update_verts(new_vs)
        self.main_mesh.vs[self.sub_mesh_index[index], :] = new_vs

    def build_main_mesh(self):
        """
        build self.main_mesh out of submesh's vs
        """
        new_vs = torch.zeros_like(self.main_mesh.vs)
        new_vs_n = torch.zeros(self.main_mesh.vs.shape[0], dtype=new_vs.dtype).to(new_vs.device)
        colors = torch.zeros(self.main_mesh.vs.shape[0], dtype=int).to(new_vs.device)
        for i, m in enumerate(self.sub_mesh):
            new_vs[self.sub_mesh_index[i], :] += m.vs
            new_vs_n[self.sub_mesh_index[i]] += 1
            colors[self.sub_mesh_index[i]] = i

        new_vs = new_vs / new_vs_n[:, None]
        new_vs[new_vs_n == 0, :] = self.main_mesh.vs[new_vs_n == 0, :]
        self.main_mesh.update_verts(new_vs)
        self.main_mesh.v_color = colors

    def export(self, file, build_main=True):
        """
        export the entire mesh (self.main_mesh)
        :param file: file to output to
        :param vcolor: color for vertices, Default: None
        :param build_main: build main mesh before exporting, Default: True
        :param segment: color the verts according to submesh classes
        """
        with torch.no_grad():
            if build_main:
                self.build_main_mesh()
            self.main_mesh.export(file)

    def __getitem__(self, i: int) -> Mesh:
        """
        get submesh at index i
        :param i: index of submesh
        :return: submesh at index i
        """
        if type(i) != int:
            raise TypeError('number submesh must be int')
        if i >= self.n_submeshes:
            raise OverflowError(f'index {i} for submesh is out of bounds, max index is {self.n_submeshes - 1}')
        return self.sub_mesh[i]

    def __iter__(self):
        return iter(self.sub_mesh)

    @staticmethod
    def create_submesh(vs_index: torch.Tensor, mesh: Mesh) -> (Mesh, torch.Tensor):
        """
        create a submesh out on a mesh object
        :param vs_index: indices of the submesh
        :param mesh: the mesh to sub
        :return: the new submesh
        """
        vs_mask = torch.zeros(mesh.vs.shape[0])
        vs_mask[vs_index] = 1
        # include faces where at least one vertex is included from that group
        faces_mask = vs_mask[mesh.faces].sum(dim=-1) > 0
        new_faces = mesh.faces[faces_mask].clone()
        # grab all vertices that were added when forming full triangles
        all_verts = new_faces.view(-1)
        new_vs_mask = torch.zeros(mesh.vs.shape[0]).long().to(all_verts.device)
        new_vs_mask[all_verts] = 1
        # remove duplicate vertices from face vertices by reconverting mask
        new_vs_index = PartMesh.mask_to_index(new_vs_mask)
        new_vs = mesh.vs[new_vs_index, :].clone()
        vs_mask = torch.zeros(mesh.vs.shape[0])# TODO is vs_mask not the same as new_vs_mask
        vs_mask[new_vs_index] = 1

        # shift face vertex indexing so it corresponds with new smaller vertex subset
        cummusum = torch.cumsum(1 - vs_mask, dim=0)
        new_faces -= cummusum[new_faces].to(new_faces.device).long()
        # passes coordinates of relevant vertices and properly indexed face vertices
        m = Mesh.from_tensor(mesh, new_vs.detach(), new_faces.detach(), gfmm=False)
        # return indexes updated from slight expansion
        return m, new_vs_index

    @staticmethod
    def index_to_mask(index: torch.Tensor, len:int):
        mask = torch.zeros(len)
        for i in index:
            mask[i] = 1
        return mask

    @staticmethod
    def mask_to_index(mask: torch.Tensor):
        lst = []
        mask = mask.long()
        for i, val in enumerate(mask):
            if val == 1:
                lst.append(i)
        return torch.tensor(lst).type(torch.long)

    @staticmethod
    def segment_shape(vs: torch.Tensor, seg_num: int):
        """
        segment shape to 8 classes depence on the center of mass
        :param vs: tensor NX3
        :return: tensor size N with value being the class 0-7 (including 7)
        """
        center = vs.mean(dim=0)
        diff = vs - center[None, :]
        eighth = torch.zeros(vs.shape[0]).float().to(diff.device)
        if seg_num >= 2:
            eighth += 1 *(diff[:, 0] > 0).float()
        if seg_num >= 4:
            eighth += 2 * (diff[:, 1] > 0).float()
        if seg_num >= 8:
            eighth += 4 * (diff[:, 2] > 0).float()
        return eighth.long()

    @staticmethod
    def grid_segment(vs: torch.Tensor, n: int) -> torch.tensor:
        '''
        :param vs: the main mesh's vertices
        :param n: the the number of splits on each of the 3 axes, n^3 groups formed
        :return:
        '''
        maxx, _ = vs.max(dim=0)
        minn, _ = vs.min(dim=0)
        unit = (maxx - minn) / n
        vs_new = vs - minn[None, :]
        vs_cordinants = (vs_new / unit).int()
        vs_cordinants[vs_cordinants == n] -= 1
        return vs_cordinants[:, 0] + vs_cordinants[:, 1] * n + vs_cordinants[:, 2] * (n ** 2)

    @staticmethod
    def vs_e_dict(edges):
        d = dict()
        for i, e in enumerate(edges):
            k = tuple(sorted(e))
            d[k] = i
        return d

    @staticmethod
    def vs_bfs(start_vs, faces, max_depth):
        if max_depth <= 0:
            return start_vs
        q = Queue()
        [q.put((c, 0)) for c in start_vs]
        visited = start_vs
        while not q.empty():
            i, depth = q.get()
            for f in faces:
                if i in f:
                    for j in f:
                        if j not in visited:
                            if depth + 1 <= max_depth:
                                q.put((j, depth + 1))
                            visited.append(j)
        return sorted(visited)
