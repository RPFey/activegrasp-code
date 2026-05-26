from typing import Callable
import numpy as np
import open3d as o3d 
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sciR

from kinova_control_py.line_mesh import LineMesh
from kinova_control_py.pose_util import convert_pose_to_pos_quat
from kinova_control_py.tsdf_integration import O3DTsdfIntegration

def load_color_mesh(data_file_path):
    read_mesh = o3d.io.read_triangle_mesh(data_file_path)
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = read_mesh.vertices
    new_mesh.triangles = read_mesh.triangles
    new_mesh.triangle_uvs = read_mesh.triangle_uvs
    new_mesh.triangle_material_ids = read_mesh.triangle_material_ids
    new_mesh.vertex_colors = read_mesh.vertex_colors
    new_mesh.textures = [read_mesh.textures[1], read_mesh.textures[1]] 

    return new_mesh

class Gripper:
    def __init__(self) -> None:
        self.tip_to_hand = 0.04 # cm
        self.hand_width = 0.10 # cm
        self.wrist_to_hand = 0.04 # cm

        self._wrist = np.array([0, 0, -self.wrist_to_hand])
        self._hand = np.array([0, 0, 0])

        self._tip1_base = np.array([self.hand_width / 2, 0, 0])
        self._tip2_base = np.array([-self.hand_width / 2, 0, 0])

        self._tip1 = self._tip1_base + np.array([0, 0, self.tip_to_hand])
        self._tip2 = self._tip2_base + np.array([0, 0, self.tip_to_hand])

        self.points = np.array([self._wrist, self._hand, self._tip1_base, self._tip2_base, self._tip1, self._tip2])
        self.lines = [[0, 1], [1, 2], [1, 3], [2, 4], [3, 5]]
        self.colors = [[1, 0, 0] for i in range(len(self.lines))]

    def get_cylinder_mesh(self, radius:float = 0.01, color = None):
        if color is not None:
            self.colors = [color for i in range(len(self.lines))]
    
        line_mesh = LineMesh(self.points, self.lines, self.colors, radius=radius)
        return line_mesh.cylinder_segments
    
    def transform(self, T):
        """ Transform the gripper by a transformation matrix T """
        self.points = np.dot(T[:3, :3], self.points.T).T + T[:3, 3]

    def sample_collision_points(self, density = 10):
        collision_points = []

        for link in self.lines:
            start = self.points[link[0]]
            end = self.points[link[1]]
            direction = end - start
            distance = np.linalg.norm(direction)
            direction = direction / distance
            
            samples = np.arange(0, density + 1)
            samples = samples / density * distance
            points = start + samples[:, None] * direction
            collision_points.extend(points)

        return np.array(collision_points)

    def digit_to_base(self):

        return np.array([
            [0, 0, 1, -self.hand_width / 2], 
            [-1, 0, 0, 0], 
            [0, -1, 0, self.tip_to_hand],
            [0, 0, 0, 1]
        ])
    
    def sample_poses_from_mesh(self, mesh, n):
        """
        Sample n poses on the surface of the mesh, Everything is done in the base frame
        
        """

        # Create a scene and add the triangle mesh
        mesh_l = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh_l)

    def sample_poses(self, mesh, n, 
                 feasible_check:Callable = lambda x: True,
                 tsdf_query: Callable = lambda x: np.ones(x.shape[0]),
                 distance_along_normal:float = 0.03,
                 tsdf_threshold_lower:float = 0.01,
                 tsdf_threshold_upper:float = 0.05):
        """
        Sample n poses on the surface of the mesh, Everything is done in the base frame
        
        """

        # Create a scene and add the triangle mesh
        mesh_l = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh_l)  # we do not need the geometry ID for mesh

        # compute normal
        mesh.compute_triangle_normals()
        normals = np.asarray(mesh.triangle_normals)
        faces = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)

        # random select 100 faces
        # increase the number of faces to get more samples, when there is no touches
        select_face_num = n

        face_idx = np.arange(normals.shape[0])
        np.random.shuffle(face_idx)
        select_face = faces[face_idx[:select_face_num]]
        select_normals = normals[face_idx[:select_face_num]]

        # compute the center of the face
        center = np.mean(vertices[select_face], axis=1)

        # move along the normal direction
        center = center + select_normals * distance_along_normal
        normal_opposite = select_normals * -1
        center_opposite = center + normal_opposite * distance_along_normal
        
        # sample two sides on the surface
        center = np.vstack([center, center_opposite])
        select_normals = np.vstack([select_normals, normal_opposite])
        
        # tsdfs = np.ones_like(select_normals[:, 0]) * 0.03
        sample_coords = []
        for i in tqdm(range(select_face_num)):
            # construct the rotation matrix
            z_axis = select_normals[i] * -1
            dummy = np.cross(np.array([0, 0, -1]), z_axis)
            dummy2 = np.cross(np.array([-1, 0, 0]), z_axis)
            
            axis = dummy if np.linalg.norm(dummy) > np.linalg.norm(dummy2) else dummy2
            x_axis = np.cross(z_axis, axis)
            if x_axis[2] > 0:
                x_axis = -x_axis
            y_axis = np.cross(z_axis, x_axis)

            c_w = center[i]
            query_point = o3d.core.Tensor([[center[i, 0], center[i, 1], center[i, 2]]], dtype=o3d.core.Dtype.Float32)
            tsdf = scene.compute_distance(query_point) # tsdfs[i]
            # filter by tsdf value
            if tsdf <= tsdf_threshold_lower or \
                        tsdf > tsdf_threshold_upper:
                continue
            
            # Do svd to find the closest rotation matrix
            R = np.array([x_axis, y_axis, z_axis]).T
            u, s, vh = np.linalg.svd(R, full_matrices=True)
            s = np.ones((3, ))
            R = u @ (s[..., None] * vh)

            touch_poses = np.eye(4)
            touch_poses[:3, :3] = R
            touch_poses[:3, 3] = c_w

            # check collisions from the world frame
            # the points for collision check is to linear interpolation
            # between joints
            grip_m = Gripper()
            digit_to_base = grip_m.digit_to_base()
            base_to_world = touch_poses @ np.linalg.inv(digit_to_base) # gripper base to world
            points = grip_m.sample_collision_points()
            collision_points = (base_to_world[:3, :3] @ points.T + base_to_world[:3, [3]]).T
            query_points = o3d.core.Tensor(collision_points, dtype=o3d.core.Dtype.Float32)
            
            # check occupancy from the object body frame
            occupancy = scene.compute_occupancy(query_points)
            occ_np = occupancy.numpy()
            if np.any(occ_np > 0):
                continue

            tsdfs = tsdf_query(collision_points)
            if np.min(tsdfs) <= 0.:
                continue

            if feasible_check(base_to_world):
                sample_coords.append(base_to_world)

        return sample_coords
    
    @ staticmethod
    def visualize_grasp_poses(sample_coords, geoms = []):
        """
            Visualize the grasp poses
                sample_coords: list of 4x4 transformation matrix
                        from gripper coord to world coord
        """
        for transform in sample_coords:
            grip_m = Gripper()
            grip_m.transform(transform)
            geoms.extend(grip_m.get_cylinder_mesh(radius = 0.002))
            
            sample_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0., 0, 0])
            # show the touch pose
            # sample_coord.transform(transform)
            # show the tool frame
            sample_coord.transform(transform)
            geoms.append(sample_coord)

        original_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10, origin=[0., 0, 0])
        geoms.append(original_frame)
        o3d.visualization.draw_geometries(geoms)
    
    @staticmethod
    def load_existing_poses(path, object_pose, geoms=[]):
        """
        Load existing poses from a file
        
        """
        poses = np.loadtxt(path, delimiter=',')

        sample_coords = []
        for p in poses:
            coord = np.eye(4)
            coord[:3, 3] = p[:3]

            quat = sciR.from_quat(p[3:])
            coord[:3, :3] = quat.as_matrix()

            coord = np.linalg.inv(object_pose) @ coord
            sample_coords.append(coord)

        Gripper.visualize_grasp_poses(object_pose, sample_coords, geoms=geoms)
        
        model_base_p = []
        for p in sample_coords:
            model_base_p.append(convert_pose_to_pos_quat(p))
        model_base_p = np.array(model_base_p)
        np.savetxt('can_base.txt', model_base_p, delimiter=',', fmt='%.4f')
        return sample_coords
    
    @staticmethod
    def load_candidate(path):
        poses = np.loadtxt(path, delimiter=',')

        sample_coords = []
        for p in poses:
            coord = np.eye(4)
            coord[:3, 3] = p[:3]

            quat = sciR.from_quat(p[3:])
            coord[:3, :3] = quat.as_matrix()
            sample_coords.append(coord)

        return sample_coords

if __name__ == "__main__":
    import argparse 

    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default="data")
    args.add_argument("--mesh", type=str, default="../../gaussian_splatting/weights/CAD_model/mustard.stl")
    opt = args.parse_args()

    tsdf_fusion = O3DTsdfIntegration()
    tsdf_fusion.build_from_data_folder(opt.data_dir)
    world_mesh = tsdf_fusion.get_mesh()

    mesh = o3d.io.read_triangle_mesh(opt.mesh)

    # estimate the object pose
    from gaussian_splatting_py.foundation_pose_interface import FoundationPoseInterface
    foundation_pose_interface = FoundationPoseInterface(opt.mesh, 
                                        "/home/user/Documents/ActiveTouch/src/gaussian_splatting/weights/2024-01-11-20-02-45", 
                                        "/home/user/Documents/ActiveTouch/src/gaussian_splatting/weights/2023-10-28-18-33-37")
    data_pack = foundation_pose_interface.infer_from_data_dir(opt.data_dir)
    # select the first 
    object_pose = None
    for result in data_pack:
      # select z upward
      if result["object_pose"][2, 2] > 0 :
        object_pose = result["object_pose"]
        break
    
    if object_pose is not None:
        # def tsdf_check(points):
        #     collision_points = (object_pose[:3, :3] @ points.T + object_pose[:3, [3]]).T
        #     return tsdf_fusion.get_tsdfs(collision_points)

        # # sample poses near the gripper
        # gripper = Gripper()
        # sample_coords = gripper.sample_poses(mesh, 50, tsdf_query = tsdf_check)

        # # visualization 
        original_mesh = mesh
        mesh_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10, origin=[0., 0, 0])
        original_mesh.transform(object_pose)
        mesh_coord.transform(object_pose)
        # Gripper.visualize_grasp_poses(object_pose, sample_coords, [world_mesh, original_mesh])

        Gripper.load_existing_poses("/home/user/Documents/ActiveTouch/mustard.txt", object_pose, [mesh_coord, original_mesh])