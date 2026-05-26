import sksparse.cholmod as skch

import numpy as np
import torch
import os, os.path as osp
import open3d as o3d
import glob
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from multiprocessing import Process, Pipe, Queue
import time
import theseus as th
from theseus import SO3

import contact_graspnet_pytorch
from contact_graspnet_pytorch.config_utils import load_config

from gaussian_splatting_py.grasp.contact_graspnet import ContactGraspNet, CONTACTGRASP_CONFIG
from gaussian_splatting_py.grasp.se3diffusion import SE3Diffusion
from gaussian_splatting_py.grasp.ActiveNGF import ActiveNGFEstimator
from gaussian_splatting_py.grasp.vgn import VGNGrasp
from gaussian_splatting_py.grasp.ACE import ACEGraspEstimator

from franka_env.grasp_generator import ClutterRemovalSim, render_images, evaluate_grasp_pose, Label
from franka_env.btsim import Rotation, Transform, CameraIntrinsic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_grasps(num_object, seed, child_conn, queue, save_dir, nun_views = 8):
    sim = ClutterRemovalSim("pile", gui=False, seed=seed)
    sim.reset(num_object)
    sim.save_state()
    
    # n = 8 # np.random.randint(MAX_VIEWPOINT_COUNT) + 1
    depth_imgs, extrinsics, segs = render_images(sim, nun_views)
    
    # create point cloud using open3d
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=depth_imgs[0].shape[1],
        height=depth_imgs[0].shape[0],
        fx=sim.camera.intrinsic.fx,
        fy=sim.camera.intrinsic.fy,
        cx=sim.camera.intrinsic.cx,
        cy=sim.camera.intrinsic.cy
    )
    
    # find all visible indices
    all_segs = np.stack(segs, axis=0)
    unique_ids = np.unique(all_segs)
    print("Unique IDs: ", unique_ids)
    
    # hold ids >= 2 and all other ids are set to 0
    unique_ids = np.where(unique_ids < 2, 0, unique_ids)
    all_segs = np.where(all_segs < 2, 0, all_segs)
    
    data_packs = dict(
        masks = all_segs,
        depths = np.stack(depth_imgs, axis = 0),
        K = np.array([
                [sim.camera.intrinsic.fx, 0, sim.camera.intrinsic.cx],
                [0, sim.camera.intrinsic.fy, sim.camera.intrinsic.cy],
                [0, 0, 1],
            ]),
        camtoworlds = np.stack(
            [np.linalg.inv(Transform.from_list(ext).as_matrix()) for ext in extrinsics], axis=0
        )
    )
    
    queue.put(data_packs)
    child_conn.send(("input", ))
    
    # wait for the parent process to finish
    child_conn.poll(None)
    
    # receive all the grasps for each category
    # print(f"receive grasps for {num_object} objects for seed {seed}")
    all_grasps = child_conn.recv()
    
    cate_results = {}
    for unique_id, grasp_poses in all_grasps.items():    
        # evaluate 
        if len(grasp_poses) > 0:
            results = []
            for execute_grasp in grasp_poses:
                sim.restore_state()
                outcome, width = evaluate_grasp_pose(sim, execute_grasp, target_id=int(unique_id))
                if outcome == Label.SUCCESS:
                    # print("Grasp success")
                    results.append(1)
                else:
                    results.append(0)
            
            results = np.array(results, dtype=np.uint8)
            cate_results[unique_id] = results
            
            # save results
            if save_dir is not None and np.any(results > 0):
                grasp_poses[:, :3, 3] -= 0.04 * grasp_poses[:, :3, 2]
                orientation = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                grasp_poses = grasp_poses @ np.linalg.inv(orientation)[None, :, :]
                
                all_pts = []
                seg_ids = []
                for i, pcd in seg_pcds.items():
                    all_pts.append(pcd)
                    seg_ids.append(np.ones(len(pcd), dtype=np.int32) * i)
                    
                all_pts = np.concatenate(all_pts, axis=0)
                seg_ids = np.concatenate(seg_ids, axis=0)
                target_pts = all_pts[seg_ids == unique_id]
                scene_pts = all_pts[seg_ids != unique_id]
                target_pcd = o3d.geometry.PointCloud(); target_pcd.points = o3d.utility.Vector3dVector(target_pts)
                scene_pcd = o3d.geometry.PointCloud(); scene_pcd.points = o3d.utility.Vector3dVector(scene_pts)

                complete_pc, _, \
                    target_index, target_mean = \
                        BulletEvaluator.process_point_clouds(target_pcd, scene_pcd, 2048, 2048)
                
                np.savez(
                    os.path.join(save_dir, f"seed{seed}-obj{num_object}-t{unique_id}.npz"),
                    scene_pts = complete_pc, # + target_mean,
                    target_index = target_index,
                    grasps = grasp_poses[results > 0, :, :],
                    F_grasps = grasp_poses[results == 0, :, :],
                )
        
    child_conn.send(("result", cate_results))
    return 

class BulletEvaluator:
    def __init__(self, policy, save_dir, num_grasps = 512, save_data = True, viz = False):
        self.num_grasps = num_grasps
        self.save_dir = save_dir
        self.save_data = save_data
        self.viz = viz
        
        if save_data:
            os.makedirs(save_dir, exist_ok=True)
            
        if policy == "contact_graspnet":
            # "ckpt_dir", "z_range", "local_regions", "filter_grasps", "skip_border_objects", "forward_passes"
            # import contact_graspnet_pytorch
            package_dir = osp.dirname(contact_graspnet_pytorch.__file__)
            ckpt_dir = osp.join(package_dir, "..", "checkpoints", "contact_graspnet")
            global_config = load_config(ckpt_dir, batch_size=5)
            
            FLAGS = CONTACTGRASP_CONFIG(ckpt_dir, [0.1, 1.1], True, True, False, 5)
            self.policy = ContactGraspNet(
                global_config, FLAGS
            )
            
        elif policy in ["se3diff", "se3diff_scene", "se3diff_dual"]:
            # "ckpt_dir", "z_range", "local_regions", "filter_grasps", "skip_border_objects", "forward_passes"
            FLAGS = CONTACTGRASP_CONFIG("", [0.1, 1.1], True, True, False, 5)
            model_params_dict = dict(se3diff="partial_grasp_dif", 
                                    se3diff_scene="multiobject_scene_graspdif", 
                                    se3diff_dual="multiobject_scene_graspdif_ap")
            model_name = model_params_dict[policy]
            self.policy = SE3Diffusion(FLAGS, model_name, num_grasps)

        elif policy == 'ACE':
            self.policy = ACEGraspEstimator("ACE")
            
        elif policy == 'ActiveNGF':
            self.policy = ActiveNGFEstimator("ActiveNGF", num_grasps)
    
    def evaluate_model(self, total_timestep, num_objects, seed=42, viz=False):
        """ Evaluate the model on the given number of objects
        
        Args:
            total_timestep: the current number of timesteps for SE3 Diff
            num_objects: the number of objects to evaluate on
                int or List[int]
            seed: the random seed for the simulation
            viz: whether to visualize the grasps
        Returns:
            total_trials: the total number of trials for each category
                List[Int] total trials for each instance
            success_trials: the number of successful trials for each category
                List[Int] success successes for each instance
        """
        if isinstance(num_objects, int):
            num_objects = [num_objects]
        
        save_dir = os.path.join(self.save_dir, "{:06d}".format(total_timestep))
        if self.save_data:
            os.makedirs(save_dir, exist_ok=True)
        
        # record for each category given the current number of objects
        total_trials = []
        success_trials = []
        
        for num_object in num_objects:
            
            sim = ClutterRemovalSim("pile", gui=viz, seed=seed)
            sim.reset(num_object)
            sim.save_state()
            
            # render synthetic depth images
            # MAX_VIEWPOINT_COUNT = 4
            n = 8 # np.random.randint(MAX_VIEWPOINT_COUNT) + 1
            depth_imgs, extrinsics, segs = render_images(sim, n)
            
            # create point cloud using open3d
            intrinsics = o3d.camera.PinholeCameraIntrinsic(
                width=depth_imgs[0].shape[1],
                height=depth_imgs[0].shape[0],
                fx=sim.camera.intrinsic.fx,
                fy=sim.camera.intrinsic.fy,
                cx=sim.camera.intrinsic.cx,
                cy=sim.camera.intrinsic.cy
            )
            
            # find all visible indices
            all_segs = np.stack(segs, axis=0)
            unique_ids = np.unique(all_segs)
            print("Unique IDs: ", unique_ids)
            
            for target in unique_ids:
                total_trial = 0
                success_trial = 0
                
                # skip desk and box
                if target < 2:
                    continue
                
                # scene point cloud
                target_pcd = o3d.geometry.PointCloud()
                scene_pcd = o3d.geometry.PointCloud()
                    
                data_packs = []
                for depth, ext, seg in zip(depth_imgs, extrinsics, segs):
                    # 1 - red; 0 - blue
                    seg_color = np.stack(
                        [(seg == target).astype(np.uint8) * 255, 
                         np.zeros_like(seg, dtype=np.uint8), 
                         (seg != target).astype(np.uint8) * 255], axis=-1
                    )
                    
                    data_pack = dict(
                        mask = (seg == target).astype(np.uint8),
                        depths = depth,
                        camtoworld = np.linalg.inv(Transform.from_list(ext).as_matrix()),
                        image = seg_color,
                        K = np.array([
                            [sim.camera.intrinsic.fx, 0, sim.camera.intrinsic.cx],
                            [0, sim.camera.intrinsic.fy, sim.camera.intrinsic.cy],
                            [0, 0, 1],
                        ]),
                    )
                    data_packs.append(data_pack)
                    
                grasp_poses, meta = self.policy.predict_grasp_pose(dataset=data_packs, viz=self.viz)
                
                # evaluate 
                if len(grasp_poses) > 0:
                    results = []
                    for grasp in grasp_poses:
                        execute_grasp = grasp.copy()
                        # inertial offset 
                        execute_grasp[:3, 3] += 0.04 * execute_grasp[:3, 2]
                        execute_grasp = execute_grasp @ np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                        sim.restore_state()
                        outcome, width = evaluate_grasp_pose(sim, execute_grasp, target_id=int(target))
                        if outcome == Label.SUCCESS:
                            print("Grasp success")
                            results.append(1)
                        else:
                            results.append(0)
                    
                    results = np.array(results)
                    total_trial += len(results)
                    success_trial += np.sum(results).item()
                    
                    # one success is enough
                    if np.any(results > 0):
                        # only save successful grasps
                        success_grasps = grasp_poses[results > 0]
                        failure_grasps = grasp_poses[results == 0]
                        
                        if self.save_data:
                            np.savez(
                                os.path.join(save_dir, f"seed{seed}-obj{num_object}-t{target}.npz"),
                                scene_pts = complete_pc,
                                target_index = target_index,
                                grasps = success_grasps,
                                F_grasps = failure_grasps,
                            )
                
                success_trials.append(success_trial)
                total_trials.append(total_trial)
        
        return total_trials, success_trials 
    
    @staticmethod
    def process_point_clouds(target_pcd, scene_pcd, 
                                dnum_scene_points=-1, dnum_target_pts=512):
        """
        Process Point Cloud
        """
        # normalize & scale
        num_target_pts = np.asarray(target_pcd.points).shape[0]
        desired_num_target_pts = dnum_target_pts if dnum_scene_points > 0 else 1024
        if num_target_pts > desired_num_target_pts:
            target_pcd = target_pcd.farthest_point_down_sample(desired_num_target_pts)
        elif num_target_pts < 128:
            print("Too few points for target ... Continue ... ")
            return None, None, None, None
        
        # crop scene point cloud
        extent_scale = 1.
        target_extent = target_pcd.get_max_bound() - target_pcd.get_min_bound()
        target_center = (np.asarray(target_pcd.points)).mean(axis=0)
        croped_scene_num_pts = 0 
        while croped_scene_num_pts < 256:
            extent_scale += 0.25
            crop_box = o3d.geometry.AxisAlignedBoundingBox(
                min_bound = target_center - extent_scale * target_extent,
                max_bound = target_center + extent_scale* target_extent
            )
            croped_scene_pcd = scene_pcd.crop(crop_box)
            croped_scene_num_pts = len(croped_scene_pcd.points)
        
        scene_pcd = croped_scene_pcd
        num_scene_points = 1024 if dnum_scene_points <= 0 else dnum_scene_points
        if len(scene_pcd.points) > num_scene_points:
            scene_pcd = scene_pcd.farthest_point_down_sample(num_scene_points)
            
        target_mean = np.mean(np.asarray(target_pcd.points), axis=0)
        complete_pc = np.concatenate([np.asarray(target_pcd.points), np.asarray(scene_pcd.points)], axis=0)
        target_index = np.concatenate([np.ones(len(target_pcd.points)), np.zeros(len(scene_pcd.points))], axis=0)
        complete_pc_norm = (complete_pc - target_mean) * 8
        
        return complete_pc, complete_pc_norm, target_index, target_mean
    
    def _run_model(self, datapacks):
        all_segs = datapacks['masks']
        unique_ids = np.unique(all_segs)
        
        grasp_poses_ids, scores_ids = {}, {}
        
        for target in unique_ids:
            if target < 2:
                continue
            
            data_packs = []
            target_segs = (all_segs == target).astype(np.uint8)
            for depth, ext, seg in zip(datapacks['depths'], datapacks['camtoworlds'], target_segs):
                
                seg_color = np.stack(
                    [(seg == 1).astype(np.uint8) * 255, 
                        np.zeros_like(seg, dtype=np.uint8), 
                        (seg != 1).astype(np.uint8) * 255], axis=-1
                )
                
                data_packs.append(dict(
                    mask = seg.astype(np.uint8),
                    depths = depth,
                    camtoworld = ext,
                    image = seg_color,
                    K = datapacks['K'],
                ))
                
            # run prediction
            grasp_poses, meta = self.policy.predict_grasp_pose(dataset=data_packs, viz=self.viz)

            if len(grasp_poses) == 0:
                continue

            # do some adjustment for bullet
            orientation = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            grasp_poses[:, :3, 3] += 0.04 * grasp_poses[:, :3, 2]
            grasp_poses = grasp_poses @ orientation[None, :, :]
            

            grasp_poses_ids[target] = grasp_poses
            scores_ids[target] = meta['scores']
            
        return grasp_poses_ids, scores_ids
    
    
    def run_multiple_process(self, num_objects, seeds, num_processes=4, num_views=8):
        """ Run the evaluation in multiple processes """
        
        # get grid of num_objects and seeds
        object_seed_pairs = [(n, s) for n in num_objects for s in seeds]
        num_pairs = len(object_seed_pairs)
        
        # create pipes for communication
        # the key is experiment id
        process_pool, parent_conns, child_conns, queues = {}, {}, {}, {}
        gts, preds = {}, {}
        
        def __wait_process():
            finish_ids = []
                
            # check all the connections
            for i, conn in parent_conns.items():
                if conn.poll(0.1):
                    # receive the point clouds from the child process
                    data = conn.recv()
                    
                    if data[0] == 'input':
                        # run something
                        dataset = queues[i].get()
                        
                        # grasp_poses, meta = self.policy.predict_grasp_pose(dataset=dataset, viz=True)
                        print(f"Run Prediction for {object_seed_pairs[i]}")
                        grasps_ids, scores_ids = self._run_model(dataset)
                        preds[i] = scores_ids
                        
                        # send the point clouds to the child process
                        conn.send(grasps_ids)
                    else:
                        gts[i] = data[1]
                        process_pool[i].join()
                        finish_ids.append(i)
            
            for i in finish_ids:
                # remove the connection
                # close the connection and remove the process
                parent_conns[i].close()
                child_conns[i].close()
                del parent_conns[i]
                del child_conns[i]
                del process_pool[i]
                del queues[i]
                    
            time.sleep(0.1)
        
        idx = 0 
        while idx < num_pairs:
            pair = object_seed_pairs[idx]
            # get the pairs for this process
            n, s = pair

            if len(process_pool) < num_processes:
                print(f"Process {pair}")
                parent_conn, child_conn = Pipe()
                q = Queue()
                
                # create a new process
                save_dir = self.save_dir if self.save_data else None
                p = Process(target=evaluate_grasps, args=(n, s, child_conn, q, save_dir, num_views))
                p.start()
                process_pool[idx] = p
                parent_conns[idx] = parent_conn
                child_conns[idx] = child_conn
                queues[idx] = q
                idx += 1
                
            else:
                __wait_process()
        
        while len(process_pool) > 0:
            __wait_process()

        return gts, preds, object_seed_pairs
    
if __name__ == '__main__':
    import argparse
    mp.set_start_method('spawn')
    
    args = argparse.ArgumentParser()
    args.add_argument("--save_dir", type=str, default="/mnt/kostas-graid/datasets/boshu/grasp_buffer")
    args.add_argument("--policy", type=str, default="contact_graspnet", 
                      choices=["se3diff", "se3diff_scene", "se3diff_dual", "contact_graspnet", "ACE", "ActiveNGF"])
    args.add_argument("--save_data", action='store_true', default=False, help='save the grasp data to disk')
    args.add_argument("--viz", action='store_true', default=False, help='visualize the grasps')
    args.add_argument("--num_objects", type=int, default=2, help='number of objects to evaluate on')
    args.add_argument("--num_seeds", type=int, default=2, help='number of seeds to test on')
    args.add_argument("--num_processes", type=int, default=1, help='number of seeds to test on')
    args.add_argument("--num_views", type=int, default=8, help='number of views for grasping')
    args.add_argument("--num_grasps", type=int, default=512, help='number of grasps to execute')
    opt = args.parse_args()

    save_dir = opt.save_dir
    evaluator = BulletEvaluator(opt.policy, save_dir, 
                                 save_data=opt.save_data, num_grasps=opt.num_grasps, viz = opt.viz)
    
    assert opt.num_objects > 1, "Number of objects must be greater than 1"
    
    total_exps = []
    success_exps = []
    
    if opt.num_processes > 1:
        gts, preds, object_seed_pairs = evaluator.run_multiple_process(
                                    list(range(2, opt.num_objects + 1)), 
                                    list(range(opt.num_seeds)), num_processes=opt.num_processes, num_views=opt.num_views
        )
    
        
        stats = {}
        total_pred = []
        total_gt = []
        total_alphas = []
        for exp_id, pair in enumerate(object_seed_pairs):
            n, s = pair
            if exp_id in gts:
                cates_gt = gts[exp_id]
                
                # collect success rate
                if n not in stats:
                    stats[n] = [0, 0]
                for results in cates_gt.values():
                    stats[n][0] += 1
                    stats[n][1] += 1 if np.any(results > 0) else 0 
                      
                cates_pred = preds[exp_id]
                for unique_id in cates_pred:
                    p, g = cates_pred[unique_id], cates_gt[unique_id]
                    total_pred.append(p)
                    total_gt.append(g)     

        for k, v in stats.items():
            print(f"No. objs {k}, total trial {v[0]}, success {v[1]}")
            
        import matplotlib.pyplot as plt
        total_pred = np.concatenate(total_pred, axis=0)
        total_gt = np.concatenate(total_gt, axis=0)
        # plt.figure()
        # plt.scatter(total_gt, total_pred, marker='+')
        # plt.xlabel("GT")
        # plt.ylabel("Pred")
        # plt.savefig("calib.png")
        
        # get quantiles 0.5
        n_bins = 15
        pred_q = np.linspace(0, 1, n_bins + 1)
        sr = []
        for i in range(n_bins):
            sr.append(np.mean(total_gt[(total_pred >= pred_q[i]) & (total_pred < pred_q[i + 1])]))
        
        expname = opt.policy
        plt.figure()
        plt.plot(np.arange(n_bins) / n_bins, sr)
        ax = plt.gca()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Quantile")
        ax.set_ylabel("SR")
        plt.savefig(f"{expname}-calib.png")
        
        np.savez(f"{expname}-calib.npz", 
                 xaxis=np.arange(n_bins) / n_bins, yaxis=sr, total_pred=total_pred, total_gt=total_gt)    
            
    else:
        evaluator.evaluate_model(
            1000, opt.num_objects, opt.num_seeds, opt.viz
        )