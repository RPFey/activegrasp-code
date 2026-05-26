import argparse
import torch
import open3d as o3d   
import matplotlib.pyplot as plt

from pytorch3d.ops import ball_query

from gaussian_splatting_py.grasp.base import GraspEstimator
from gaussian_splatting_py.grasp.contact_graspnet import ContactGraspNet, CONTACTGRASP_CONFIG
from gaussian_splatting_py.splatting import Runner
from gaussian_splatting_py.datasets import KinovaDataset
from gaussian_splatting_py.splatting import Config as GSConfig
from gaussian_splatting_py.uncern_render import equal_hist

# contact graspnet imports
from contact_graspnet_pytorch.config_utils import load_config

class FisherGrasp:
    def __init__(self, splatting:Runner, grasp_estimator:ContactGraspNet):
        self.splatting = splatting
        self.grasp_estimator = grasp_estimator

    def predict_grasp_pose(self, **obs):
        pass

    def estimate_uncertainty(self, **obs):
        # # query the splatting uncertainty score
        # info = [(p["camtoworld"], p["K"]) for p in self.splatting.trainset]
        # c2ws, Ks = [i[0] for i in info], info[0][1]
        # height, width = self.splatting.trainset[0]["image"].shape[:2]
        # H_train = self.splatting.computeHtrain(c2ws, Ks, height, width)
        # H_train_inv = torch.reciprocal(H_train + 1e-6)
        # uncertainty = torch.sum(H_train_inv, dim=1)

        # organize the rgbd data for the grasp estimator
        dataset = []
        with torch.no_grad():
            for i in range(len(self.splatting.trainset)):
                data = self.splatting.trainset[i]
                renders = self.splatting.render_at_pose(data["camtoworld"], data["K"], width, height, H_train_inv)

                uncern = renders["uncern"].squeeze(0).cpu().numpy()
                uncern = equal_hist(uncern)
                uncern_color = plt.cm.viridis(uncern)[:, :, :3]
                uncern_color = uncern_color * 255
                uncern_color = torch.from_numpy(uncern_color)
                
                dataset.append({
                    "mask": data["mask"].cpu(),
                    "image": uncern_color,
                    "depths": renders["depth"].squeeze(0).cpu(), # use the rendered depth
                    "K": data["K"].cpu(),
                    "camtoworld": data["camtoworld"].cpu(),
                    # "uncertainty": renders["uncern"].squeeze(0).cpu()
                })

        grasp_poses, meta = self.grasp_estimator.predict_grasp_pose(dataset=dataset, viz=True)
        pc_full, pc_colors, fuse_pc_segments, base_transform = self.grasp_estimator.extract_pcd_batch(dataset)
        contact_pts = torch.from_numpy(meta["contact_pts"]).to(uncertainty.device).float()
        
        ball_radius = 0.05
        NN_number = 20
        means = self.splatting.splats["means"].data
        dists, index, nn = ball_query(contact_pts.unsqueeze(0), means.unsqueeze(0), None, None,
                             NN_number, ball_radius, False)

        # visualization
        index = index[0].reshape(-1)
        gaussian_index = torch.unique(index)

        # IDEA # 1 -- perturb the data using uncertainty 

        # IDEA # 2 

        return gaussian_index


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # args.add_argument("--config", type=str, default="config.yaml")
    args.add_argument("--gs_dir", type=str, default="/home/user/Documents/ActiveTouch/src/gaussian_splatting/gaussian_splatting_py/results/box/ckpts/ckpt_29999_rank0.pt")
    args.add_argument("--grasp_dir", type=str, default="/home/user/Documents/contact_graspnet_pytorch/checkpoints/contact_graspnet")
    args.add_argument("--datadir", type=str, default="/home/user/Documents/data/2025-01-07-16-56-14")
    args = args.parse_args()

    cfg = GSConfig()
    dataset = KinovaDataset(args.datadir)

    # set running steps
    cfg.max_steps = 8_000
    cfg.init_scale = 0.005
    cfg.init_type = "romatch"
    cfg.sh_degree = 0
    cfg.depth_loss = len(dataset.depths) > 0

    import pdb; pdb.set_trace()
    # load the gaussian splatting model here 
    splatting = Runner(0, 0, 1, cfg, train_dataset=dataset, parser=None)
    ckpts = torch.load(args.gs_dir, map_location=splatting.device, weights_only=True)
    for k in splatting.splats.keys():
        splatting.splats[k].data = ckpts["splats"][k]

    # load the grasp estimator model here
    global_config = load_config(args.grasp_dir, batch_size=5)
    # "ckpt_dir", "z_range", "local_regions", "filter_grasps", "skip_border_objects", "forward_passes"
    FLAGS = CONTACTGRASP_CONFIG(args.grasp_dir, [0.1, 1.1], True, True, False, 1)
    grasp_estimator = ContactGraspNet(
        global_config, FLAGS
    )

    # FisherGRasp
    fisher_grasp = FisherGrasp(splatting, grasp_estimator)
    fisher_grasp.estimate_uncertainty(data_path=args.datadir)

