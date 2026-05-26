import copy
import json
import logging
import math
import os
import time
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import cv2
import gaussian_splatting_py.datasets as datasets
import imageio
import matplotlib.pyplot as plt
import nerfview  # type: ignore
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from einops import rearrange
from fused_ssim import fused_ssim
from gaussian_splatting_py.datasets.traj import (generate_ellipse_path_z,
                                                 generate_interpolated_path,
                                                 generate_spiral_path)
from gaussian_splatting_py.lib_bilagrid import (BilateralGrid, color_correct,
                                                slice, total_variation_loss)
from gaussian_splatting_py.tools.uncern_render import (equal_hist,
                                                 rasterization_fisher_wrapper)
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from PIL import Image, ImageTk
from rich.logging import RichHandler
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import (PeakSignalNoiseRatio,
                                StructuralSimilarityIndexMeasure)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never

from gaussian_splatting_py.foundation.romatch_utils import (
    AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed,
    unproject_rgbd)

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("rich")

# This is under-dev by  gsplat, 
try:
    from gsplat.optimizers import SelectiveAdam  # type: ignore
    SelectiveAdamInstall = True
except ModuleNotFoundError:
    SelectiveAdamInstall = False


DEBUG_VIZ = os.environ.get("DEBUG_VIZ", "0") == "1"

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "/home/user/Documents/data/colmap_data"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "random"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1.

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    spatial_lr_scale: float = 1.0

    skip_monodepth: bool = False

    isotropic: bool = False

    save_init_pcd: bool = False

    semantic_mode: bool = False
    sem_weight: float = 0.1 # to balance the scale of semantic loss with the others

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)

def create_splats_with_optimizers(
    parser: datasets.ColmapParser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    spatial_lr_scale: float = 1.0,
    isotropic: bool = False,
    semantic_mode: bool = False,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    # TODO Add Depth Initialization Here
    if init_type in ["sfm", "romatch", "rgbd"]:
        points = torch.from_numpy(parser.points).float()

        if semantic_mode:
            rgbs = torch.from_numpy(parser.points_rgb).float()[..., :3]
            semantic_labels = torch.from_numpy(parser.points_rgb).float()[..., -1:]
        else:
            rgbs = torch.from_numpy(parser.points_rgb).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)

    scale_dim = 3 if not isotropic else 1
    # scales = torch.log(torch.ones(points.shape[0]).to(points) * init_scale).unsqueeze(-1).repeat(1, scale_dim)  # [N, 3]
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, scale_dim)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale * spatial_lr_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3 * spatial_lr_scale),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if semantic_mode:
        assert sh_degree == 0, "SH degree should be 0 for semantic mode"
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

        semantics = torch.logit(semantic_labels) # [N, 1]
        params.append(("semantics", torch.nn.Parameter(semantics), 2.5e-3)) # [N, 1]

    elif feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam and SelectiveAdamInstall:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config, 
        parser = None, train_dataset = None
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        self.parser = parser
        if train_dataset is None:
            # load the dataset
            self.load_dataset(cfg)
        else:
            self.trainset = train_dataset
            self.scene_scale = cfg.global_scale

        if cfg.init_type == "sfm":
            assert hasattr(self.parser, "points") and \
                        hasattr(self.parser, "points_rgb"), "Need to load the dataset first."
        else:
            self.parser = self.trainset.get_parser(cfg.init_type, init_from_touch=False, add_semantics=cfg.semantic_mode)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        if cfg.save_init_pcd and hasattr(self.parser, "points") and hasattr(self.parser, "points_rgb"):
            logging.info(f"Saving initial points to {cfg.result_dir}/init-points.txt")
            np.savetxt(f"{cfg.result_dir}/init-points.txt", np.concatenate([self.parser.points, self.parser.points_rgb], axis=1))
            logging.info("Saving done.")
        logging.info("Initializing model...")
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            spatial_lr_scale=cfg.spatial_lr_scale,
            isotropic=cfg.isotropic,
            semantic_mode=cfg.semantic_mode,
        )
        logging.info(f"Model initialized. Number of GS: {len(self.splats['means'])}")

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        self.global_tic = time.time()

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )
            
            if cfg.semantic_mode:
                self.sem_server = viser.ViserServer(port=cfg.port + 1, verbose=False)
                self.sem_viewer = nerfview.Viewer(
                    server=self.sem_server,
                    render_fn=self._sem_viewer_render_fn,
                    mode="training",
                )

    def load_dataset(self, cfg):
        """ Load the Dataset """
        # Load data: Training data should contain initial points and colors.

        if os.path.exists(f"{cfg.data_dir}/transforms.json"):
            # load the Kinova Collected Dataset
            self.trainset = datasets.KinovaDataset(cfg.data_dir,
                                                   skip_monodepth=cfg.skip_monodepth)
            self.valset = self.trainset
            self.scene_scale = 1. * cfg.global_scale
        else:
            # load the Colmap Dataset
            self.parser = datasets.ColmapParser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
            )
            self.trainset = datasets.ColmapDataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
            )
            self.valset = datasets.ColmapDataset(self.parser, split="val")
            self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale

        print("Scene scale:", self.scene_scale)

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """
        Render splats to an image.

        Args:
            camtoworlds: Camera-to-world matrices. [C, 4, 4]
            Ks: Camera intrinsics. [C, 3, 3]
            width: Image width.
            height: Image height.
            masks: Masks to render. [C, H, W]
            **kwargs: Additional arguments for rasterization.
        """
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]

        if self.cfg.isotropic:
            scales = torch.exp(self.splats["scales"].repeat(1, 3))  # [N, 3]
        else:
            scales = torch.exp(self.splats["scales"])  # [N, 3]

        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        elif self.cfg.semantic_mode:
            colors = self.splats["colors"]
            semantics = self.splats["semantics"]    
            colors = torch.concat([semantics, colors], dim=-1)  # [N, 4]
            colors = torch.sigmoid(colors) # map from logits to [0, 1]
            kwargs["sh_degree"] = None
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        
        if self.cfg.semantic_mode:
            semantics = render_colors[..., 0:1]  # [H, W, 1]
            render_colors = render_colors[..., 1:]  # [H, W, 3]
            info["semantics"] = semantics
        return render_colors, render_alphas, info

    def train(self, begin_step: int = 0, evaluate: bool = True):
        """ Train the model 
        
        Args:
            begin_step: The step to start training from.
            evaluate: Whether to evaluate the model after training.
        """
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = begin_step

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        # preload all the data
        data_packs = [self.trainset[i] for i in range(len(self.trainset))]

        # TODO Use multiple workers for data loading in ROS. 
        # Bugs here for ROS
        # trainloader = torch.utils.data.DataLoader(
        #     self.trainset,
        #     batch_size=cfg.batch_size,
        #     shuffle=True,
        #     num_workers=1
        #     # pin_memory=True
        # )
        # trainloader_iter = iter(trainloader)

        # Training loop.
        self.global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                # NOTE: the order of lock might be important here
                # while self.sem_viewer.state.status == "paused":
                #     time.sleep(0.01)
                # self.sem_viewer.lock.acquire()
                tic = time.time()

            # try:
            #     data = next(trainloader_iter)
            # except StopIteration:
            #     trainloader_iter = iter(trainloader)
            #     data = next(trainloader_iter)
            
            data = copy.deepcopy(data_packs[step % len(data_packs)])
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.unsqueeze(0)
                else:
                    data[k] = torch.tensor(v).to(device).unsqueeze(0)   

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)   # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = (data["image"].to(device) / 255.0)   # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = torch.tensor(data["image_id"]).to(device)
            # masks = data["mask"].to(device).bool() if "mask" in data else None  # [1, H, W]
            masks = None 
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)   # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)
            
            if step == 0:
                import matplotlib.pyplot as plt
                import seaborn as sns
                colors_np = colors[0].detach().cpu().numpy()
                colors_np = np.clip(colors_np, a_min=0, a_max=1)

                plt.imsave(f"{self.cfg.result_dir}/init-render.png", colors_np)
                plt.imsave(f"{self.cfg.result_dir}/init-gt.png", pixels[0].detach().cpu().numpy())
                if cfg.depth_loss:
                    sns.heatmap(depths[0].detach().cpu().numpy().squeeze())
                    plt.savefig(f"{self.cfg.result_dir}/init-depth.png")
                    plt.close()
            
            if DEBUG_VIZ:
                breakpoint()

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            if "valid_rgb" in data: # assuming we don't mix colmap with kinova data
                loss = data["valid_rgb"].to(loss) * loss

            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                # disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                # disp_gt = torch.where(depths_gt > 0.0, 1.0 / depths_gt, torch.zeros_like(depths_gt))  # [1, M]
                
                mask = depths_gt > 0.05 # valid depth larger than 5cm
                depthloss = F.l1_loss(depths_gt * mask, depths * mask) * self.scene_scale 
                depthloss = depthloss  * data['valid_depth'].to(depthloss)
                loss += depthloss * cfg.depth_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss = (
                    loss
                    + cfg.opacity_reg
                    * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )
            if cfg.scale_reg > 0.0:
                loss = (
                    loss
                    + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                )
            
            if cfg.semantic_mode:
                semantic_render = info["semantics"]
                if step < 2:
                    sns.heatmap(semantic_render[0, ..., 0].detach().cpu().numpy(), square=True)
                    plt.savefig(f"{self.cfg.result_dir}/semantic-render.png")
                    plt.close()
                    
                    sns.heatmap(data["mask"][0].detach().cpu().numpy(), square=True)
                    plt.savefig(f"{self.cfg.result_dir}/semantic-gt.png")
                    plt.close()
                semantic_loss = F.binary_cross_entropy(semantic_render, rearrange(data["mask"].to(semantic_render), "b h w -> b h w 1"), reduce="mean")
                loss = (
                    loss + cfg.sem_weight * semantic_loss.mean()
                )

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            if cfg.semantic_mode:
                desc += f"sem_loss={semantic_loss.item():.4f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                self.save_ckpt(step=step)

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            if evaluate:
                # eval the full set
                if step in [i - 1 for i in cfg.eval_steps]:
                    self.eval(step)
                    self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size
        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        H_train_inv = None

        info = [(p["camtoworld"], p["K"]) for p in self.trainset]
        c2ws, Ks = [i[0] for i in info], info[0][1]
        height, width = self.trainset[0]["image"].shape[:2]
        H_train = self.computeHtrain(c2ws, Ks, height, width, color_fisher=0.0, depth_fisher=1.0)
        H_train_inv = torch.reciprocal(H_train + 0.0001)
        
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]
            

            torch.cuda.synchronize()
            tic = time.time()
            render_pack = self.render_at_pose(camtoworlds, Ks, width, height, H_train_inv)
            
            colors = render_pack["colors"]
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                if "depths" in data:
                    # depth
                    depth_gt = data["depths"].to(device) 
                    depth = render_pack["depth"]
                    depth_canvas_list = [depth_gt.reshape(-1, height, width), depth]

                    depth_canvas = torch.cat(depth_canvas_list, dim=2).squeeze(0).cpu().numpy()
                    depth_canvas = (depth_canvas - depth_canvas.min()) / (depth_canvas.max() - depth_canvas.min())
                    # apply color map Virius
                    depth_canvas = plt.cm.viridis(depth_canvas)[:, :, :3]
                    depth_canvas = (depth_canvas * 255).astype(np.uint8)
                    imageio.imwrite(
                        f"{self.render_dir}/{stage}_depth_step{step}_{i:04d}.png",
                        depth_canvas,
                    )

                    # # uncomment to get uint16 depth results
                    # depth_render = depth[0].cpu().numpy() * 1000
                    # depth_render = depth_render.astype(np.uint16)
                    # cv2.imwrite(
                    #     f"{self.render_dir}/{i:04d}.png",
                    #     depth_render,
                    # )

                    error_list = depth_canvas_list

                else:
                    error_list = canvas_list

                error = torch.abs(error_list[1] - error_list[0])
                mask = error_list[0] > 0.01
                error = (error * mask).squeeze(0).cpu().numpy()
                error_normalize = equal_hist(error)
                # error_normalize = (depth_error - depth_error.min()) / (depth_error.max() - depth_error.min())
                error_viz = plt.cm.viridis(error_normalize)[:, :, :3]
                error_viz = (error_viz * 255).astype(np.uint8)

                uncern = render_pack["uncern"].squeeze(0).cpu().numpy()
                uncern = equal_hist(uncern)
                # uncern = (uncern - uncern.min()) / (uncern.max() - uncern.min())
                uncern_canvas = plt.cm.viridis(uncern)[:, :, :3]
                uncern_canvas = (uncern_canvas * 255).astype(np.uint8)
                uncern_canvas = np.concatenate([uncern_canvas, error_viz], axis=1)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_uncern_step{step}_{i:04d}.png",
                    uncern_canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int) -> str:
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        # camtoworlds_all = self.parser.camtoworlds[5:-5]
        camtoworlds_all = self.parser.camtoworlds
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 30
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
                n_frames=240,
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        video_path = f"{video_dir}/traj_{step}.mp4"
        writer = imageio.get_writer(video_path, fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            # canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            # canvas = (canvas * 255).astype(np.uint8)
            # writer.append_data(canvas)
        
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")
        return video_path

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{self.cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        # NOTE: we hard-coded the focal length and image size to be close to our real sense camera
        img_wh = (640, 480)
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        K = np.array([
            [380.8, 0.0, 310.5],
            [0.0, 380.5, 245.0],
            [0.0, 0.0, 1.0],
                      ])
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()

    @torch.no_grad()
    def _sem_viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        # NOTE: we hard-coded the focal length and image size to be close to our real sense camera
        img_wh = (640, 480)
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        K = np.array([
            [380.8, 0.0, 310.5],
            [0.0, 380.5, 245.0],
            [0.0, 0.0, 1.0],
                      ])
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]

        semantics = info["semantics"][0] # [H, W, 1]

        # Map semantics to heatmap
        semantics_np = semantics.squeeze(-1).cpu().numpy()  # [H, W]
        heatmap = plt.cm.viridis(semantics_np)[:, :, :3]  # Map to RGB heatmap
        heatmap = (heatmap * 255).astype(np.uint8)  # Convert to uint8

        # Mask heatmap by render_alphas
        render_alphas_np = render_alphas[0].cpu().numpy()  # [H, W, 1]
        heatmap = heatmap * (render_alphas_np > 0.01)  # Apply alpha mask

        return heatmap
    
    def render_at_pose(self, 
                        c2ws, Ks, 
                        width, height, 
                        H_train_inv = None):
        """
            Render the scene at the given camera pose.
            Args:
                c2w: Camera-to-world matrix. [(N), 4, 4]
                K: Camera intrinsics. [(N), 3, 3]
                width: Image width.
                height: Image height.
                H_train_inv: The inverse of the H_train matrix. [N, ]
            Return:
                render_pack: The rendered image and depth & uncertainty map.
                    rgb: The rendered image. [1, H, W, 3]
                    depth: The rendered depth map. [1, H, W]
                    
        """

        device = self.splats["means"].device
        if Ks.ndim == 2:
            Ks = Ks[None]
            if Ks.device != device:
                Ks = Ks.to(device)

        if c2ws.ndim == 2:
            c2ws = c2ws[None]
            if c2ws.device != device:
                c2ws = c2ws.to(device)

        renders, _, _ = self.rasterize_splats(
            camtoworlds=c2ws,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=self.cfg.sh_degree,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            render_mode="RGB+ED",
        )  # [1, H, W, 4]
        
        colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
        depth = renders[..., 3]  # [1, H, W]
        render_pack = {"colors": colors, "depth": depth}

        if H_train_inv is not None:
            uncern_score = torch.mean(H_train_inv, 1, keepdim=True)
            gs_color = uncern_score.repeat(1, 3)

            render_uncern, render_alphas, meta = rasterization(
                self.splats["means"],  # [N, 3]
                self.splats["quats"],  # [N, 4]
                torch.exp(self.splats["scales"]),  # [N, 3]
                torch.sigmoid(self.splats["opacities"]),  # [N]
                gs_color, # [N, 3]
                torch.linalg.inv(c2ws),  # [1, 4, 4]
                Ks,  # [1, 3, 3]
                width,
                height,
                sh_degree=None,
                render_mode="RGB",
                # this is to speedup large-scale rendering by skipping far-away Gaussians.
                radius_clip=3,
            )

            render_pack["uncern"] = render_uncern[..., 0]
        
        return render_pack

    def computeHtrain(self, c2ws, Ks, 
                        width, height, **args):
        """
            Update the H_train matrix for the current set of camera poses.
        
        """
        H_train = None
        
        for c2w in tqdm.tqdm(c2ws, desc="Computing H_train"):
            H, _ = self.computeHessian(c2w, Ks, width, height, **args)
            if H_train is None:
                H_train = H
            else:
                H_train += H

        return H_train

    @torch.no_grad()
    def computeHessian(self, 
                            c2w, K, 
                            width, height,
                            **args):
        """
            Compute the Fisher Information Matrix for the current set of camera poses.
            Args:
                c2w: Camera-to-world matrices. [(N), 4, 4]
                K: Camera intrinsics. [(N), 3, 3]
                width: Image width.
                height: Image height.
                **args: Additional arguments for rasterization.
                        color_fisher: Fisher Information for color
                        depth_fisher: Fisher Information for depth
            Return:
                fishers: Fisher Information Matrix for the current set of camera poses. [1, D]
                JTJ_tau: The tau value for the JTJ matrix

        """
        # STEP 1 extract the splats
        means = self.splats["means"] # [self.world_rank::self.world_size].contiguous()
        # means.requires_grad = True
        
        quats = torch.nn.functional.normalize(self.splats["quats"], dim=1) # [self.world_rank::self.world_size].contiguous()
        # quats.requires_grad = True
        
        scales = torch.exp(self.splats["scales"]) # [self.world_rank::self.world_size].contiguous()
        if scales.shape[1] == 1:
            scales = scales.repeat(1, 3)
        # scales.requires_grad = True
        
        opacities = torch.sigmoid(self.splats["opacities"]) # [self.world_rank::self.world_size].contiguous()
        # opacities.requires_grad = True
        
        sh_degree = self.cfg.sh_degree
        if self.cfg.semantic_mode:
            colors = self.splats["colors"]
            semantics = self.splats["semantics"]    
            colors = torch.concat([semantics, colors], dim=-1)  # [N, 4]
            colors = torch.sigmoid(colors) # map from logits to [0, 1] for both color and semantics
            sh_degree = None
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
        # colors = colors[self.world_rank::self.world_size].contiguous()
        # colors.requires_grad = True

        if K.ndim == 2:
            K = K.unsqueeze(0)

        if K.device != means.device:
            K = K.to(means.device)

        viewmats = torch.linalg.inv(c2w)  # [1, 4, 4]
        if viewmats.ndim == 2:
            viewmats = viewmats.unsqueeze(0)

        if viewmats.device != means.device:
            viewmats = viewmats.to(means.device)

        # NOTE: currently we only call this once because the Hessian of colors and semantics are not actually used in the later computation
        # To compute full hessians for colors and semantics, we need to repatively call the rasterization function by every 3 channels in the colors
        # STEP 2: call the rasterization function
        color_fisher = args.get("color_fisher", 1.0)
        depth_fisher = args.get("depth_fisher", 1.0)
        render_colors, render_alphas, meta = rasterization_fisher_wrapper(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, 3] or [N, K, 3]
            viewmats,  # [1, 4, 4]
            K,
            width,
            height,
            sh_degree=self.cfg.sh_degree,
            # TODO 
            fisher_color_scaler=color_fisher,
            fisher_depth_scaler=depth_fisher
        )

        grad_means3D, grad_sh, grad_opacities, grad_scales, grad_rotations, _, _, JTJ_tau = meta["fishers"][0]
        # TODO Collect Information for the Fisher Matrix
        fishers = torch.cat([grad_means3D, 
                             # rearrange(grad_sh, "n s c -> n (s c)"),
                            #  grad_opacities, grad_scales, grad_rotations
                             ], dim=1)
        
        return fishers, JTJ_tau
    
    def save_ckpt(self, step:int):
        global_tic = self.global_tic
        mem = torch.cuda.max_memory_allocated() / 1024**3
        stats = {
            "mem": mem,
            "ellipse_time": time.time() - global_tic,
            "num_GS": len(self.splats["means"]),
        }
        print("Step: ", step, stats)
        with open(
            f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
            "w",
        ) as f:
            json.dump(stats, f)
        data = {"step": step, "splats": self.splats.state_dict()}
        world_size = self.world_size
        if self.cfg.pose_opt:
            if world_size > 1:
                data["pose_adjust"] = self.pose_adjust.module.state_dict()
            else:
                data["pose_adjust"] = self.pose_adjust.state_dict()
        if self.cfg.app_opt:
            if world_size > 1:
                data["app_module"] = self.app_module.module.state_dict()
            else:
                data["app_module"] = self.app_module.state_dict()

        saved_ckpt_path = f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
        torch.save(
            data, saved_ckpt_path
        )

        return saved_ckpt_path

def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        if cfg.max_steps not in cfg.eval_steps:
            cfg.eval_steps.append(cfg.max_steps)
        runner.train(evaluate=True)

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__" and False:
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
        "ours": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_scale=0.005,
                opacity_reg=0.01,
                scale_reg=0.01,
                depth_loss = 1.,
                init_type="romatch",
                sh_degree=0,
                strategy=DefaultStrategy(verbose=True)
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    cli(main, cfg, verbose=True)


def train_gs_sem(run_dir: str, result_dir: str, load_ckpt: str = None):
    from gaussian_splatting_py.datasets.kinova import KinovaDataset
    train_dataset = KinovaDataset(run_dir)
    
    training_step = 300
    cfg = Config()
    # set running steps
    cfg.max_steps = training_step
    cfg.init_scale = 0.05
    cfg.init_type = "romatch"
    cfg.depth_loss = len(train_dataset.depths) > 0
    cfg.result_dir = result_dir
    cfg.semantic_mode = True
    cfg.sh_degree = 0
    cfg.init_opa = 0.9

    gs_trainer = Runner(0, 0, 1, cfg, train_dataset=train_dataset, parser=None)

    
    if load_ckpt is None:
        gs_trainer.train()
    else:
        # load weights
        ckpt = torch.load(load_ckpt, weights_only=True)
        for k in gs_trainer.splats.keys():
            gs_trainer.splats[k].data = ckpt["splats"][k]

        c2w = torch.Tensor([[ 5.4440e-01,  5.6300e-01,  6.2182e-01,  2.3333e-01],
                [ 8.3883e-01, -3.6539e-01, -4.0356e-01,  1.4393e-01],
                [-5.5511e-17,  7.4130e-01, -6.7118e-01,  2.1261e-01],
                [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]).cuda()

        K = torch.Tensor([[400.,   0., 400.],
                [  0., 400., 400.],
                [  0.,   0.,   1.]]).cuda()
        width = 800
        height = 800
        args = {}

        out = gs_trainer.computeHessian(c2w, K, width, height, **args)

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

if __name__ == "__main__":
    tyro.cli(train_gs_sem)