import numpy as np
import torch
from matplotlib import pyplot as plt
import itertools
import os

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MeshField
from torch_robotics.robots import RobotPointMass, RobotPanda
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes
from torch_robotics.environments.grid_map_sdf import GridMapSDF
from mpd.utils.loading import load_params_from_yaml

DATA_DIR = 'data_trajectories/EnvEmpty-RobotAllegro/0'

class EnvEmpty(EnvBase):
    def __init__(self, 
                 name='EnvEmpty', 
                 tensor_args=None,
                 precompute_sdf_obj_fixed=False,
                 sdf_cell_size=0.005, 
                 **kwargs):
        self.args = load_params_from_yaml(os.path.join(DATA_DIR, 'args.yaml'))
        self.precompute_sdf_obj_fixed = precompute_sdf_obj_fixed
        self.sdf_cell_size = sdf_cell_size
        self.env_name = 'EnvEmpty'
        # self.env_filename = 'deps/torch_robotics/torch_robotics/data/urdf/objects/slatwall-hook/slatwall_hook.urdf'
        # self.env_position = torch.tensor(self.args['obstacle_position'], **tensor_args)
        # self.env_quaternion = torch.tensor(self.args['obstacle_orientation'], **tensor_args) # TODO: still manually set
        # self.obstacle_scale = self.args['obstacle_scale']
        obj_list = []

        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1, -1, -3.142, -3.142, -3.142], [1, 1, 1, 3.142, 3.142, 3.142]], 
                                **tensor_args),  # environments limits
            obj_fixed_list=obj_list,
            tensor_args=tensor_args,
            **kwargs
        )
        self.dim = 3
    
    def get_gpmp2_params(self, robot=None):
        params = dict(
            opt_iters=250,
            num_samples=64,
            sigma_start=1e-3,
            sigma_gp=1e-1,
            sigma_goal_prior=1e-3,
            sigma_coll=1e-4,
            step_size=1e0,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.1,
            sigma_start_sample=1e-3,
            sigma_goal_sample=1e-3,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'sparse_computation': False,
                'sparse_computation_block_diag': False,
                'method': 'cholesky',
                # 'method': 'cholesky-sparse',
                # 'method': 'inverse',
            },
            stop_criteria=0.1,
        )
        if isinstance(robot, RobotPanda):
            return params
        else:
            raise NotImplementedError

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=torch.pi/30,
            n_radius=torch.pi/4,
            n_pre_samples=50000,

            max_time=90
        )
        if isinstance(robot, RobotPanda):
            return params
        else:
            raise NotImplementedError

if __name__ == '__main__':
    env = EnvEmpty(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )
    fig, ax = create_fig_and_axes(env.dim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    env.render(ax)
    plt.show()

    # # Render sdf
    # fig, ax = create_fig_and_axes(env.dim)
    # env.render_sdf(ax, fig)

    # # Render gradient of sdf
    # env.render_grad_sdf(ax, fig)
    # plt.show()
