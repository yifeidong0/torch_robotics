import itertools
import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.environments.utils import create_grid_spheres
from torch_robotics.environments.grid_map_sdf import GridMapSDF
from torch_robotics.robots import RobotPointMass
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes

class EnvCage2D(EnvBase):

    def __init__(self,
                 name='EnvCage2D',
                 tensor_args=None,
                 precompute_sdf_obj_fixed=True,
                 sdf_cell_size=0.005,
                 **kwargs
                 ):
        self.precompute_sdf_obj_fixed = precompute_sdf_obj_fixed
        self.tensor_args = tensor_args 
        self.sdf_cell_size = sdf_cell_size
        obj_list = [
            MultiSphereField(
                np.array(
                [[ 0.49003329, -0.09933467],
                [ 0.37314733, -0.33280785],
                [ 0.1545085,  -0.47552826],
                [-0.20121021, -0.45772749],
                [-0.33805781, -0.36839777],
                [-0.49438554,  0.07471907]]),
                np.array(
                [0.2, 0.3, 0.2315, 0.12, 0.2, 0.27]
                )
                ,
                tensor_args=tensor_args
            ),
        ]

        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # environments limits
            obj_fixed_list=[ObjectField(obj_list, 'cage2d')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs
        )

    def update_obstacles(self, sphere_centers, shpere_radii):
        """
        Update the obstacles (obj_list) and other related variables.

        Parameters:
        """
        # Re-initialize the obj_fixed_list with the new obstacles
        new_obj_list = [
            MultiSphereField(
                sphere_centers,
                shpere_radii,
                tensor_args=self.tensor_args
            ),
        ]
        self.obj_fixed_list = [ObjectField(new_obj_list, 'cage2d')]
        self.obj_all_list = set(itertools.chain.from_iterable((
            self.obj_fixed_list if self.obj_fixed_list is not None else [],
            self.obj_extra_list if self.obj_extra_list is not None else [])
        ))

        # Update the SDF of the environment if precomputing is enabled
        if self.precompute_sdf_obj_fixed:
            self.grid_map_sdf_obj_fixed = GridMapSDF(
                self.limits, self.sdf_cell_size, self.obj_fixed_list, tensor_args=self.tensor_args
            )

        # Recompute occupancy map if necessary
        # self.build_occupancy_map(self.cell_size)

        # Optionally, re-render the environment after updating obstacles
        # self.render(ax) # You can render again if desired.

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=0.01,
            n_radius=0.3,
            n_pre_samples=50000,
            max_time=50
        )

        if isinstance(robot, RobotPointMass):
            return params
        else:
            raise NotImplementedError

    def get_gpmp2_params(self, robot=None):
        params = dict(
            n_support_points=64,
            dt=0.04,
            opt_iters=300,
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=1e-5,
            step_size=1e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.2,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'method': 'cholesky',
            },
        )

        if isinstance(robot, RobotPointMass):
            return params
        else:
            raise NotImplementedError

    def get_chomp_params(self, robot=None):
        params = dict(
            n_support_points=64,
            dt=0.04,
            opt_iters=1,  # Keep this 1 for visualization
            weight_prior_cost=1e-4,
            step_size=0.05,
            grad_clip=0.05,
            sigma_start_init=0.001,
            sigma_goal_init=0.001,
            sigma_gp_init=0.3,
            pos_only=False,
        )

        if isinstance(robot, RobotPointMass):
            return params
        else:
            raise NotImplementedError


if __name__ == '__main__':
    env = EnvCage2D(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )
    sphere_centers = np.array(
        [[ 0.4, -0.1],
        [ 0.3, -0.3],
        [ 0.1,  -0.4],
        [-0.2, -0.4],
        [-0.3, -0.3],
        [-0.5,  0.1]]
    )
    sphere_radii = np.array([0.2, 0.3, 0.3, 0.3, 0.2, 0.27])
    env.update_obstacles(sphere_centers, sphere_radii)

    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()
