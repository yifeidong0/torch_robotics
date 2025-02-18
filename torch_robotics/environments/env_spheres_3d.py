import numpy as np
import torch
from matplotlib import pyplot as plt
import itertools

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField
from torch_robotics.robots import RobotPointMass, RobotPanda
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes
from torch_robotics.environments.grid_map_sdf import GridMapSDF

def fibonacci_hemisphere(n, center, radius):
    """
    Generate n approximately uniformly distributed points over the lower half 
    (z < z_center) of a hemisphere using a Fibonacci spiral approach.

    Parameters:
        n (int): Number of points.
        center (tuple): Center of the hemisphere (x, y, z).
        radius (float): Radius of the hemisphere.

    Returns:
        np.ndarray: Array of shape (n, 3) containing the generated points.
    """
    x_c, y_c, z_c = center

    # Golden ratio for uniform distribution
    phi = (1 + np.sqrt(5)) / 2  
    
    # Generate Fibonacci spiral points on a sphere
    i = np.arange(1, n + 1)
    theta = 2 * np.pi * i / phi  # Spread points evenly in azimuth
    z = -1 + (2 * i - 1) / n  # Evenly spaced in z direction
    x = np.sqrt(1 - z**2) * np.cos(theta)
    y = np.sqrt(1 - z**2) * np.sin(theta)

    # Keep only lower hemisphere points (z < z_center)
    lower_mask = z < 0  
    x, y, z = x[lower_mask], y[lower_mask], z[lower_mask]

    # Scale by radius and shift to the given center
    x = x_c + radius * x
    y = y_c + radius * y
    z = z_c + radius * z

    return np.column_stack((x, y, z))

def generate_random_obstacles_3d(num_obstacles=6, fix_obstacles=False, fixed_centers=None, fixed_radii=None, robot_radius=0.15):
    """
    Generates a 3D environment with `num_obstacles` spherical obstacles arranged on the lower half
    of a spherical shell (i.e. with z < 0), forming a bowl-like obstacle space.
    The sphere object's start state is sampled from inside the bowl (with higher z values)
    and is guaranteed to be collision-free with respect to the obstacles.
    A fixed goal is set at [0, 0, 1] (representing an escape upward).

    Parameters:
        num_obstacles (int): Number of obstacles to generate.
        fix_obstacles (bool): If True, use the provided fixed_centers and fixed_radii.
        fixed_centers (array-like): Predefined obstacle centers if fix_obstacles is True.
        fixed_radii (array-like): Predefined obstacle radii if fix_obstacles is True.
    
    Returns:
        centers (np.ndarray): Array of shape (num_obstacles, 3) with obstacle centers.
        radii (np.ndarray): Array of shape (num_obstacles,) with obstacle radii.
        start_pos (list): A valid start position [x, y, z] (inside the bowl and collision-free).
        goal_pos (list): The fixed goal position [0, 0, 1].
    """
    import numpy as np

    # Define the spherical shell (bowl) parameters.
    base_shell_radius = 0.4          # Base radius of the shell.
    base_shell_center = np.array([0, 0, 0.5])  # Center of the shell; the bowl covers z <= 0.
    # thickness = 0.0                  # Variation in the radial distance.
    centers = fibonacci_hemisphere(num_obstacles*2, base_shell_center, base_shell_radius)
    centers += np.random.normal(0, 0.03, centers.shape)  # Add noise to the obstacle centers
    print(f"centers: {centers.shape}")

    if fix_obstacles and fixed_centers is not None and fixed_radii is not None:
        centers = np.array(fixed_centers)
        radii = np.array(fixed_radii)
    else:
        radii_list = []
        for _ in range(num_obstacles):
            obs_radius = np.random.uniform(0.15, 0.25)
            radii_list.append(obs_radius)
        # centers = np.array(centers_list)
        # print(f"centers: {centers.shape}")
        radii = np.array(radii_list)

    # Sample a valid start state for the sphere object.
    # We choose a region inside the bowl (with higher z values) so that the object is inside the bowl.
    # Here, we restrict theta to [0, pi/2] so that z = r*cos(theta) is nonnegative.
    while True:
        start_rad = np.random.uniform(0, base_shell_radius * 1.0)  # use a fraction of the shell radius
        start_theta = np.random.uniform(0, 2 * np.pi)              # only the upper half of the ball
        start_phi = np.random.uniform(0, 2 * np.pi)
        start_candidate = np.array([
            start_rad * np.sin(start_theta) * np.cos(start_phi) + base_shell_center[0],
            start_rad * np.sin(start_theta) * np.sin(start_phi) + base_shell_center[1],
            start_rad * np.cos(start_theta) + base_shell_center[2]
        ])
        # Ensure the start candidate is collision-free.
        collision = False
        for i in range(num_obstacles):
            if np.linalg.norm(start_candidate - centers[i]) < radii[i]+robot_radius:
                collision = True
                break
        if not collision:
            start_pos = start_candidate
            break

    # Set a fixed goal representing an escape
    goal_pos = np.array([0, 0, -1])
    return centers, radii, start_pos.tolist(), goal_pos.tolist()

class EnvSpheres3D(EnvBase):

    def __init__(self, 
                 name='EnvDense2D', 
                 tensor_args=None,
                 precompute_sdf_obj_fixed=False,
                 sdf_cell_size=0.005, 
                 **kwargs):
        self.precompute_sdf_obj_fixed = precompute_sdf_obj_fixed
        self.sdf_cell_size = sdf_cell_size
        self.env_name = 'EnvSpheres3D'
        centers = torch.tensor([
                    [-0.3, 0.3, 0.85],
                    [-0.35, -0.25, 0.45],
                    [-0.45, 0.15, 0.0],
                    [0.45, 0.35, 0.],
                    [0.55, 0.35, 0.55],
                    [0.65, -0.4, 0.25],
                    [0.2, -0.35, 0.5],
                    [0.35, 0.0, 0.9],
                    [0., -0.3, 0.0],
                    [0.0, 0.45, 0.35],
                ], **tensor_args)
        radii = torch.tensor([0.15,]*10, **tensor_args)
        # radii = torch.tensor([0.15,]*6, **tensor_args)
        spheres = MultiSphereField(centers, radii, tensor_args=tensor_args)

        obj_field = ObjectField([spheres], 'spheres')
        obj_list = [obj_field]

        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1, -1], [1, 1, 1]], **tensor_args),  # environments limits
            obj_fixed_list=obj_list,
            tensor_args=tensor_args,
            **kwargs
        )

        self.spheres_flat = torch.cat([centers, radii.unsqueeze(1)], dim=1).flatten() # torch.Size([40])

    def generate_rand_obstacles(self, n_obstacles=6):
        """
        Generate random obstacles in the environment.

        Parameters:
        n_obstacles (int): Number of obstacles to generate.
        """
        # TODO: update it together with dataset generation
        centers, radii, start_pos, goal_pos = generate_random_obstacles_3d(n_obstacles, fix_obstacles=False)
        self.update_obstacles(centers, radii)

        return centers, radii, start_pos, goal_pos
    
    def update_obstacles(self, sphere_centers, sphere_radii):
        """
        Update the obstacles (obj_list) and other related variables.

        Parameters:
        """
        # Re-initialize the obj_fixed_list with the new obstacles
        new_obj_list = [
            MultiSphereField(
                sphere_centers,
                sphere_radii,
                tensor_args=self.tensor_args
            ),
        ]
        self.obj_fixed_list = [ObjectField(new_obj_list, 'sphere3d')]
        self.obj_all_list = set(itertools.chain.from_iterable((
            self.obj_fixed_list if self.obj_fixed_list is not None else [],
            self.obj_extra_list if self.obj_extra_list is not None else [])
        ))

        # Update the SDF of the environment if precomputing is enabled
        if self.precompute_sdf_obj_fixed:
            self.grid_map_sdf_obj_fixed = GridMapSDF(
                self.limits, self.sdf_cell_size, self.obj_fixed_list, tensor_args=self.tensor_args
            )

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
    env = EnvSpheres3D(
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

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()
