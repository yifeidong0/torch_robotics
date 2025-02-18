import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os

from torch_robotics.environments.primitives import plot_sphere
from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch
from torch_robotics.visualizers.plot_utils import plot_coordinate_frame

import matplotlib.collections as mcoll
from mpd.utils.loading import load_params_from_yaml

DATA_DIR = 'data_trajectories/EnvHook3D-RobotTape3D/0' # TODO:

class RobotRigid3D(RobotBase):
    def __init__(self, 
                 name='RobotRigid3D', 
                 q_limits=torch.tensor([[-1, -1, -1, -3.142, -3.142, -3.142], 
                                        [1, 1, 1, 3.142, 3.142, 3.142]]),  # configuration space limits
                 tensor_args=None, 
                 **kwargs):
        self.q_limits = q_limits
        super().__init__(
            name=name,
            q_limits=q_limits,
            link_names_for_object_collision_checking=['base'],
            link_margins_for_object_collision_checking=[0.01],
            link_idxs_for_object_collision_checking=[0],
            num_interpolated_points_for_object_collision_checking=1,
            tensor_args=tensor_args,
            **kwargs
        )

    def get_position(self, x): # pose actually
        return x[..., :self.q_limits.shape[-1]]
    
    def fk_map_collision_impl(self, q, **kwargs):
        """The robot is a rigid 3D body, so FK is just identity."""
        return q.unsqueeze(-2)  # Adding task space dimension

    def render(self, ax, q=None, color='blue', **kwargs):
        """Visualize the 3D rigid body."""
        ax.scatter(q[0].item(), q[1].item(), q[2].item(), color=color)
        plot_coordinate_frame(ax, q, arrow_length=0.1)

    def render_trajectories(self, ax, trajs=None, colors=['gray'], linestyle='solid', **kwargs):
        """Visualize multiple 3D trajectories.
        
        Args:
            ax: The Matplotlib 3D axis.
            trajs (tensor or numpy.ndarray): Shape (n_trajs, 64, 3), representing multiple trajectories.
            colors (list): List of colors for each trajectory.
            linestyle (str): Line style for the trajectories.
        """
        if trajs is not None:
            print(f"!!!!!!!!!%%$$!!!!!!!!!!!trajs.shape: {trajs.shape}")
            trajs_pos = trajs[..., :3] # Do not render the orientation
            trajs_np = to_numpy(trajs_pos)  # Convert tensor to numpy if needed
            
            num_trajs, num_points, _ = trajs_np.shape
            
            # Create line segments for 3D plotting
            segments = np.array(list(zip(trajs_np[..., 0], trajs_np[..., 1], trajs_np[..., 2]))).swapaxes(1, 2)
            line_segments = Line3DCollection(segments, colors=colors, linestyle=linestyle)
            ax.add_collection(line_segments)

            # Reshape for scatter plot
            points = trajs_np.reshape(-1, 3)
            
            # Assign colors per trajectory
            colors_scatter = [colors[i % len(colors)] for i in range(num_trajs) for _ in range(num_points)]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors_scatter, s=2**2)  # Small dots for waypoints



class RobotTape3D(RobotRigid3D):
    def __init__(self,
                name='RobotTape3D',
                q_limits=torch.tensor([[-1, -1, -1, -3.142, -3.142, -3.142], 
                                       [1, 1, 1, 3.142, 3.142, 3.142]]),  # configuration space limits. NOTE: the dimension here affects hard_cond dimension
                **kwargs):
        self.args = load_params_from_yaml(os.path.join(DATA_DIR, 'args.yaml'))
        bd = self.args['search_bounds_r3']
        # q_limits = torch.tensor([[bd[0][0], bd[1][0], bd[2][0], -3.142, -3.142, -3.142],
        #                          [bd[0][1], bd[1][1], bd[2][1], 3.142, 3.142, 3.142]], **kwargs['tensor_args'])
        q_limits = torch.tensor(bd, **kwargs['tensor_args']).T
        print(f"@!@!@!@!@!q_limits: {q_limits}")
        self.object_scale = self.args['object_scale']
        self.robot_filename = "deps/torch_robotics/torch_robotics/data/urdf/robots/rigid_3d/tape/tape.urdf"

        super().__init__(
            name=name,
            q_limits=to_torch(q_limits, **kwargs['tensor_args']),
            **kwargs
        )
    
