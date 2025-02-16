import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from torch_robotics.environments.primitives import plot_sphere
from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch

import matplotlib.collections as mcoll


class RobotPointMass(RobotBase):

    def __init__(self,
                 name='RobotPointMass',
                 q_limits=torch.tensor([[-1, -1], [1, 1]]),  # configuration space limits
                 **kwargs):
        if 'object_radius' in kwargs:
            link_names_for_object_collision_checking = kwargs['object_radius']
        else:
            link_names_for_object_collision_checking = 0.01
        super().__init__(
            name=name,
            q_limits=to_torch(q_limits, **kwargs['tensor_args']),
            link_names_for_object_collision_checking=['link_0'],
            link_margins_for_object_collision_checking=[link_names_for_object_collision_checking],
            link_idxs_for_object_collision_checking=[0],
            num_interpolated_points_for_object_collision_checking=1,
            **kwargs
        )

    def fk_map_collision_impl(self, q, **kwargs):
        # There is no forward kinematics. Assume it's the identity.
        # Add tasks space dimension
        return q.unsqueeze(-2)

    def render(self, ax, q=None, color='blue', cmap='Blues', margin_multiplier=1., **kwargs):
        if q is not None:
            margin = self.link_margins_for_object_collision_checking[0] * margin_multiplier
            q = to_numpy(q)
            if q.ndim == 1:
                if self.q_dim == 2:
                    circle1 = plt.Circle(q, margin, color=color, zorder=10)
                    ax.add_patch(circle1)
                elif self.q_dim == 3:
                    plot_sphere(ax, q, np.zeros_like(q), margin, cmap)
                else:
                    raise NotImplementedError
            elif q.ndim == 2:
                if q.shape[-1] == 2:
                    # ax.scatter(q[:, 0], q[:, 1], color=color, s=10 ** 2, zorder=10)
                    circ = []
                    for q_ in q:
                        circ.append(plt.Circle(q_, margin, color=color))
                        coll = mcoll.PatchCollection(circ, zorder=10)
                        ax.add_collection(coll)
                elif q.shape[-1] == 3:
                    # ax.scatter(q[:, 0], q[:, 1], q[:, 2], color=color, s=10 ** 2, zorder=10)
                    for q_ in q:
                        plot_sphere(ax, q_, np.zeros_like(q_), margin, cmap)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

    def render_trajectories(
            self, ax, trajs=None, start_state=None, goal_state=None, colors=['blue'],
            linestyle='solid', **kwargs):
        if trajs is not None:
            trajs_pos = self.get_position(trajs)
            trajs_np = to_numpy(trajs_pos)
            if self.q_dim == 3:
                segments = np.array(list(zip(trajs_np[..., 0], trajs_np[..., 1], trajs_np[..., 2]))).swapaxes(1, 2)
                line_segments = Line3DCollection(segments, colors=colors, linestyle=linestyle)
                ax.add_collection(line_segments)
                points = np.reshape(trajs_np, (-1, 3))
                colors_scatter = []
                for segment, color in zip(segments, colors):
                    colors_scatter.extend([color]*segment.shape[0])
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors_scatter, s=2**2)
            else:
                segments = np.array(list(zip(trajs_np[..., 0], trajs_np[..., 1]))).swapaxes(1, 2)
                line_segments = mcoll.LineCollection(segments, colors=colors, linestyle=linestyle)
                ax.add_collection(line_segments)
                points = np.reshape(trajs_np, (-1, 2))
                colors_scatter = []
                for segment, color in zip(segments, colors):
                    colors_scatter.extend([color]*segment.shape[0])
                ax.scatter(points[:, 0], points[:, 1], color=colors_scatter, s=2**2)
        if start_state is not None:
            start_state_np = to_numpy(start_state)
            if len(start_state_np) == 3:
                ax.plot(start_state_np[0], start_state_np[1], start_state_np[2], 'go', markersize=7)
            else:
                ax.plot(start_state_np[0], start_state_np[1], 'go', markersize=7)
        if goal_state is not None:
            goal_state_np = to_numpy(goal_state)
            if len(goal_state_np) == 3:
                ax.plot(goal_state_np[0], goal_state_np[1], goal_state_np[2], marker='o', color='purple', markersize=7)
            else:
                ax.plot(goal_state_np[0], goal_state_np[1], marker='o', color='purple', markersize=7)


class RobotPointMass3D(RobotPointMass):

    def __init__(self, **kwargs):
        super().__init__(
            name='RobotPointMass3D',
            q_limits=torch.tensor([[-1, -1, -1], [1, 1, 1]], **kwargs['tensor_args']),  # configuration space limits
            **kwargs
        )


class RobotSphere3D(RobotPointMass3D):
    """
    A 3D sphere robot whose geometry is a sphere with a specified radius.
    It inherits the forward kinematics (centered at q) from the point-mass robot,
    and uses the sphere radius for both collision checking and rendering.
    """
    def __init__(self, radius: float = 0.15, **kwargs):
        # Set the collision margin to be the sphere's radius.
        kwargs['object_radius'] = radius
        super().__init__(**kwargs)
        self.radius = radius

    def render_trajectories(
            self, ax, trajs=None, start_state=None, goal_state=None, colors=['blue'],
            linestyle='solid', **kwargs):
        if trajs is not None:
            trajs_pos = self.get_position(trajs)
            trajs_np = to_numpy(trajs_pos)

            if self.q_dim == 3:
                segments = np.array(list(zip(trajs_np[..., 0], trajs_np[..., 1], trajs_np[..., 2]))).swapaxes(1, 2)
                line_segments = Line3DCollection(segments, colors=colors, linestyle=linestyle)
                ax.add_collection(line_segments)

                points = np.reshape(trajs_np, (-1, 3))
                num_trajs, num_points, _ = trajs_np.shape
                
                # Expand colors to match the number of points
                colors_scatter = [colors[i % len(colors)] for i in range(num_trajs) for _ in range(num_points)]
                
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors_scatter, s=2**2)
            else:
                segments = np.array(list(zip(trajs_np[..., 0], trajs_np[..., 1]))).swapaxes(1, 2)
                line_segments = mcoll.LineCollection(segments, colors=colors, linestyle=linestyle)
                ax.add_collection(line_segments)

                points = np.reshape(trajs_np, (-1, 2))
                num_trajs, num_points, _ = trajs_np.shape

                # Expand colors to match the number of points
                colors_scatter = [colors[i % len(colors)] for i in range(num_trajs) for _ in range(num_points)]
                
                ax.scatter(points[:, 0], points[:, 1], color=colors_scatter, s=2**2)

        if start_state is not None:
            start_state_np = to_numpy(start_state)
            if len(start_state_np) == 3:
                ax.plot(start_state_np[0], start_state_np[1], start_state_np[2], 'go', markersize=7)
            else:
                ax.plot(start_state_np[0], start_state_np[1], 'go', markersize=7)

        if goal_state is not None:
            goal_state_np = to_numpy(goal_state)
            if len(goal_state_np) == 3:
                ax.plot(goal_state_np[0], goal_state_np[1], goal_state_np[2], marker='o', color='purple', markersize=7)
            else:
                ax.plot(goal_state_np[0], goal_state_np[1], marker='o', color='purple', markersize=7)


if __name__ == "__main__":
    # Create a 3D sphere robot with radius 0.2
    tensor_args = {'dtype': torch.float32, 'device': 'cpu'}
    robot = RobotSphere3D(radius=0.5, tensor_args=tensor_args)

    # Sample a configuration (center of the sphere)
    q_sample = torch.tensor([0.5, -0.3, 0.1], **tensor_args)

    # Plotting the sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    robot.render(ax, q=q_sample, color='cyan', cmap='viridis')

    # Render a trajectory
    trajs = torch.rand(10, 5, 3, **tensor_args)
    start_state = torch.tensor([0.5, -0.3, 0.1], **tensor_args)
    goal_state = torch.tensor([0.2, 0.4, -0.3], **tensor_args)
    robot.render_trajectories(ax, trajs=trajs, start_state=start_state, goal_state=goal_state, colors=['blue'])

    # Set axis limits for better visualization
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()