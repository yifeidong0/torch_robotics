import abc
from abc import ABC

import torch

from torch_robotics.torch_utils.torch_utils import to_numpy


class RobotBase(ABC):

    def __init__(
            self,
            name='RobotBase',
            q_limits=None,
            self_collision_margin=0.001,
            num_interpolate=4,
            link_interpolate_range=[0, 7],
            tensor_args=None,
            **kwargs
    ):
        self.name = name
        self.tensor_args = tensor_args

        ################################################################################################
        # Configuration space
        assert q_limits is not None, "q_limits cannot be None"
        self.q_limits = q_limits
        self.q_min = q_limits[0]
        self.q_max = q_limits[1]
        self.q_min_np = to_numpy(self.q_min)
        self.q_max_np = to_numpy(self.q_max)
        self.q_distribution = torch.distributions.uniform.Uniform(self.q_min, self.q_max)
        self.q_dim = len(self.q_min)

        # Collision field
        self.self_collision_margin = self_collision_margin
        self.num_interpolate = num_interpolate
        self.link_interpolate_range = link_interpolate_range  # which links to interpolate for collision checking

    def random_q(self, n_samples=10):
        # Random position in configuration space
        q_pos = self.q_distribution.sample((n_samples,))
        return q_pos

    def get_position(self, x):
        return x[..., :self.q_dim]

    def get_velocity(self, x):
        return x[..., self.q_dim:2 * self.q_dim]

    def get_acceleration(self, x):
        raise NotImplementedError

    def distance_q(self, q1, q2):
        return torch.linalg.norm(q1 - q2, dim=-1)

    def fk_map(self, q, **kwargs):
        if q.ndim == 1:
            q = q.unsqueeze(0)  # add batch dimension
        return self.fk_map_impl(q, **kwargs)

    @abc.abstractmethod
    def fk_map_impl(self, q, pos_only=False, return_dict=False):
        # q: (..., q_dim)
        # return: dict{'link_tensor_pos': (..., taskspaces, x_dim) OR (..., taskspaces, x_dim+1, x_dim+1),
        #               (optional 'grasped_object_coll_points_pos': (..., number_of_coll_points, x_dim)) }
        raise NotImplementedError

    @abc.abstractmethod
    def render(self, ax, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def render_trajectories(self, ax, trajs=None, **kwargs):
        raise NotImplementedError
