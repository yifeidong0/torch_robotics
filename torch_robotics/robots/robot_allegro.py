import pybullet as p
import pybullet_data
import time
import torch
from collections import OrderedDict
import torch
import numpy as np
import einops

from torch_robotics.environments.primitives import MultiSphereField # added to avoid loop import
from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.torch_kinematics_tree.models.robots import DifferentiableAllegroHand
from torch_robotics.torch_kinematics_tree.models.robot_tree import convert_link_dict_to_tensor
from torch_robotics.torch_kinematics_tree.geometrics.utils import (
    link_pos_from_link_tensor, link_rot_from_link_tensor, link_quat_from_link_tensor
)
from torch_robotics.visualizers.plot_utils import plot_coordinate_frame
from torch_robotics.environments.primitives import MultiSphereField
from torch_robotics.torch_utils.torch_utils import to_numpy
from torch_robotics.torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model

class RobotAllegro(RobotBase):
    def __init__(self, tensor_args=None, **kwargs):
        self.diff_allegro = DifferentiableAllegroHand(device=tensor_args['device'])
        
        # Set joint limits
        self.jl_lower, self.jl_upper, _, _ = self.diff_allegro.get_joint_limit_array()
        q_limits = torch.tensor([self.jl_lower, self.jl_upper], **tensor_args)
        
        # Define end-effector links (fingertips)
        self.finger_links = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
        link_names_for_object_collision_checking = [
            "index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"
        ]
        link_margins_for_object_collision_checking = [
            0.01, 0.01, 0.01, 0.01
        ]

        # Initialize the base class
        super().__init__(
            name='RobotAllegro',
            q_limits=q_limits,
            tensor_args=tensor_args,
            link_names_for_object_collision_checking=link_names_for_object_collision_checking,
            num_interpolated_points_for_object_collision_checking=len(link_names_for_object_collision_checking),
            link_margins_for_object_collision_checking=link_margins_for_object_collision_checking,
            **kwargs
        )

    def fk_map_collision_impl(self, q, **kwargs):
        q_orig_shape = q.shape
        if len(q_orig_shape) == 3:
            b, h, d = q_orig_shape
            q = einops.rearrange(q, 'b h d -> (b h) d')
        elif len(q_orig_shape) == 2:
            h = 1
            b, d = q_orig_shape
        else:
            raise NotImplementedError

        # Compute FK for all links
        link_pose_dict = self.diff_allegro.compute_forward_kinematics_all_links(q, return_dict=True)
        link_tensor = convert_link_dict_to_tensor(link_pose_dict, self.diff_allegro.get_link_names())

        if len(q_orig_shape) == 3:
            link_tensor = einops.rearrange(link_tensor, "(b h) t d1 d2 -> b h t d1 d2", b=b, h=h)

        return link_pos_from_link_tensor(link_tensor)

    def get_position(self, q):
        # Compute FK for fingertip positions
        fingertip_poses = self.diff_allegro.compute_forward_kinematics_all_links(q, link_list=self.finger_links)
        return link_pos_from_link_tensor(fingertip_poses)

    def render(self, ax, q=None, color='blue', arrow_length=0.05, arrow_alpha=1.0, arrow_linewidth=2.0, **kwargs):
        """ Render the skeleton of the Allegro hand """
        # Draw skeleton
        skeleton = get_skeleton_from_model(self.diff_allegro, q, self.diff_allegro.get_link_names())
        skeleton.draw_skeleton(ax=ax, color=color)

        # Compute FK
        fks_dict = self.diff_allegro.compute_forward_kinematics_all_links(q.unsqueeze(0), return_dict=True)

        # Draw fingertip frames
        for finger in self.finger_links:
            frame_finger = fks_dict[finger]
            plot_coordinate_frame(
                ax, frame_finger, tensor_args=self.tensor_args,
                arrow_length=arrow_length, arrow_alpha=arrow_alpha, arrow_linewidth=arrow_linewidth
            )

    def render_trajectories(self, ax, trajs=None, start_state=None, goal_state=None, colors=['gray'], **kwargs):
        """ Render multiple trajectories of the Allegro hand """
        if trajs is not None:
            trajs_pos = self.get_position(trajs)
            for traj, color in zip(trajs_pos, colors):
                for t in range(traj.shape[0]):
                    q = traj[t]
                    self.render(ax, q, color=color, **kwargs, arrow_length=0.03, arrow_alpha=0.5, arrow_linewidth=1.0)
            if start_state is not None:
                self.render(ax, start_state, color='green')
            if goal_state is not None:
                self.render(ax, goal_state, color='purple')


def main():
    # Initialize PyBullet in GUI mode
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # URDF path
    p.setGravity(0, 0, -9.81)

    # Load Allegro Hand URDF
    allegro_id = p.loadURDF("/home/yif/Documents/KTH/git/mpd-cage/deps/torch_robotics/torch_robotics/data/urdf/robots/allegro_hand_description/allegro_hand_description_left.urdf", useFixedBase=True)

    # Initialize RobotAllegro using PyTorch
    tensor_args = {'device': 'cpu', 'dtype': torch.float32}
    robot_allegro = RobotAllegro(tensor_args=tensor_args)

    # Set an initial joint configuration (16 DOF Allegro Hand)
    q_init = torch.zeros(16, dtype=torch.float32)

    # Apply initial joint state in PyBullet
    for i in range(len(q_init)):
        p.resetJointState(allegro_id, i, q_init[i].item())

    # Run PyBullet simulation loop
    while True:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)  # Run at 240Hz

    # Disconnect PyBullet (never reaches here due to infinite loop)
    p.disconnect()

if __name__ == "__main__":
    main()