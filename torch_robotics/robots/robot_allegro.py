import pybullet as p
import pybullet_data
import time
import torch
from collections import OrderedDict
import torch
import numpy as np
import einops
import cv2

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

# Mapping of 16D robot joints to 20D PyBullet joints (skip fixed joints)
JOINT_MAP = [0, 1, 2, 3,  # Index Finger
             5, 6, 7, 8,  # Middle Finger
             10, 11, 12, 13,  # Ring Finger
             15, 16, 17, 18]  # Thumb Finger
PARENT_CHILD_LINK_PAIRS = [(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4),  # Index Finger
                            (-1, 5), (5, 6), (6, 7), (7, 8), (8, 9),  # Middle Finger
                            (-1, 10), (10, 11), (11, 12), (12, 13), (13, 14),  # Ring Finger
                            (-1, 15), (15, 16), (16, 17), (17, 18), (18, 19)]  # Thumb Finger

def check_self_collision(robot_id, visualize=0):
    """
    Check if the robot is in self-collision.
    Returns True if a collision is detected.
    """
    sphere_id = None
    for i in range(-1, p.getNumJoints(robot_id)):
        for j in range(i+1, p.getNumJoints(robot_id)):
            if (i, j) in PARENT_CHILD_LINK_PAIRS:
                continue

            contact_points = p.getClosestPoints(robot_id, robot_id, -0.01, i, j)
            if len(contact_points) > 0:
                # Mark in red of the two links in contact
                # for point in contact_points:
                #     p.addUserDebugLine(point[5], point[6], [1, 0, 0], 30)

                # Add a temporary ball at one of the contact points
                sphere_id = p.createMultiBody(baseMass=0, 
                                              baseInertialFramePosition=[0, 0, 0], 
                                              baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.2),
                                              basePosition=contact_points[0][5], 
                                              baseOrientation=[0, 0, 0, 1], 
                                              useMaximalCoordinates=True)
                
                time.sleep(5) if visualize else None
                p.removeAllUserDebugItems()

                return True, sphere_id
    return False, sphere_id

class RobotAllegro(RobotBase):
    def __init__(self, tensor_args=None, **kwargs):
        self.robot_name = 'RobotAllegro'
        self.diff_allegro = DifferentiableAllegroHand(device=tensor_args['device'])
        
        # Set joint limits
        self.jl_lower, self.jl_upper, _, _ = self.diff_allegro.get_joint_limit_array()
        q_limits = torch.tensor([self.jl_lower, self.jl_upper], **tensor_args)
        print(f"!!!!!!!q_limits: {q_limits}")
        
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
        # fingertip_poses = self.diff_allegro.compute_forward_kinematics_all_links(q, link_list=self.finger_links)
        return q
    
    def render(self, ax, q=None, color='blue', arrow_length=0.05, arrow_alpha=1.0, arrow_linewidth=2.0, **kwargs):
        """ Render the skeleton of the Allegro hand """
        raise NotImplementedError

    def render_trajectories(self, trajs, visualize=False, collision_traj=False, train_step=None, full_traj=False):
        """Captures 3x3 RGB images from the PyBullet camera for a single Allegro trajectory.

        Args:
            trajs (numpy.ndarray): Shape (N, T, 16), where N = number of trajectories,
                                T = number of waypoints, 16 = Allegro hand joint angles.
            visualize (bool): Whether to visualize the simulation.

        Returns:
            images (list of np.ndarray): A list of 3x3 RGB images captured from the camera.
        """

        # Initialize PyBullet
        if visualize:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # URDF path
        p.setGravity(0, 0, -9.81)

        # Load Allegro Hand URDF
        allegro_id = p.loadURDF(
            "/home/yif/Documents/KTH/git/mpd-cage/deps/torch_robotics/torch_robotics/data/urdf/robots/allegro_hand_description/allegro_hand_description_left.urdf",
            useFixedBase=True,
            globalScaling=10,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )

        # Select the first trajectory and sample 3x3 evenly spaced waypoints
        first_traj = trajs[0]  # Shape: (T, 16)
        num_waypoints = first_traj.shape[0]
        if not full_traj:
            sample_indices = np.linspace(0, num_waypoints - 1, num=3*3, dtype=int)  # Select waypoints
        else:
            sample_indices = np.arange(num_waypoints)

        # Manually set the camera view when using DIRECT mode
        if visualize:
            p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=60, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])
            viewMatrix = p.getDebugVisualizerCamera()[5]  # Extract view matrix from GUI
        else:
            viewMatrix = p.computeViewMatrix(
                cameraEyePosition=[2.8 * np.cos(np.radians(20)), 2.8 * np.sin(np.radians(20)), 0.8],  # Adjust based on Yaw/Pitch
                cameraTargetPosition=[0, 0, 0],  # Looking at origin
                cameraUpVector=[0, 0, 1]  # Z-axis as up direction
            )

        # Common projection matrix (same for both modes)
        projectionMatrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=640/480, nearVal=0.1, farVal=100.0
        )

        width, height, rgbImg, _, _ = p.getCameraImage(
            width=640,
            height=480,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
    
        # Iterate through sampled waypoints
        images = []
        for idx in sample_indices:
            joint_angles = first_traj[idx]  # Get joint configuration at sampled waypoint

            # Apply joint angles to PyBullet (map 16D -> 20D)
            for i, mapped_index in enumerate(JOINT_MAP):
                p.resetJointState(allegro_id, mapped_index, joint_angles[i].item())

            # is_collision, sphere_id = check_self_collision(allegro_id)

            # Step simulation and capture image
            # p.stepSimulation()
            width, height, rgbImg, _, _ = p.getCameraImage(
                width=640,
                height=480,
                viewMatrix=viewMatrix,
                projectionMatrix=projectionMatrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )

            # # Remove the temporary ball
            # if is_collision:
            #     p.removeBody(sphere_id)

            # Convert image to numpy array and append to list
            rgbImg = np.array(rgbImg, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]  # Remove alpha channel
            images.append(rgbImg)

            time.sleep(1) if visualize else None  # Optional delay for visualization

        p.disconnect()

        color_images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]

        # Mark ID in the images
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, idx in enumerate(sample_indices):
            cv2.putText(color_images[i], f"step: {idx}", (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)        

        if not full_traj:
            # Arrange images in 3x3 grid
            final_image = cv2.vconcat([cv2.hconcat(color_images[:3]), cv2.hconcat(color_images[3:6]), cv2.hconcat(color_images[6:])])

            if collision_traj: # put a label on the image bottom right corner
                cv2.putText(final_image, "Collision Trajectory", (10, 80), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(final_image, "Collision-Free Trajectory", (10, 80), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if train_step is not None:
                cv2.putText(final_image, f"Train Step: {train_step}", (10, 130), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imwrite(f"allegro_trajectory_{train_step}.png", cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    
            return np.array(final_image)

        else: # save the full trajectory as a video in mp4 in 15 fps (inference)
            for i, idx in enumerate(sample_indices[:15]):
                if collision_traj: # put a label on the image bottom right corner
                    cv2.putText(color_images[i], "Collision Trajectory", (10, 80), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(color_images[i], "Collision-Free Trajectory", (10, 80), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            out = cv2.VideoWriter(f'allegro_trajectory_inference.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))
            for i in range(len(color_images)):
                out.write(color_images[i])
            out.release()

            return None


def main():
    # Initialize RobotAllegro using PyTorch
    tensor_args = {'device': 'cpu', 'dtype': torch.float32}
    robot_allegro = RobotAllegro(tensor_args=tensor_args)

    # Set an initial joint configuration (16 DOF Allegro Hand)
    # q_init = torch.zeros(16, dtype=torch.float32)
    # q_goal = torch.ones(16, dtype=torch.float32)

    trajs = np.random.rand(1, 64, 16)  # Random trajectory for testing

    # Render a trajectory
    robot_allegro.render_trajectories(trajs, visualize=0, full_traj=True)

if __name__ == "__main__":
    main()