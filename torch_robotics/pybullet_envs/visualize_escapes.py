import pybullet as p
import numpy as np
import os
import time
import imageio

class VisualizeEscapePybullet:
    def __init__(self, env, robot, task, visualize=True, 
                **kwargs,):
        self.env = env
        self.robot = robot
        self.task = task

        # Pybullet setup
        if visualize:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setTimeStep(1./240.)
        p.setRealTimeSimulation(0)

        self.obs_pos = self.env.env_position
        self.obs_qtn = self.env.env_quaternion
        self.obs_scale = self.env.obstacle_scale
        self.obs_filename = self.env.env_filename
        self.obj_scale = self.robot.object_scale
        self.obj_filename = self.robot.robot_filename

        # Create object and obstacle
        self.load_object()
        self.add_obstacles()

        # Initialize search parameters
        self.set_camera_view()

    def set_camera_view(self):
        # Set camera parameters to view the y-z plane
        camera_distance = 2  # Distance from the target object
        camera_yaw = 90      # Yaw angle, 90 degrees to face the y-z plane
        camera_pitch = 0     # Pitch angle, keeping it horizontal
        camera_target_position = [0, 0, 0]  # Target position at origin in the y-z plane
        p.resetDebugVisualizerCamera(cameraDistance=camera_distance,
                                     cameraYaw=camera_yaw,
                                     cameraPitch=camera_pitch,
                                     cameraTargetPosition=camera_target_position)
        
    def load_object(self):
        '''
        Load object into pybullet.
        '''
        self.object_id = p.loadURDF(self.obj_filename, 
                                    [0,0,0], 
                                    [0,0,0,1], 
                                    useFixedBase=False, 
                                    globalScaling=self.obj_scale)
    
    def reset_object(self, obj_pos, obj_qtn):
        '''
        Reset object state.
        '''
        p.resetBasePositionAndOrientation(self.object_id, obj_pos, obj_qtn)

    def add_obstacles(self):
        '''
        Add obstacle to pybullet.
        '''
        self.obstacle_id = p.loadURDF(
            fileName=self.obs_filename,
            basePosition=self.obs_pos,
            baseOrientation=self.obs_qtn,
            useFixedBase=True,
            globalScaling=self.obs_scale
        )

    # def make_video(self, video_path):
    #     '''
    #     Save video of the simulation.
    #     '''
    #     frames = []
    #     for i in range(240):
    #         p.stepSimulation()
    #         frame = p.getCameraImage(960, 720, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
    #         frames.append(frame)
    #     frames = np.array(frames)
    #     os.makedirs(os.path.dirname(video_path), exist_ok=True)
    #     np.save(video_path, frames)
    def visualize_escape_pb(self, trajs, make_video=True, video_path='escape.mp4', first_n_trajs=5, fps=25):
        """
        Visualize the trajectories in PyBullet.
        """
        print("Visualizing escape in PyBullet...")
        print(f"trajs.shape: {trajs.shape}")
        first_n_trajs = min(first_n_trajs, trajs.shape[0])
        trajs_pos = trajs[:first_n_trajs,:,:3]
        trajs_euler = trajs[:first_n_trajs,:,3:]
        frames = []
        for traj in range(trajs_pos.shape[0]):
            print(f"ID traj: {traj}")
            for i in range(trajs_pos.shape[1]):
                pos = trajs_pos[traj, i]
                euler = trajs_euler[traj, i]
                quat = p.getQuaternionFromEuler(euler)
                p.resetBasePositionAndOrientation(self.object_id, pos, quat)
                p.stepSimulation()
                time.sleep(0.04)

                if make_video:
                    frame = p.getCameraImage(480, 360, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
                    frames.append(frame)
        
        if make_video and frames:
            video_path = video_path if video_path.endswith('.mp4') else video_path + '.mp4'
            os.makedirs(os.path.dirname(video_path), exist_ok=True)

            # Save frames as a video using imageio
            with imageio.get_writer(video_path, fps=fps, format='mp4', codec='libx264') as writer:
                for frame in frames:
                    writer.append_data(frame)

            print(f"Video saved to {video_path}")

        p.disconnect()

        #     self.make_video(video_path)


