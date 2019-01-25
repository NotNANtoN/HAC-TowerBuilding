from tkinter import *
from tkinter import ttk
import time
import numpy as np
import copy
import random
from mujoco_py import load_model_from_path, MjSim, MjViewer
from gym.envs.robotics import rotations, utils
from gym.utils import seeding

def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def bound_angle(angle):
    bounded_angle = np.absolute(angle) % (2*np.pi)
    if angle < 0:
        bounded_angle = -bounded_angle
    return bounded_angle

class Environment():

    def __init__(self, model_name, goal_space_train, goal_space_test, project_state_to_end_goal, end_goal_thresholds, initial_state_space, subgoal_bounds, project_state_to_subgoal, subgoal_thresholds, max_actions = 1200, num_frames_skip = 10, show = False):

        self.name = model_name

        # Create Mujoco Simulation
        self.model = load_model_from_path("./mujoco_files/" + model_name)
        self.sim = MjSim(self.model)

        # Set dimensions and ranges of states, actions, and goals in order to configure actor/critic networks
        if model_name == "pendulum.xml":
            self.state_dim = 2*len(self.sim.data.qpos) + len(self.sim.data.qvel)
        else:
            self.state_dim = len(self.sim.data.qpos) + len(self.sim.data.qvel) # State will include (i) joint angles and (ii) joint velocities
        self.action_dim = len(self.sim.model.actuator_ctrlrange) # low-level action dim
        self.action_bounds = self.sim.model.actuator_ctrlrange[:,1] # low-level action bounds
        self.action_offset = np.zeros((len(self.action_bounds))) # Assumes symmetric low-level action ranges
        self.end_goal_dim = len(goal_space_test)
        self.subgoal_dim = len(subgoal_bounds)
        self.subgoal_bounds = subgoal_bounds

        # Projection functions
        self.project_state_to_end_goal = project_state_to_end_goal
        self.project_state_to_subgoal = project_state_to_subgoal


        # Convert subgoal bounds to symmetric bounds and offset.  Need these to properly configure subgoal actor networks
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]


        # End goal/subgoal thresholds
        self.end_goal_thresholds = end_goal_thresholds
        self.subgoal_thresholds = subgoal_thresholds

        # Set inital state and goal state spaces
        self.initial_state_space = initial_state_space
        self.goal_space_train = goal_space_train
        self.goal_space_test = goal_space_test
        self.subgoal_colors = ["Magenta", "Green", "Red", "Blue", "Cyan", "Orange", "Maroon", "Gray", "White", "Black"]

        self.max_actions = max_actions

        # Implement visualization if necessary
        self.visualize = show  # Visualization boolean
        if self.visualize:
            self.viewer = MjViewer(self.sim)
        self.num_frames_skip = num_frames_skip


        # The following variables are hardcoded for the tower building env:
        self.n_objects = 1 # try out 2-3 later
        self.min_tower_height = 1  # try out 2 later
        self.max_tower_height = 4  # try out 2-3 later

        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.block_gripper = False
        self.n_substeps = 20
        self.gripper_extra_height = 0.0
        self.target_in_the_air = True # maybe make dynamic?
        self.target_offset = 0.0
        self.obj_range = 0.15
        self.target_range = 0.15
        self.distance_threshold = 0.05
        self.initial_qpos = {
            'robot0:slide0': 0.0,
            'robot0:slide1': 0.0,
            'robot0:slide2': 0.0,
            'object0:joint': [0.1, 0.0, 0.05, 1., 0., 0., 0.],
            'object1:joint': [0.2, 0.0, 0.05, 1., 0., 0., 0.],
            'object2:joint': [0.3, 0.0, 0.05, 1., 0., 0., 0.],
            'object3:joint': [0.4, 0.0, 0.05, 1., 0., 0., 0.],
            'object4:joint': [0.5, 0.0, 0.05, 1., 0., 0., 0.],
        }
        self.reward_type = 'sparse'
        self.gripper_goal = 'gripper_none' # can be 'gripper_none', 'gripper_random' and 'gripper_above'
        self.goal_size = (self.n_objects * 3)
        if self.gripper_goal != 'gripper_none':
            self.goal_size += 3
        gripperSkipIdx = 0 if self.gripper_goal == "gripper_none" else 3
        objectIdx = gripperSkipIdx + self.n_objects * 3

        self.table_height = 0.5
        self.obj_height = 0.05

        if self.name == "assets/fetch/build_tower.xml":
            self._env_setup(self.initial_qpos)
            self.state_dim = self._get_obs_len()
            self.action_dim = 4
            #self.action_bounds = np.array([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])#self.sim.model.actuator_ctrlrange[:, 1]
            self.action_bounds = np.ones((self.action_dim))
            self.action_offset = np.zeros((self.action_dim))  # Assumes symmetric low-level action ranges

            self.end_goal_dim = self.goal_size
            self.project_state_to_end_goal = self._obs2goal_tower
            self.end_goal_thresholds = [0.05 for i in range(self.goal_size)]

            self.use_full_state_space_as_subgoal_space = True

            if not self.use_full_state_space_as_subgoal_space:
                self.subgoal_dim = self.goal_size
                self.project_state_to_subgoal = self._obs2goal_tower
                self.subgoal_bounds_offset = np.concatenate([[1, 0.25, 0.55] for i in range(self.subgoal_dim // 3)])
                self.subgoal_bounds_symmetric = np.ones((self.subgoal_dim))
                self.subgoal_bounds = [([self.table_height, 1.5] if (i+1) % 3 == 0 else [0, 1.5]) for i in
                                       range(self.subgoal_dim)]
            else:

                #self.subgoal_dim = self.state_dim
                self.subgoal_dim = 3 + self.n_objects * 3
                self.project_state_to_subgoal = self._obs2goal_subgoal
                self.subgoal_bounds = [[0, 1.5] for _ in range(self.subgoal_dim)]
                #self.subgoal_bounds_offset = np.zeros((self.subgoal_dim))
                #objects_offset = np.concatenate([[1, 0.25, 0.55] for i in range(self.n_objects)])
                #self.subgoal_bounds_offset = np.zeros(self.subgoal_dim)
                #self.subgoal_bounds_offset[3:3+3*self.n_objects] = objects_offset
                #self.subgoal_bounds_offset = np.concatenate([[1, 0.25, 0.55] if gripperSkipIdx <= i <= objectIdx
                #                                             else [0, 0, 0] for i in range(0, self.subgoal_dim, 3)])

                #self.subgoal_bounds_symmetric = np.ones((self.subgoal_dim)) * 2
                #self.subgoal_bounds = [([self.table_height, 1.5] if (i+1) % 3 == 0 else [0, 1.5]) if
                #                       3 <= i < 3+3*self.n_objects else [-7, 7] for i in range(self.subgoal_dim)]
            # in ur5 subgoal bound: np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi], [-4, 4], [-4, 4], [-4, 4]])

            # Try the following, borrowed from original code
            self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
            self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

            for i in range(len(self.subgoal_bounds)):
                self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0]) / 2
                self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

            print("Subgoal offset:", self.subgoal_bounds_offset)
            print("Subgoal bounds:", self.subgoal_bounds)
            #pos_threshold = 0.05
            #angle_threshold = np.deg2rad(10)
            #velo_threshold = 2
            #objects_threshold = np.concatenate([[pos_threshold for _ in range(3)] for _ in range(self.n_objects * 2)])
            #objects_rotation = np.concatenate([[angle_threshold for _ in range(3)] for _ in range(self.n_objects)])
            #objects_velo_pos = np.concatenate([[velo_threshold for _ in range(3)] for _ in range(self.n_objects)])
            #self.subgoal_thresholds = np.concatenate((np.array([pos_threshold for i in range(3)]), objects_threshold,
            #                                     np.array([pos_threshold for i in range(2)]), objects_rotation,
            #                                     objects_velo_pos, objects_rotation * 4,
            #                                          np.array([velo_threshold for i in range(3)]),
            #                                          np.array([velo_threshold for i in range(2)])))
            self.subgoal_thresholds = [0.05 for i in range(self.subgoal_dim)]
            print("Subgoal thresholds: ", self.subgoal_thresholds)
            print("Shape thresholds:", np.shape(self.subgoal_thresholds))

            if __debug__:
                print("Action bounds: ", self.action_bounds)

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def get_agent_params(self, agent_params):
        agent_params["subgoal_noise"] = [0.03 for i in range(self.subgoal_dim)]
        return agent_params

    def _obs2goal_tower(self, sim, obs):
        if self.gripper_goal != 'gripper_none':
            goal = obs[:self.goal_size]
        else:
            goal = obs[3:self.goal_size + 3]
        return goal

    def _obs2goal_subgoal(self, sim, obs):
        #return np.concatenate((np.array([bound_angle(sim.data.qpos[i]) for i in range(len(sim.data.qpos))]),
        #     np.array([4 if sim.data.qvel[i] > 4 else -4 if sim.data.qvel[i] < -4 else sim.data.qvel[i] for i in
        #               range(len(sim.data.qvel))])))


        #return obs


        return obs[:3+self.n_objects*3]

    # Get state for tower building env:
    def _get_obs(self):

        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        robot_qpos, robot_qvel = self.sim.data.qpos, self.sim.data.qvel
        object_pos, object_rot, object_velp, object_velr = ([] for _ in range(4))
        object_rel_pos = []

        if self.n_objects > 0:
            for n_o in range(self.n_objects):
                oname = 'object{}'.format(n_o)
                this_object_pos = self.sim.data.get_site_xpos(oname)
                # rotations
                this_object_rot = rotations.mat2euler(self.sim.data.get_site_xmat(oname))
                # velocities
                this_object_velp = self.sim.data.get_site_xvelp(oname) * dt
                this_object_velr = self.sim.data.get_site_xvelr(oname) * dt
                # gripper state
                this_object_rel_pos = this_object_pos - grip_pos
                this_object_velp -= grip_velp

                object_pos = np.concatenate([object_pos, this_object_pos])
                object_rot = np.concatenate([object_rot, this_object_rot])
                object_velp = np.concatenate([object_velp, this_object_velp])
                object_velr = np.concatenate([object_velr, this_object_velr])
                object_rel_pos = np.concatenate([object_rel_pos, this_object_rel_pos])
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.array(np.zeros(3))

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])
        return obs

    # Get state len:
    def _get_obs_len(self):

        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        robot_qpos, robot_qvel = self.sim.data.qpos, self.sim.data.qvel
        object_pos, object_rot, object_velp, object_velr = ([] for _ in range(4))
        object_rel_pos = []

        if self.n_objects > 0:
            for n_o in range(self.n_objects):
                oname = 'object{}'.format(n_o)
                this_object_pos = self.sim.data.get_site_xpos(oname)
                # rotations
                this_object_rot = rotations.mat2euler(self.sim.data.get_site_xmat(oname))
                # velocities
                this_object_velp = self.sim.data.get_site_xvelp(oname) * dt
                this_object_velr = self.sim.data.get_site_xvelr(oname) * dt
                # gripper state
                this_object_rel_pos = this_object_pos - grip_pos
                this_object_velp -= grip_velp

                object_pos = np.concatenate([object_pos, this_object_pos])
                object_rot = np.concatenate([object_rot, this_object_rot])
                object_velp = np.concatenate([object_velp, this_object_velp])
                object_velr = np.concatenate([object_velr, this_object_velr])
                object_rel_pos = np.concatenate([object_rel_pos, this_object_rel_pos])
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.array(np.zeros(3))

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])
        print("obs len: ", len(obs))
        return len(obs)

    # Get state, which concatenates joint positions and velocities
    def get_state(self):

        if self.name == "pendulum.xml":
            return np.concatenate([np.cos(self.sim.data.qpos),np.sin(self.sim.data.qpos),
                               self.sim.data.qvel])
        elif self.name == "assets/fetch/build_tower.xml":
            return self._get_obs()
        else:
            return np.concatenate((self.sim.data.qpos, self.sim.data.qvel))

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos(
            'robot0:grip')

        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        print("inital gripper xpos", self.initial_gripper_xpos)
        if self.n_objects > 0:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def _reset_sim_tower(self):
        self.step_ctr = 0
        self.sim.set_state(self.initial_state)

        # Randomize start position of objects.
        for o in range(self.n_objects):
            oname = 'object{}'.format(o)
            object_xpos = self.initial_gripper_xpos[:2]
            close = True
            while close:
                # self.initial_gripper_xpos[:2]
                object_xpos = [1,0.25] + np.random.uniform(-self.obj_range, self.obj_range,
                                                                                     size=2)
                close = False
                dist_to_nearest = np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2])
                # Iterate through all previously placed boxes and select closest:
                for o_other in range(o):
                    other_xpos = self.sim.data.get_joint_qpos('object{}:joint'.format(o_other))[:2]
                    dist = np.linalg.norm(object_xpos - other_xpos)
                    dist_to_nearest = min(dist, dist_to_nearest)
                if dist_to_nearest < 0.1:
                    close = True

            object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(oname))
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            object_qpos[2] = self.table_height + (self.obj_height / 2) * self.obj_height * 1.05
            self.sim.data.set_joint_qpos('{}:joint'.format(oname), object_qpos)

        self.sim.forward()
        return True

    # Reset simulation to state within initial state specified by user
    def reset_sim(self):

        if self.name == "assets/fetch/build_tower.xml":
            self._reset_sim_tower()
        else:

            # Reset joint positions and velocities
            for i in range(len(self.sim.data.qpos)):
                self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

            for i in range(len(self.sim.data.qvel)):
                self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

            self.sim.step()

        # Return state
        return self.get_state()

    # Set action for tower building env:
    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)
        self.step_ctr += 1

    # Execute low-level action for number of frames specified by num_frames_skip
    def execute_action(self, action):
        if __debug__:
            print("action: ", action)
        if self.name == "assets/fetch/build_tower.xml":
            self._set_action(action)
            #self.sim.data.ctrl[:] = action
        else:
            self.sim.data.ctrl[:] = action
        for _ in range(self.num_frames_skip):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        return self.get_state()


    # Visualize end goal.  This function may need to be adjusted for new environments.
    def display_end_goal(self,end_goal):
        # Goal can be visualized by changing the location of the relevant site object.
        if self.name == "pendulum.xml":
            self.sim.data.mocap_pos[0] = np.array([0.5*np.sin(end_goal[0]),0,0.5*np.cos(end_goal[0])+0.6])
        elif self.name == "assets/fetch/build_tower.xml":
            for i in range(self.n_objects):
                self.model.body_pos[-13 + i] = end_goal[i*3:(i*3)+3]
        elif self.name == "ur5.xml":

            theta_1 = end_goal[0]
            theta_2 = end_goal[1]
            theta_3 = end_goal[2]

            # shoulder_pos_1 = np.array([0,0,0,1])
            upper_arm_pos_2 = np.array([0,0.13585,0,1])
            forearm_pos_3 = np.array([0.425,0,0,1])
            wrist_1_pos_4 = np.array([0.39225,-0.1197,0,1])


            # Transformation matrix from shoulder to base reference frame
            T_1_0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.089159],[0,0,0,1]])

            # Transformation matrix from upper arm to shoulder reference frame
            T_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],[np.sin(theta_1), np.cos(theta_1), 0, 0],[0,0,1,0],[0,0,0,1]])

            # Transformation matrix from forearm to upper arm reference frame
            T_3_2 = np.array([[np.cos(theta_2),0,np.sin(theta_2),0],[0,1,0,0.13585],[-np.sin(theta_2),0,np.cos(theta_2),0],[0,0,0,1]])

            # Transformation matrix from wrist 1 to forearm reference frame
            T_4_3 = np.array([[np.cos(theta_3),0,np.sin(theta_3),0.425],[0,1,0,0],[-np.sin(theta_3),0,np.cos(theta_3),0],[0,0,0,1]])

            # Determine joint position relative to original reference frame
            # shoulder_pos = T_1_0.dot(shoulder_pos_1)
            upper_arm_pos = T_1_0.dot(T_2_1).dot(upper_arm_pos_2)[:3]
            forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
            wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

            joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

            """
            print("\nEnd Goal Joint Pos: ")
            print("Upper Arm Pos: ", joint_pos[0])
            print("Forearm Pos: ", joint_pos[1])
            print("Wrist Pos: ", joint_pos[2])
            """

            for i in range(3):
                self.sim.data.mocap_pos[i] = joint_pos[i]

        else:
            assert False, "Provide display end goal function in environment.py file"

    def _sample_goal_tower(self):
        obs = self._get_obs()
        target_goal = None
        if obs is not None:
            # I changed obs['observation'] into obs because it is just a numpy array
            if self.gripper_goal != 'gripper_none':
                goal = obs.copy()[:self.goal_size]
            else:
                goal = obs.copy()[3:(self.goal_size + 3)]

            if self.gripper_goal != 'gripper_none' and self.n_objects > 0:
                target_goal_start_idx = 3
            else:
                target_goal_start_idx = 0

            stack_tower = (self.max_tower_height - self.min_tower_height + 1) == self.n_objects

            if not stack_tower:
                if self.n_objects > 0:
                    target_range = self.n_objects
                else:
                    target_range = 1
                for n_o in range(target_range):
                    # too_close = True
                    while True:
                        # self.initial_gripper_xpos[:3]
                        target_goal = [1, 0.25, self.table_height + 0.05] + np.random.uniform(-self.target_range,
                                                                                             self.target_range,
                                                                                             size=3)
                        target_goal += self.target_offset
                        rnd_height = random.randint(self.min_tower_height, self.max_tower_height)
                        self.goal_tower_height = rnd_height
                        target_goal[2] = self.table_height + (rnd_height * self.obj_height) - (self.obj_height / 2)
                        too_close = False
                        for i in range(0, target_goal_start_idx, 3):
                            other_loc = goal[i:i + 3]
                            dist = np.linalg.norm(other_loc[:2] - target_goal[:2], axis=-1)
                            if dist < 0.1:
                                too_close = True
                        if too_close is False:
                            break

                    goal[target_goal_start_idx:target_goal_start_idx + 3] = target_goal.copy()
                    target_goal_start_idx += 3
            else:
                target_goal_xy = [1, 0.25] + np.random.uniform(-self.target_range, self.target_range, size=2)
                self.goal_tower_height = self.n_objects
                for n_o in range(self.n_objects):
                    height = n_o + 1
                    target_z = self.table_height + (height * self.obj_height) - (self.obj_height / 2)
                    target_goal = np.concatenate((target_goal_xy, [target_z]))
                    goal[target_goal_start_idx:target_goal_start_idx + 3] = target_goal.copy()
                    target_goal_start_idx += 3

            # Final gripper position
            if self.gripper_goal != 'gripper_none':
                gripper_goal_pos = goal.copy()[-3:]
                if self.gripper_goal == 'gripper_above':
                    gripper_goal_pos[2] += (3 * self.obj_height)
                elif self.gripper_goal == 'gripper_random':
                    too_close = True
                    while too_close:
                        gripper_goal_pos = [1, 0.25, self.table_height + 0.05] + \
                                           np.random.uniform(-self.target_range,
                                                                  self.target_range, size=3)
                        # gripper_goal_pos[0] += 0.25
                        gripper_goal_pos[2] += 0.14
                        if np.linalg.norm(gripper_goal_pos - target_goal, axis=-1) >= 0.1:
                            too_close = False
                goal[:3] = gripper_goal_pos
            return goal.copy()
        else:
            return []

    # Function returns an end goal
    def get_next_goal(self, test):

        end_goal = np.zeros((len(self.goal_space_test)))

        if self.name == "ur5.xml":

            goal_possible = False
            while not goal_possible:
                end_goal = np.zeros(shape=(self.end_goal_dim,))
                end_goal[0] = np.random.uniform(self.goal_space_test[0][0],self.goal_space_test[0][1])

                end_goal[1] = np.random.uniform(self.goal_space_test[1][0],self.goal_space_test[1][1])
                end_goal[2] = np.random.uniform(self.goal_space_test[2][0],self.goal_space_test[2][1])

                # Next need to ensure chosen joint angles result in achievable task (i.e., desired end effector position is above ground)

                theta_1 = end_goal[0]
                theta_2 = end_goal[1]
                theta_3 = end_goal[2]

                # shoulder_pos_1 = np.array([0,0,0,1])
                upper_arm_pos_2 = np.array([0,0.13585,0,1])
                forearm_pos_3 = np.array([0.425,0,0,1])
                wrist_1_pos_4 = np.array([0.39225,-0.1197,0,1])

                # Transformation matrix from shoulder to base reference frame
                T_1_0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.089159],[0,0,0,1]])

                # Transformation matrix from upper arm to shoulder reference frame
                T_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],[np.sin(theta_1), np.cos(theta_1), 0, 0],[0,0,1,0],[0,0,0,1]])

                # Transformation matrix from forearm to upper arm reference frame
                T_3_2 = np.array([[np.cos(theta_2),0,np.sin(theta_2),0],[0,1,0,0.13585],[-np.sin(theta_2),0,np.cos(theta_2),0],[0,0,0,1]])

                # Transformation matrix from wrist 1 to forearm reference frame
                T_4_3 = np.array([[np.cos(theta_3),0,np.sin(theta_3),0.425],[0,1,0,0],[-np.sin(theta_3),0,np.cos(theta_3),0],[0,0,0,1]])

                forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
                wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

                # Make sure wrist 1 pos is above ground so can actually be reached
                if np.absolute(end_goal[0]) > np.pi/4 and forearm_pos[2] > 0.05 and wrist_1_pos[2] > 0.15:
                    goal_possible = True
        elif self.name == "assets/fetch/build_tower.xml":
            end_goal =  self._sample_goal_tower()

        elif not test and self.goal_space_train is not None:
            for i in range(len(self.goal_space_train)):
                end_goal[i] = np.random.uniform(self.goal_space_train[i][0],self.goal_space_train[i][1])
        else:
            assert self.goal_space_test is not None, "Need goal space for testing. Set goal_space_test variable in \"design_env.py\" file"

            for i in range(len(self.goal_space_test)):
                end_goal[i] = np.random.uniform(self.goal_space_test[i][0],self.goal_space_test[i][1])


        # Visualize End Goal
        self.display_end_goal(end_goal)

        return end_goal


    # Visualize all subgoals
    def display_subgoals(self,subgoals):

        # Display up to 10 subgoals and end goal
        if len(subgoals) <= 11:
            subgoal_ind = 0
        else:
            subgoal_ind = len(subgoals) - 11
        
        for i in range(1,min(len(subgoals),11)):
            if self.name == "pendulum.xml":
                self.sim.data.mocap_pos[i] = np.array([0.5*np.sin(subgoals[subgoal_ind][0]),0,0.5*np.cos(subgoals[subgoal_ind][0])+0.6])
                # Visualize subgoal
                self.sim.model.site_rgba[i][3] = 1
                subgoal_ind += 1
            elif self.name == "assets/fetch/build_tower.xml":
                # display only the goal positions of the blocks
                if self.use_full_state_space_as_subgoal_space:
                    #for n in range(self.n_objects):
                    #    self.model.body_pos[-9 + (3 * (i-1)) + n] = self._obs2goal_tower(self.sim, subgoals[i-1])[n*3:(n*3)+3]

                    # Gripper pos:
                    self.model.body_pos[-1] = subgoals[i-1][0:3]
                    # Objects pos:
                    for n in range(self.n_objects):
                        self.model.body_pos[-10 + (3 * (i-1)) + n] = subgoals[i-1][3 + n*3:(n*3) + 6]
                else:
                    for n in range(self.n_objects):
                        self.model.body_pos[-9 + (3 * (i-1)) + n] = subgoals[i-1][n*3:(n*3)+3]
                subgoal_ind += 1
            elif self.name == "ur5.xml":

                theta_1 = subgoals[subgoal_ind][0]
                theta_2 = subgoals[subgoal_ind][1]
                theta_3 = subgoals[subgoal_ind][2]

                # shoulder_pos_1 = np.array([0,0,0,1])
                upper_arm_pos_2 = np.array([0,0.13585,0,1])
                forearm_pos_3 = np.array([0.425,0,0,1])
                wrist_1_pos_4 = np.array([0.39225,-0.1197,0,1])


                # Transformation matrix from shoulder to base reference frame
                T_1_0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.089159],[0,0,0,1]])

                # Transformation matrix from upper arm to shoulder reference frame
                T_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],[np.sin(theta_1), np.cos(theta_1), 0, 0],[0,0,1,0],[0,0,0,1]])

                # Transformation matrix from forearm to upper arm reference frame
                T_3_2 = np.array([[np.cos(theta_2),0,np.sin(theta_2),0],[0,1,0,0.13585],[-np.sin(theta_2),0,np.cos(theta_2),0],[0,0,0,1]])

                # Transformation matrix from wrist 1 to forearm reference frame
                T_4_3 = np.array([[np.cos(theta_3),0,np.sin(theta_3),0.425],[0,1,0,0],[-np.sin(theta_3),0,np.cos(theta_3),0],[0,0,0,1]])

                # Determine joint position relative to original reference frame
                # shoulder_pos = T_1_0.dot(shoulder_pos_1)
                upper_arm_pos = T_1_0.dot(T_2_1).dot(upper_arm_pos_2)[:3]
                forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
                wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

                joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

                """
                print("\nSubgoal %d Joint Pos: " % i)
                print("Upper Arm Pos: ", joint_pos[0])
                print("Forearm Pos: ", joint_pos[1])
                print("Wrist Pos: ", joint_pos[2])
                """

                # Designate site position for upper arm, forearm and wrist
                for j in range(3):     
                    self.sim.data.mocap_pos[3 + 3*(i-1) + j] = np.copy(joint_pos[j])
                    self.sim.model.site_rgba[3 + 3*(i-1) + j][3] = 1

                # print("\nLayer %d Predicted Pos: " % i, wrist_1_pos[:3])

                subgoal_ind += 1
            else:
                # Visualize desired gripper position, which is elements 18-21 in subgoal vector
                self.sim.data.mocap_pos[i] = subgoals[subgoal_ind]
                # Visualize subgoal
                self.sim.model.site_rgba[i][3] = 1
                subgoal_ind += 1
