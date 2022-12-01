import os
import numpy as np
from termcolor import cprint
import pybullet as pb

from pybullet_planning import load_pybullet, connect, wait_for_user, LockRenderer, has_gui, WorldSaver, HideOutput, \
    reset_simulation, disconnect, set_camera_pose, has_gui, set_camera, wait_for_duration, wait_if_gui, apply_alpha
from pybullet_planning import Pose, Point, Euler
from pybullet_planning import get_num_joints, get_joint_names, get_movable_joints, set_joint_positions, joint_from_name, \
    joints_from_names, get_sample_fn, plan_joint_motion, plan_cartesian_motion, link_from_name
from pybullet_planning import dump_world, set_pose
from pybullet_planning import get_collision_fn, get_floating_body_collision_fn, expand_links, create_box

connect(use_gui=True, width=1920, height=1080)
pb.resetDebugVisualizerCamera(cameraDistance = 2, cameraYaw = 70.0, cameraPitch = -20.0, cameraTargetPosition = [0,0.5,0.5])
pos = [0,0,0]
orn = pb.getQuaternionFromEuler([np.pi/2.0, 0, np.pi/2.0])
pb.loadURDF("./simulation/room.urdf", pos, orn, useFixedBase = True)
pos = [4,0,0]
orn = pb.getQuaternionFromEuler([np.pi/2.0,0,0])
pb.loadURDF("./simulation/wall.urdf", pos, orn, useFixedBase = True)
pos = [0,0,0]
orn = pb.getQuaternionFromEuler([np.pi/2.0,0,-np.pi/2.0])
pb.loadURDF("./simulation/wall.urdf", pos, orn, useFixedBase = True)
pos = [0.079,1.166-0.09,0.560]
orn = pb.getQuaternionFromEuler([np.pi/2.0, 0, np.pi/2.0])
pb.loadURDF("environment/dual_arm_station.urdf", pos, orn, useFixedBase = True)
pos = [0.5,0.47,0.560]
orn = pb.getQuaternionFromEuler([np.pi/2.0, 0, np.pi/2.0])
table = pb.loadURDF("environment/table.urdf", pos, orn, useFixedBase = True)

HERE = os.path.dirname(__file__)
urdf = os.path.join(HERE, '../assets', 'ur5', 'ur5_robotiq.urdf')
orn = pb.getQuaternionFromEuler([np.pi/2.0,-np.pi/2.0,np.pi/2.0])
base_pose = [[0.025, 0.94, 1.42], orn]
robot = pb.loadURDF(urdf,
                    basePosition=base_pose[0],
                    baseOrientation=base_pose[1],
                    useFixedBase=True,
                    flags=pb.URDF_USE_SELF_COLLISION)

arm_joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
arm_joints = joints_from_names(robot, arm_joint_names)
# * if a subset of joints is used, use:
# arm_joints = joints_from_names(robot, arm_joint_names[1:]) # this will disable the gantry-x joint
cprint('Used joints: {}'.format(get_joint_names(robot, arm_joints)), 'yellow')

# * get a joint configuration sample function:
# it separately sample each joint value within the feasible range
sample_fn = get_sample_fn(robot, arm_joints)

# * now let's plan a trajectory
print('Disabled collision links needs to be given (can be parsed from a SRDF via compas_fab)')

ik_tool_link = link_from_name(robot, 'tool_tip_link')
ee_poses = []
pose1 = Pose(point = np.array([0.7, 0.6, 0.7]), euler=Euler(0, 0, 0))
pose2 = Pose(point = np.array([0.6, 0.6, 0.7]), euler=Euler(0, 0, 0))
ee_poses.append(pose1)
ee_poses.append(pose2)
base_link = link_from_name(robot, 'base_link')
path = plan_cartesian_motion(robot, base_link, ik_tool_link, ee_poses)

if path is None:
    cprint('Gradient-based ik cartesian planning cannot find a plan!', 'red')
else:
    cprint('Gradient-based ik cartesian planning find a plan!', 'green')
    time_step = 0.03
    for conf in path:
        set_joint_positions(robot, arm_joints, conf)
        wait_for_duration(time_step)
