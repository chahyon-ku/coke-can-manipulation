# some code from decentralized multiarm multiarm.cs.columbia.edu
#objects from eleramp/pybullet-object-models
import pybullet as p
import pybullet_utils as pu
import pybullet_data
from ur5_robotiq_controller import UR5RobotiqPybulletController
from matplotlib import pyplot as plt
import numpy as np
import random

PI = 3.14159

# physicsClient = p.connect(p.GUI, options="--width=1920 --height=1080 --mp4=data.mp4 --mp4fps=40")
physicsClient = p.connect(p.GUI, options="--width=1920 --height=1080")
p.configureDebugVisualizer(p.COV_ENABLE_GUI,1, lightPosition = [0, 0, 5])
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)
p.setRealTimeSimulation(True)
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([1.5708,0,1.5708])
p.loadURDF("./environment/room.urdf", startPos, startOrientation, useFixedBase = True)
startPos = [4,0,0]
startOrientation = p.getQuaternionFromEuler([1.5708,0,0])
p.loadURDF("./environment/wall.urdf", startPos, startOrientation, useFixedBase = True)
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([1.5708,0,-1.5708])
p.loadURDF("./environment/wall.urdf", startPos, startOrientation, useFixedBase = True)

p.resetDebugVisualizerCamera(cameraDistance = 1.5, cameraYaw = 80.0, cameraPitch = -10.0, cameraTargetPosition = [0,0.6,1])

#load robots
startOrientation = p.getQuaternionFromEuler([1.5708,-1.5708,1.5708])
robot_base_pose = [[0.025, 0.0, 1.194], startOrientation]
robotR = UR5RobotiqPybulletController(base_pose = robot_base_pose, initial_arm_joint_values=[-1.5708,0,-1.5708,0,0,0])
robot_base_pose = [[0.025, 0.94, 1.194], startOrientation]
robotL = UR5RobotiqPybulletController(base_pose = robot_base_pose, initial_arm_joint_values=[-1.5708,0,-1.5708,0,0,0])
ee_index = 8

startPos = [0.079,1.166-0.09,0.560]
startOrientation = p.getQuaternionFromEuler([1.5708,0,1.5708])
p.loadURDF("environment/dual_arm_station.urdf",startPos, startOrientation, useFixedBase = True)
startPos = [0.5,0.47,0.560]
startOrientation = p.getQuaternionFromEuler([1.5708,0,1.5708])
p.loadURDF("environment/table.urdf",startPos, startOrientation, useFixedBase = True)
# load can
startPos = [0.5,0.7,0.610]
startOrientation = p.getQuaternionFromEuler([1.5708,0,0])
canID = p.loadURDF("environment/can.urdf",startPos, startOrientation)
pu.step_real(1)

# camera setup
res = 256
viewMatrix = p.computeViewMatrix(
    cameraEyePosition=[0.5, -0.2, 0.9],
    cameraTargetPosition=[0.5, 0.7, 0.6],
    cameraUpVector=[1, 0, 0])
projectionMatrix = p.computeProjectionMatrixFOV(
    fov=55.0,
    aspect=1.0,
    nearVal=0.25,
    farVal=1.9)

folder = './data'
f = open(folder + '/labels.csv', 'w')
for i in range(5):
    # move the can
    x = random.uniform(0.2, 0.8)
    y = random.uniform(0.7, 1.0)
    pos = [x, y, 0.58]
    orn = p.getQuaternionFromEuler([1.5708,0,0])
    p.resetBasePositionAndOrientation(canID, pos, orn)
    pu.step_real(1)
    # capture pose and images
    pose = p.getBasePositionAndOrientation(canID)
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=256, 
        height=256,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        renderer = p.ER_BULLET_HARDWARE_OPENGL)

    # write pose to csv
    s = str(i) + ', ' + str(pose[0][0]) + ', ' + str(pose[0][1]) + ', ' + str(pose[0][2]) + '\n'
    f.write(s)
    # save images
    rgbFile = folder + '/rgb/' + str(i) + '.jpg'
    depthFile = folder + '/depth/' + str(i) + '.npy'
    segFile = folder + '/seg/' + str(i) + '.jpg'
    plt.imsave(rgbFile, rgbImg)
    np.save(depthFile, depthImg)
    plt.imsave(segFile, segImg)
f.close()

pu.step_real(1)
p.disconnect()