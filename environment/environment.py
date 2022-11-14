# some code from decentralized multiarm multiarm.cs.columbia.edu
#objects from eleramp/pybullet-object-models
import pybullet as pb
import pybullet_utils as pbu
from ur5_robotiq_controller import UR5RobotiqPybulletController
from matplotlib import pyplot as plt
import numpy as np
import os

# physicsClient = pb.connect(pb.GUI, options="--width=1920 --height=1080 --mp4=data.mp4 --mp4fps=40")
physicsClient = pb.connect(pb.GUI, options="--width=1920 --height=1080")
pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 1, lightPosition = [0, 0, 5])
pb.setGravity(0,0,-9.81)
pb.setRealTimeSimulation(True)
pos = [0,0,0]
orn = pb.getQuaternionFromEuler([np.pi/2.0, 0, np.pi/2.0])
pb.loadURDF("./environment/room.urdf", pos, orn, useFixedBase = True)
pos = [4,0,0]
orn = pb.getQuaternionFromEuler([np.pi/2.0,0,0])
pb.loadURDF("./environment/wall.urdf", pos, orn, useFixedBase = True)
pos = [0,0,0]
orn = pb.getQuaternionFromEuler([np.pi/2.0,0,-np.pi/2.0])
pb.loadURDF("./environment/wall.urdf", pos, orn, useFixedBase = True)

pb.resetDebugVisualizerCamera(cameraDistance = 1.7, cameraYaw = 80.0, cameraPitch = -10.0, cameraTargetPosition = [0,0.6,1])

#load robots
orn = pb.getQuaternionFromEuler([np.pi/2.0,-np.pi/2.0,np.pi/2.0])
robot_base_pose = [[0.025, 0.0, 1.194], orn]
robotR = UR5RobotiqPybulletController(base_pose = robot_base_pose, initial_arm_joint_values=[-np.pi/2.0,0,-np.pi/2.0,0,0,0])
robot_base_pose = [[0.025, 0.94, 1.194], orn]
robotL = UR5RobotiqPybulletController(base_pose = robot_base_pose, initial_arm_joint_values=[-np.pi/2.0,0,-np.pi/2.0,0,0,0])
ee_index = 8

pos = [0.079,1.166-0.09,0.560]
orn = pb.getQuaternionFromEuler([np.pi/2.0, 0, np.pi/2.0])
pb.loadURDF("environment/dual_arm_station.urdf", pos, orn, useFixedBase = True)
pos = [0.5,0.47,0.560]
orn = pb.getQuaternionFromEuler([np.pi/2.0, 0, np.pi/2.0])
pb.loadURDF("environment/table.urdf", pos, orn, useFixedBase = True)
# load can
pos = [0.5,0.7,0.610]
orn = pb.getQuaternionFromEuler([np.pi/2.0, 0, 0])
canID = pb.loadURDF("environment/can.urdf", pos, orn)
pbu.step_real(1)

# camera setup
w = 640.0
h = 480.0
projectionMatrix = pb.computeProjectionMatrixFOV(
    fov = 55.0,
    aspect = w/h,
    nearVal = 0.25,
    farVal = 1.9)

# make sure folders exist
os.makedirs('./data', exist_ok = True)
os.makedirs('./data/rgb', exist_ok = True)
os.makedirs('./data/seg', exist_ok = True)
os.makedirs('./data/depth', exist_ok = True)

folder = './data'
f = open(folder + '/labels.csv', 'w')
for i in range(1000):
    # change light position
    lX = np.random.uniform(-1.0, 1.0)
    lY = np.random.uniform(-1.0, 1.0)
    lZ = np.random.uniform(3.0, 5.0)
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 1, lightPosition = [lX, lY, lZ])
    # move the camera
    eyeX = np.random.uniform(0.4, 0.6)
    eyeY = np.random.uniform(-0.3, -0.1)
    eyeZ = np.random.uniform(0.8, 1.0)
    tX = np.random.uniform(0.4, 0.6)
    tY = np.random.uniform(0.5, 0.7)
    tZ = np.random.uniform(0.4, 0.6)
    viewMatrix = pb.computeViewMatrix(
        cameraEyePosition=[eyeX, eyeY, eyeZ],
        cameraTargetPosition=[tX, tY, tZ],
        cameraUpVector=[0, 0, 1])
    # move the can
    x = np.random.uniform(0.2, 0.8)
    y = np.random.uniform(0.7, 1.0)
    pos = [x, y, 0.58]
    orn = pb.getQuaternionFromEuler([np.pi/2.0, 0, 0])
    pb.resetBasePositionAndOrientation(canID, pos, orn)
    pbu.step_real(1)
    # capture pose and images
    pose = pb.getBasePositionAndOrientation(canID)
    width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
        width = 640, 
        height = 480,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        renderer = pb.ER_BULLET_HARDWARE_OPENGL)

    # write pose to csv
    s = str(i) + ', ' + str(pose[0][0]) + ', ' + str(pose[0][1]) + ', ' + \
        str(pose[0][2]) + ', ' + str(np.pi/2.0) + ', ' + str(0.0) + ', ' + str(0.0) + '\n'
    f.write(s)
    # save images
    rgbFile = folder + '/rgb/' + str(i) + '.jpg'
    depthFile = folder + '/depth/' + str(i) + '.npy'
    segFile = folder + '/seg/' + str(i) + '.jpg'
    plt.imsave(rgbFile, rgbImg)
    np.save(depthFile, depthImg)
    plt.imsave(segFile, segImg)

f.close()

pbu.step_real(1)
pb.disconnect()