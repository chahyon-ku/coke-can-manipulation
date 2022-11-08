import argparse

import cv2
import numpy as np
import pybullet as pb
import pybullet_data as pbd
import pyrealsense2 as rs


def init_simulation(args):
    physics = pb.connect(pb.GUI, options="--width=1920 --height=1080 --mp4=handoff.mp4 --mp4fps=40")
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
    pb.setGravity(0, 0, -9.81)
    pb.setRealTimeSimulation(True)


def init_camera(width, height, frame_rate):
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, frame_rate)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, frame_rate)

    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)
    # while True:
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_image = np.array(depth_frame.data, dtype=np.uint16)
    color_image = np.array(color_frame.data, dtype=np.uint8)

    cv2.imshow('color', color_image)
    cv2.imshow('depth', depth_image)
    cv2.waitKey()


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # init_simulation(args)
    init_camera(640, 480, 30)


if __name__ == '__main__':
    main()