import argparse
import pyrealsense2 as rs
import cv2
import numpy as np


def init(args):
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    config.enable_stream(rs.stream.color, args.camera_width, args.camera_height, rs.format.rgb8, args.camera_frame_rate)
    config.enable_stream(rs.stream.depth, args.camera_width, args.camera_height, rs.format.z16, args.camera_frame_rate)

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


def step(args):
    pass


def add_arguments(parser):
    parser.add_argument('--camera_width', type=int, default=640)
    parser.add_argument('--camera_height', type=int, default=480)
    parser.add_argument('--camera_frame_rate', type=int, default=30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    init(args)
    while True:
        step(args)
