import argparse
import pyrealsense2 as prs
import cv2
import numpy as np


def add_arguments(parser):
    parser.add_argument('--camera_width', type=int, default=640)
    parser.add_argument('--camera_height', type=int, default=480)
    parser.add_argument('--camera_frame_rate', type=int, default=30)


def run(args, state):
    pipeline = prs.pipeline()
    config = prs.config()

    pipeline_wrapper = prs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    config.enable_stream(prs.stream.color, args.camera_width, args.camera_height, prs.format.rgb8,
                         args.camera_frame_rate)
    config.enable_stream(prs.stream.depth, args.camera_width, args.camera_height, prs.format.z16,
                         args.camera_frame_rate)

    profile = pipeline.start(config)

    align = prs.align(prs.stream.color)
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_image = np.array(depth_frame.data, dtype=np.uint16)
    color_image = np.array(color_frame.data, dtype=np.uint8)

    cv2.imshow('color', color_image)
    cv2.imshow('depth', depth_image)
    cv2.waitKey()

    while True:
        pass


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    run(args, None)


if __name__ == '__main__':
    main()
