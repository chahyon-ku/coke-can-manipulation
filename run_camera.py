import argparse
import pyrealsense2 as prs
import cv2
import numpy as np
import all.state


def add_arguments(parser):
    parser.add_argument('--camera_width', type=int, default=960)
    parser.add_argument('--camera_height', type=int, default=540)
    parser.add_argument('--camera_frame_rate', type=int, default=30)


def run(args, state: all.state.State):
    pipeline = prs.pipeline()
    config = prs.config()

    pipeline_wrapper = prs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    config.enable_stream(prs.stream.color, args.camera_width, args.camera_height, prs.format.bgr8,
                         args.camera_frame_rate)
    config.enable_stream(prs.stream.depth, 640, 480, prs.format.z16,
                         args.camera_frame_rate)

    cfg = pipeline.start(config)
    profile = cfg.get_stream(prs.stream.color)  # Fetch stream profile for depth stream
    intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
    intrinsic = np.zeros((3, 3))
    intrinsic[0, 0] = intr.fx
    intrinsic[1, 1] = intr.fy
    intrinsic[0, 2] = intr.ppx
    intrinsic[1, 2] = intr.ppy
    state.set_intrinsic(intrinsic)
    align = prs.align(prs.stream.color)

    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.array(depth_frame.data, dtype=np.uint16)
        color_image = np.array(color_frame.data, dtype=np.uint8)

        if state is not None:
            state.set_color_image(color_image)
            state.set_depth_image(depth_image)

        # cv2.imshow('color', color_image)
        # cv2.imshow('depth', depth_image)
        if state.get_is_grasping():
            break


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    run(args, None)


if __name__ == '__main__':
    main()
