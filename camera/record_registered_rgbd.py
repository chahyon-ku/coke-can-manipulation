# NOTE:
# This is a very rough and ready piece of code. Please feel free to improve it
# Some basic things it ought to include are:
# - saving the intrinsics and depth scale to a config.
# - separating concerns (code into set of functions)
# - fix the format in which depth is saved.
# - fix the read functions


import cv2
import numpy as np
import pyrealsense2 as rs
import os


def display_and_save_images():
    # TODO: Use this to save intrinsic matrix
    # https://github.com/IntelRealSense/librealsense/issues/869

    # TODO: Check this link to get depth image in meters
    #       save info as meta data
    # https://github.com/IntelRealSense/librealsense/issues/9508

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    device_product_line = str(device.get_info(rs.camera_info.product_line))
    print(device_product_line)
    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.rgb8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    save_counter = 0
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())[:, :, 2::-1]

        cv2.imshow('color', color_image)
        cv2.imshow('depth', depth_image.astype(np.uint16))

        k = cv2.waitKey(20) & 0xFF
        quit_on_key = ['q']
        end_loop = any([k == ord(q) for q in quit_on_key])
        
        if k == 27:
            break
        elif k == ord('s'):
            save_color = lambda im: cv2.imwrite(f'data/{save_counter}_color.png', im)
            save_color(color_image)
            save_gray = lambda im: cv2.imwrite(f'data/{save_counter}_depth.png', im, [cv2.CV_16U])
            save_gray(depth_image)
            save_counter = save_counter + 1
        elif end_loop: break

    cv2.destroyAllWindows()


# SAMPLE
def load_rgb(path):
    assert os.path.isfile(path), f"Path {path} doesn't exist"
    return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

# SAMPLE
def load_depth(path):
    assert os.path.isfile(path), f"Path {path} doesn't exist"
    # return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2GRAY)
    return cv2.imread(path, cv2.IMREAD_ANYDEPTH)



if __name__=='__main__': 
    display_and_save_images()