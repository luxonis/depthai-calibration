#!/usr/bin/env python3

description=\
"""
Dynamic recalibration script.
Capable of correcting extrinsic rotation (e.g. rotation change between sensors) without the need of full recalibration.
Recommended way of doing dynamic calibration is pointing the camera to a static scene, and running the script.
Recommended to try dynamic calibration if depth quality degraded over time.

Requires initial intrinsic calibration.
This script supports all sensor combinations that calibrate.py supports.
"""
import os
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/sachin/.local/lib/python3.10/site-packages/cv2/qt/plugins'

from cmath import inf
import numpy as np
import depthai as dai
import math
import argparse
from pathlib import Path
import time
import traceback
from feature_helper import FeaturesHelper, keypoint_to_point2f
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('TkAgg')  # Use the Tk backend
import matplotlib.pyplot as plt

import cv2

# plt = matplotlib.pyplot

epilog_text="Dynamic recalibration."
parser = argparse.ArgumentParser(
    epilog=epilog_text, description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('-vd', '--videoDir', type=str, default=None, help='Path to video directory')

parser.add_argument("-rd", "--rectifiedDisp", default=True, action="store_false",
                    help="Display rectified images with lines drawn for epipolar check")
parser.add_argument("-drgb", "--disableRgb", default=False, action="store_true",
                    help="Disable rgb camera Calibration")
parser.add_argument("-ep", "--maxEpiploarError", default="1.0", type=float, required=False,
                    help="Sets the maximum epiploar allowed with rectification")
parser.add_argument("-rlp", "--rgbLensPosition", default=None, type=int,
                    required=False, help="Set the manual lens position of the camera for calibration")
parser.add_argument("-fps", "--fps", default=10, type=int,
                    required=False, help="Set capture FPS for all cameras. Default: %(default)s")
parser.add_argument("-d", "--debug", default=False, action="store_true", help="Enable debug logs.")
parser.add_argument("-dr", "--dryRun", default=False, action="store_true", help="Dry run, don't flash obtained calib data, just save to disk.")
options = parser.parse_args()

#TODO implement RGB-stereo sync

epipolar_threshold = options.maxEpiploarError
rgbEnabled = not options.disableRgb
dryRun = options.dryRun
debug = options.debug

ransacMethod = cv2.RANSAC
# if cv2.__version__ >= "4.5.4":
#     ransacMethod = cv2.USAC_MAGSAC
#     print('Using MAGSAC')

feature_helper = FeaturesHelper(0.3, 1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Adding additional constaints that apply to stereo normal setups. 
def retrive_rt_from_essential_mat(E, pts1, pts2, k1, k2, d1, d2, T):
    p1 = k1 @ np.eye(3, 4)
    u_pts1 = cv2.undistortPoints(pts1, k1, d1, P = p1).reshape(-1, 2)
    u_pts2 = cv2.undistortPoints(pts2, k2, d2, P = p1).reshape(-1, 2)
    t_org = T[:3, 3].reshape(-1, 1)
    R_org = T[:3, :3]
    rot = Rotation.from_matrix(R_org)
    r_org, p_org, y_org = rot.as_euler("xyz", degrees=True)

    R1, R2, t_e = cv2.decomposeEssentialMat(E)
    t_est_dir = t_e / np.linalg.norm(t_e)
    t_org_dir = t_org / np.linalg.norm(t_org)

    combs = []
    for r, t in [(R1, t_e), (R2, t_e)]:
        dot_product = np.dot(t_est_dir.T, t_org_dir)
        angle_rad = np.arccos(dot_product)
        # Angle between vectors in degrees
        angle_deg = np.degrees(angle_rad)
        print(f'dot product is {dot_product}')
        print(f'angle is {angle_deg}')

        if angle_deg < 185 and angle_deg > 175:
            t = -t
        elif angle_deg < 5 and angle_deg > -5:
            pass
        else:
            continue
        rot = Rotation.from_matrix(r)
        r_e, p_e, y_e = rot.as_euler("xyz", degrees=True)
        if abs(r_e - r_org) > 8 or abs(p_e - p_org) > 8 or abs(y_e - y_org) > 8:
            continue

        combs.append((r, t))


    res = (None, None, None)
    for r, t in combs:
        p2 = k1 @ np.hstack([r, t])
        print(f'p2 is {p2}')
        rot = Rotation.from_matrix(r)
        print(f'Rotation matrix is {rot.as_euler("xyz", degrees=True)}')
        print(f'translation is {t}')
        triangulated_pts = cv2.triangulatePoints(p1, p2, u_pts1.T, u_pts2.T)
        triangulated_pts = triangulated_pts.T
        points_euclidean = triangulated_pts[:, :3] / triangulated_pts[:, 3:4]
        # if np.all(points_euclidean[:, 2] > 0):
        #     points_euclidean = points_euclidean[points_euclidean[:, 2] <= 3]
        #     res = (r, t, points_euclidean)
        print(f'Filtered points shape is {points_euclidean.shape}')
        print(f'Median of z in filterd pts is {np.median(points_euclidean[:, 2])}')
    # exit(1)

    return res









    return R_est, t_est

# TODO: Track features across multiple frames and 
# remove outliers and then use the features with better age to estimate the rotation.   
def calculate_Rt_from_frames(frame1, frame2, k1, k2, d1, d2, T):
    kps1, kps2, _, _ = feature_helper.getMatchedFeatures(frame1, frame2)
    minKeypoints = 20
    t_original = T[:3, 3].reshape(-1, 1)
    t_norm = np.linalg.norm(t_original)
    print(f'Original t shape: {t_original.shape}')
    r_original = T[:3, :3]
    if len(kps1) < minKeypoints:
        print(f'Need at least {minKeypoints} keypoints!')
        return None, None, None, None, None, None, None

    if debug:
        img_frame = feature_helper.draw_features(frame1, frame2, kps1, kps2)
        cv2.imshow("marked image ", img_frame)
        # img=cv2.drawKeypoints(frame1, kps1, frame1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow("Left", img)
        # img2=cv2.drawKeypoints(frame2, kps2, frame2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow("Right", img2)
        cv2.waitKey(1)

    # print(f'Type of pts1 is {type(pts1)}')
    pts1 = keypoint_to_point2f(kps1)
    pts2 = keypoint_to_point2f(kps2)
    method = 1
    # if method:
    E, mask = cv2.findEssentialMat(pts1, pts2, k1, d1, k2, d2, method=ransacMethod, prob=0.999, threshold=0.7)

    u_pts1 = cv2.undistortPoints(pts1, k1, d1).reshape(-1, 2)
    u_pts2 = cv2.undistortPoints(pts2, k2, d2).reshape(-1, 2)

    normalized_k = np.eye(3)
    ret, R_est, t_est, mask_pose, triangulated_pts = cv2.recoverPose(E, u_pts1, u_pts2, normalized_k, distanceThresh = 1, mask=mask)
    # ret, R_est, t_est, mask_pose, triangulated_pts = cv2.recoverPose(E, u_pts1, u_pts2, normalized_k, distanceThresh = 2, mask=mask)
    t_est *= t_norm
    print(f'estimated t: {t_est}')

    if 0:
        # ret, E, R_est, t_est, mask_pose = cv2.recoverPose(pts1, pts2, k1, d1, k2, d2, method=ransacMethod, prob=0.999, threshold=0.3)
        rot = Rotation.from_matrix(R_est)
        print(f'Ret of recoverPose is {ret}')
        print(f'Rotation LR matrix from recoverPose  {rot.as_euler("xyz", degrees=True)}')

        R1, R2, t_e = cv2.decomposeEssentialMat(E)
        print(f'From decompose Essential mat')
        rot = Rotation.from_matrix(R1)
        print(f'Rotation R1 from decomposeEssentialMat  {rot.as_euler("xyz", degrees=True)}')
        rot = Rotation.from_matrix(R2)
        print(f'Rotation R2 from decomposeEssentialMat  {rot.as_euler("xyz", degrees=True)}')
        print(f't_e from decomposeEssentialMat  {t_e*t_norm}')
        print(f'neg t_e from decomposeEssentialMat  {-t_e*t_norm}')

        print(f'Keypoints size is {pts1.shape}')
        triangulated_pts = triangulated_pts.T
        points_euclidean = triangulated_pts[:, :3] / triangulated_pts[:, 3:4]
        print(f'size of points_euclidean is ----------------------> {points_euclidean.shape}')

    R_est_x, t_est_x, points_euclidean = retrive_rt_from_essential_mat(E, pts1, pts2, k1, k2, d1, d2, T)
    if R_est_x is not None:
        rot = Rotation.from_matrix(R_est_x)
        print(f'~~~~~~~~~~~~~~~~~~Rotation LR matrix from retrive_rt_from_essential_mat  {rot.as_euler("xyz", degrees=True)}')
        print(f'~~~~~~~~~~~~~~~~~estimated t retrive_rt_from_essential_mat: {t_est_x}')
    else:
        print(f'~~~~~~~~~~~~~~~~Could not find a valid rotation matrix')

    if points_euclidean is not None:
        ax.clear()
        ax.scatter(points_euclidean[:, 0], points_euclidean[:, 1], points_euclidean[:, 2])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.draw()
        plt.pause(1)  # wait for 1 second
    # cv2.waitKey(0)

        input("Press Enter to continue...")

    # Homogenous version of undistorted points
    pts1_h = np.hstack([u_pts1, np.ones((u_pts1.shape[0], 1))])
    pts2_h = np.hstack([u_pts2, np.ones((u_pts2.shape[0], 1))])

    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(k1, d1, k2, d2, frame2.shape[::-1], R_est, t_est)
    constraint = np.diagonal(np.dot(pts2_h, np.dot(E, pts1_h.T)))
    # print(f'constraint shape is {constraint.shape}')
    # print(f'constraints are  {constraint}')
    # print(f'constraint mean is {np.mean(constraint)}')
    # print(f'constraint std is {np.std(constraint)}')
    # print(f'constraint min is {np.min(constraint)}')
    # print(f'constraint max is {np.max(constraint)}')


    return R_est, t_est, R1, R2, P1, P2, Q

def publish_input_frames(videoHandlers):
    ts = dai.Clock.now()
    endOfVideo = False
    for xLinkName in videoHandlers.keys():
        xLink = videoHandlers[xLinkName]

        if (not xLink["video"].isOpened()) or endOfVideo:
            endOfVideo = True
            xLink["video"].release()
            if debug:
                print("Video " + xLinkName + " finished")
                print(f'Path is {xLink["videoPath"]}')
            xLink["video"] = cv2.VideoCapture(xLink['videoPath'])
        read_correctly, frame = xLink["video"].read()
        if not read_correctly:
            print(f'{xLinkName} Not read correctly')
            endOfVideo = True
            xLink["video"].release()
            if debug:
                print("Video " + xLinkName + " finished")
                print(f'Path is {xLink["videoPath"]}')
            xLink["video"] = cv2.VideoCapture(xLink['videoPath'])

            continue

        # if xLinkName == "color":
        #     frame = cv2.resize(frame, rgbSize)
        height = frame.shape[0]
        if height == 800:
            crop_top = 40
            crop_bottom = 800 - 40
            frame = frame[crop_top:crop_bottom, :]
        elif height == 1080:
            frame = cv2.resize(frame, (1280, 720))

        width = frame.shape[1]
        height = frame.shape[0]
        channels = frame.shape[2]
        img = dai.ImgFrame()
        img.setTimestamp(ts)
        img.setWidth(width)
        img.setHeight(height)

        if xLinkName == "color":
            img.setType(dai.ImgFrame.Type.BGR888p)
            img.setData(frame.transpose(2, 0, 1).flatten())
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img.setType(dai.ImgFrame.Type.RAW8)
            img.setData(frame.reshape(height*width))

        if xLinkName == "left":
            img.setInstanceNum(dai.CameraBoardSocket.CAM_B)
            # left_frame = frame
            # if debug:
            #     left_frame_rect = cv2.remap(left_frame, map_left.map_x, map_left.map_y, cv2.INTER_LINEAR)
            #     cv2.imshow("left-rect", left_frame_rect)

        elif xLinkName == "right":
            img.setInstanceNum(dai.CameraBoardSocket.CAM_C)
            # right_frame = frame
            # print(f'shape is {right_frame.shape}')
            # if debug:
            #     right_frame_mod = cv2.remap(right_frame, map_right.map_x, map_right.map_y, cv2.INTER_LINEAR)
            #     cv2.imshow("right-rect", right_frame_mod)

        xLink["queue"].send(img)


def create_pipeline(videoDir, calibHandler, rgbEnabled):
    pipeline = dai.Pipeline()
    pipeline.setCalibrationData(calibHandler)

    videoLinks = {}
    if videoDir:
        xLinks = {}
        if rgbEnabled:
            input_cams = ["left", "right", "color"]
        else:
            input_cams = ["left", "right"]

        for xLinkName in input_cams:
            xLink = pipeline.create(dai.node.XLinkIn)
            xLink.setStreamName(xLinkName)
            xLinks[xLinkName] = {}
            videoLinks[xLinkName] = {}
            xLinks[xLinkName]["node"] = xLink
            videoPath = str(Path(videoDir) / (xLinkName + ".avi"))
            if not Path(videoPath).exists():
                videoPath = str(Path(videoDir) /
                                (xLinkName + ".avi"))
                if not Path(videoPath).exists():
                    print("ERROR: Video file not found: ", videoPath)
                    exit(1)
            videoLinks[xLinkName]["video"] = cv2.VideoCapture(videoPath)
            videoLinks[xLinkName]["videoPath"] = videoPath

        stereoLeftIn = xLinks["right"]["node"].out
        stereoRightIn = xLinks["left"]["node"].out
        if rgbEnabled:
            rgbIn = xLinks["color"]["node"].out
    else:

        cam_left = pipeline.create(dai.node.MonoCamera)
        cam_right = pipeline.create(dai.node.MonoCamera)
        cam_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        # leftFps = cam_left.getFps()
        # rightFps = cam_right.getFps()

        cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        cam_left.setFps(camFps)

        cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        cam_right.setFps(camFps)

        stereoLeftIn = cam_left.out
        stereoRightIn = cam_right.out
        
        if rgbEnabled:
            rgbLensPosition = None

            if options.rgbLensPosition:
                rgbLensPosition = options.rgbLensPosition
            else:
                try:
                    rgbLensPosition = calibration_handler.getLensPosition(dai.CameraBoardSocket.CAM_A)
                except:
                    pass

            rgb_cam = pipeline.createColorCamera()
            rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
            rgb_cam.setInterleaved(False)
            rgb_cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            rgb_cam.setIspScale(1, 3)

            if rgbLensPosition:
                rgb_cam.initialControl.setManualFocus(rgbLensPosition)
            rgb_cam.setFps(camFps)
            rgbIn = rgb_cam.isp

    xout_left = pipeline.create(dai.node.XLinkOut)
    xout_right = pipeline.create(dai.node.XLinkOut)

    xout_left_rect = pipeline.create(dai.node.XLinkOut)
    xout_right_rect = pipeline.create(dai.node.XLinkOut)
    xout_left_rect.setStreamName("left_rect")
    xout_right_rect.setStreamName("right_rect")

    xout_left.setStreamName("left")
    # stereoLeftIn.link(xout_left.input)

    xout_right.setStreamName("right")
    # stereoRightIn.link(xout_right.input)

    stereo = pipeline.create(dai.node.StereoDepth)

    stereoLeftIn.link(stereo.left)
    stereoRightIn.link(stereo.right)

    stereo.syncedLeft.link(xout_left.input)
    stereo.syncedRight.link(xout_right.input)

    stereo.rectifiedLeft.link(xout_left_rect.input)
    stereo.rectifiedRight.link(xout_right_rect.input)
    
    if rgbEnabled:
        xout_rgb_isp = pipeline.create(dai.node.XLinkOut)
        xout_rgb_isp.setStreamName("rgb")
        rgbIn.link(xout_rgb_isp.input)


    # if leftFps != rightFps:
    #     raise Exception("FPS between left and right cameras must be the same!")

    return pipeline, videoLinks

if __name__ == "__main__":

    camFps = options.fps
    device = dai.Device()

    try:
        if options.videoDir is None:
            calibration_handler = device.readCalibration()
            original_calibration = device.readCalibration()
        else:
            calib_path = Path(options.videoDir) / "calib.json"
            calibration_handler = dai.CalibrationHandler(calib_path)
            original_calibration = dai.CalibrationHandler(calib_path)
    except Exception as e:
        print("Dynamic recalibration requires initial intrinsic calibration!")
        raise e

    pipeline, videoHandlers = create_pipeline(options.videoDir, calibration_handler, rgbEnabled)

    with device:
        device.startPipeline(pipeline)

        left_camera_queue = device.getOutputQueue("left", 4, True)
        right_camera_queue = device.getOutputQueue("right", 4, True)
        left_rectified_camera_queue = device.getOutputQueue("left_rect", 4, True)
        right_rectified_camera_queue = device.getOutputQueue("right_rect", 4, True)

        if rgbEnabled:
            rgb_camera_queue = device.getOutputQueue("rgb", 4, True)
        
        if options.videoDir is not None:
            for xLinkName in videoHandlers.keys():
                videoHandlers[xLinkName]["queue"] = device.getInputQueue(xLinkName)

        left_camera = dai.CameraBoardSocket.CAM_B
        right_camera = dai.CameraBoardSocket.CAM_C
        rgb_camera = dai.CameraBoardSocket.CAM_A

        left_rect_frame = None
        right_rect_frame = None
        left_frame = None
        right_frame = None
        rgb_frame = None

        start_time = time.time()
        elapsed_time = time.time() - start_time
        while options.videoDir is None and elapsed_time < 5:
            leftFrameData = left_camera_queue.get()
            left_frame = leftFrameData.getCvFrame()
            rightFrameData = right_camera_queue.get()
            right_frame = rightFrameData.getCvFrame()

            stereo_img_shape = (leftFrameData.getWidth(), leftFrameData.getHeight())

            if rgbEnabled:
                rgbFrameData = rgb_camera_queue.get()
                rgb_frame = rgbFrameData.getCvFrame()
                rgb_img_shape = (rgbFrameData.getWidth(), rgbFrameData.getHeight())
            elapsed_time = time.time() - start_time
        if options.videoDir is not None:
            width = videoHandlers["left"]["video"].get(cv2.CAP_PROP_FRAME_WIDTH)
            height = videoHandlers["left"]["video"].get(cv2.CAP_PROP_FRAME_HEIGHT)
            # stereo_img_shape = (int(width), int(height))
            stereo_img_shape = (1280, 720)
            # print(f'width is {width}, height is {height}')
            # exit(1)
            if rgbEnabled:
                width = videoHandlers["color"]["video"].get(cv2.CAP_PROP_FRAME_WIDTH)
                height = videoHandlers["color"]["video"].get(cv2.CAP_PROP_FRAME_HEIGHT)
                # rgb_img_shape = (int(width), int(height))
                rgb_img_shape = (1280, 720)
                # print(f'width is {width}, height is {height}')
                # exit(1)

        left_k = calibration_handler.getCameraIntrinsics(left_camera, stereo_img_shape[0], stereo_img_shape[1])
        right_k = calibration_handler.getCameraIntrinsics(right_camera, stereo_img_shape[0], stereo_img_shape[1])

        left_d = calibration_handler.getDistortionCoefficients(left_camera)
        right_d = calibration_handler.getDistortionCoefficients(right_camera)

        left_k = np.array(left_k)
        right_k = np.array(right_k)

        left_d = np.array(left_d)
        right_d = np.array(right_d)
        original_t = calibration_handler.getCameraExtrinsics(left_camera, right_camera, False)
        original_t_lr = np.array(original_t)
        if rgbEnabled:
            rgb_k = calibration_handler.getCameraIntrinsics(rgb_camera, rgb_img_shape[0], rgb_img_shape[1])
            rgb_k = np.array(rgb_k)
            rgb_d = calibration_handler.getDistortionCoefficients(rgb_camera)
            rgb_d = np.array(rgb_d)
            original_t = calibration_handler.getCameraExtrinsics(rgb_camera, right_camera, False)
            original_t_rgb = np.array(original_t)

        estimate_counts = 0
        estimates = {} 
        estimates["lr"] = []
        estimates["rgb"] = []
        epipolar_list = {}
        epipolar_list["lr"] = {}
        epipolar_list["rgb"] = {}
        while True:
            try:
                if options.videoDir is not None:
                    publish_input_frames(videoHandlers)

                leftFrameData = left_camera_queue.get()
                rightFrameData = right_camera_queue.get()

                left_frame = leftFrameData.getCvFrame()
                right_frame = rightFrameData.getCvFrame()

                left_rect_frame = left_rectified_camera_queue.get().getCvFrame()
                right_rect_frame = right_rectified_camera_queue.get().getCvFrame()

                # if debug:
                #     cv2.imshow('left', left_frame)
                #     cv2.imshow('right', right_frame)
                #     cv2.waitKey(0)

                if len(left_frame.shape) != 2:
                    left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                if len(right_frame.shape) != 2:
                    right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
                left_mean = np.mean(left_frame)
                right_mean = np.mean(right_frame)

                print(f'left mean: {left_mean}, right mean: {right_mean}')
                if left_mean > 200 or right_mean > 200:
                    cv2.waitKey(1)
                    continue

                estimate_counts += 1
                if estimate_counts < 20:
                    print(f'Original T LR -> {original_t_lr[:3, 3]}')
                    R, T, R1, R2, P1, P2, Q = calculate_Rt_from_frames(left_frame, right_frame, left_k, right_k, left_d, right_d, original_t_lr)
                    if R is not None:
                        estimates["lr"].append((R, T, R1, R2, P1, P2, Q))

                    # rot = Rotation.from_matrix(R1)
                    # print(f'Rotation Left matrix is {rot.as_euler("xyz", degrees=True)}')
                    # rot = Rotation.from_matrix(R2)
                    # print(f'Rotation Right matrix is {rot.as_euler("xyz", degrees=True)}')
                    if rgbEnabled:
                        print(f'Original T RGB-R -> {original_t_rgb[:3, 3]}')

                        rgb_frame = rgb_camera_queue.get().getCvFrame()
                        R, T, R1, R2, P1, P2, Q = calculate_Rt_from_frames(rgb_frame, right_frame, rgb_k, right_k, rgb_d, right_d, original_t_rgb)
                        # rgbR = np.linalg.inv(rgbR) #right to rgb rotation
                        estimates["rgb"].append((R, T, R1, R2, P1, P2, Q))

                        # rot = Rotation.from_matrix(R1)
                        # print(f'Rotation Left RGB matrix is {rot.as_euler("xyz", degrees=True)}')
                        # rot = Rotation.from_matrix(R2)
                        # print(f'Rotation Right RGB matrix is {rot.as_euler("xyz", degrees=True)}')

                else:
                    print(f'Count estimate is {estimate_counts}')
                    img_shape = left_rect_frame.shape[::-1]
                    M1 = left_k
                    M2 = right_k
                    d1 = left_d
                    d2 = right_d

                    for i in range(len(estimates["lr"])):
                        R, T, R1, R2, P1, P2, Q = estimates["lr"][i]
                        rot = Rotation.from_matrix(R1)
                        print(f'Rotation Left matrix is {rot.as_euler("xyz", degrees=True)}')
                        rot = Rotation.from_matrix(R2)
                        print(f'Rotation Right matrix is {rot.as_euler("xyz", degrees=True)}')

                        mapx_l, mapy_l = cv2.initUndistortRectifyMap(M1, d1, R1, M2, img_shape, cv2.CV_32FC1)
                        mapx_r, mapy_r = cv2.initUndistortRectifyMap(M2, d2, R2, M2, img_shape, cv2.CV_32FC1)

                        img_l = cv2.remap(left_frame, mapx_l, mapy_l, cv2.INTER_LINEAR)
                        img_r = cv2.remap(right_frame, mapx_r, mapy_r, cv2.INTER_LINEAR)
                        # print("Rect image shape", img_r.shape)
                        # print('Original shape', left_frame.shape)
                        # cv2.imshow('Rect left', img_l)
                        # cv2.imshow('Rect right', img_r)

                        stereo_epipolar, disp_image = feature_helper.calculate_epipolar_error(img_l, img_r)
                        if i not in epipolar_list["lr"]:
                            epipolar_list["lr"][i] = []
                        epipolar_list["lr"][i].append(stereo_epipolar)
                        if debug:
                            cv2.imshow("lr", disp_image)
                        cv2.waitKey(2)
                        # if stereo_epipolar > epipolar_threshold:
                        #     print(f"Stereo epipolar error: {stereo_epipolar} is higher than threshold {epipolar_threshold}")
                        #     continue

                    if rgbEnabled:
                        M3 = rgb_k
                        d3 = rgb_d

                        for i in range(len(estimates["rgb"])):
                            R, T, R1, R2, P1, P2, Q = estimates["rgb"][i]
                            mapx_rgb, mapy_rgb = cv2.initUndistortRectifyMap(M3, d3, R1, P1, img_shape, cv2.CV_32FC1)
                            mapx_rgb2, mapy_rgb2 = cv2.initUndistortRectifyMap(M2, d2, R2, P2, img_shape, cv2.CV_32FC1)

                            img_rgb = cv2.remap(rgb_frame, mapx_rgb, mapy_rgb, cv2.INTER_LINEAR)
                            img_rgb2 = cv2.remap(right_frame, mapx_rgb2, mapy_rgb2, cv2.INTER_LINEAR)

                            rgb_epipolar, disp_image = feature_helper.calculate_epipolar_error(img_rgb, img_rgb2)
                            if i not in epipolar_list["rgb"]:
                                epipolar_list["rgb"][i] = []
                            epipolar_list["rgb"][i].append(rgb_epipolar)
                            if debug:
                                cv2.imshow("rgb", disp_image)
                            cv2.waitKey(1)
                            # if rgb_epipolar > epipolar_threshold:
                            #     print(f"RGB epipolar {rgb_epipolar} is higher than threshold {epipolar_threshold}")
                            #     continue
                if estimate_counts >= 30:
                    print("Breaking------------------")

                    break
            except Exception as e:
                print(e)
                cv2.waitKey(1)
                # tb = traceback.format_exc()
                # error_line = tb.split("\n")[-3]
                # print(tb)
                traceback.print_exc()
                exit(1)
                continue

        for i in range(len(estimates["lr"])):
            print(f'lr epipolar errors from {i}\'th estimation are : {epipolar_list["lr"][i]}')
        for i in range(len(estimates["rgb"])):
            print(f'rgb-r epipolar errors from {i}\'th estimation are : {epipolar_list["rgb"][i]}')
        
        exit(1)
        #save rotation data
        lrSpecExtrinsics = calibration_handler.getCameraExtrinsics(left_camera, right_camera, True)
        specTranslation = (lrSpecExtrinsics[0][3], lrSpecExtrinsics[1][3], lrSpecExtrinsics[2][3])
        lrCompExtrinsics = calibration_handler.getCameraExtrinsics(left_camera, right_camera, False)
        compTranslation = (lrCompExtrinsics[0][3], lrCompExtrinsics[1][3], lrCompExtrinsics[2][3])
        calibration_handler.setCameraExtrinsics(left_camera, right_camera, R, compTranslation, specTranslation)

        calibration_handler.setStereoLeft(left_camera, R1)
        calibration_handler.setStereoRight(right_camera, R2)

        if rgbEnabled:
            rgbSpecExtrinsics = calibration_handler.getCameraExtrinsics(right_camera, rgb_camera, True)
            specTranslation = (rgbSpecExtrinsics[0][3], rgbSpecExtrinsics[1][3], rgbSpecExtrinsics[2][3])
            rgbCompExtrinsics = calibration_handler.getCameraExtrinsics(right_camera, rgb_camera, False)
            compTranslation = (rgbCompExtrinsics[0][3], rgbCompExtrinsics[1][3], rgbCompExtrinsics[2][3])
            calibration_handler.setCameraExtrinsics(right_camera, rgb_camera, rgbR, compTranslation, specTranslation)

        #flash updates

        is_write_successful = False
        if not dryRun:
            calibFile = str((Path(__file__).parent / Path(f"calib_{device.getMxId()}_backup.json")).resolve().absolute())
            original_calibration.eepromToJsonFile(calibFile)
            print(f"Original calibration data on the device is backed up at: {calibFile}")

            is_write_successful = device.flashCalibration(calibration_handler)
            if not is_write_successful:
                print(f"Error: failed to save calibration to EEPROM")
        else:
            calibFile = str((Path(__file__).parent / Path(f"calib_{device.getMxId()}_dynamic_calib.json")).resolve().absolute())
            calibration_handler.eepromToJsonFile(calibFile)
            print(f"Dynamic calibration data on the device is saved at: {calibFile}")


        if options.rectifiedDisp:
            image_data_pairs = []
            image_data_pairs.append((img_l, img_r))
            if rgbEnabled:
                image_data_pairs.append((img_rgb, img_rgb2))

            display_rectification(image_data_pairs)

