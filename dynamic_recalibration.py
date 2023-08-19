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

from cmath import inf
import numpy as np
import cv2
import depthai as dai
import math
import argparse
from pathlib import Path
import time
import traceback
from feature_helper import FeaturesHelper, keypoint_to_point2f

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
if cv2.__version__ >= "4.5.4":
    ransacMethod = cv2.USAC_MAGSAC

feature_helper = FeaturesHelper()

def calculate_Rt_from_frames(frame1, frame2, k1, k2, d1, d2):
    kps1, kps2, _, _ = feature_helper.getMatchedFeatures(frame1, frame2)
    minKeypoints = 20
    if len(kps1) < minKeypoints:
        raise Exception(f'Need at least {minKeypoints} keypoints!')

    if debug:
        img=cv2.drawKeypoints(frame1, kps1, frame1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Left", img)
        img2=cv2.drawKeypoints(frame2, kps2, frame2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Right", img2)
        cv2.waitKey(1)
    # print(f'Type of pts1 is {type(pts1)}')
    pts1 = keypoint_to_point2f(kps1)
    pts2 = keypoint_to_point2f(kps2)
    E, mask = cv2.findEssentialMat(pts1, pts2, k1, d1, k2, d2, method=cv2.RANSAC, prob=0.999, threshold=0.7)

    points, R_est, t_est, mask_pose = cv2.recoverPose(E, pts1, pts2, mask=mask)
    print(f' First E: {E}, R: {R_est}, t: {t_est}')

    ret, E, R_est, t_est, mask_pose = cv2.recoverPose(pts1, pts2, k1, d1, k2, d2, method=ransacMethod, prob=0.999, threshold=0.7)
    print(f' Second ret: {ret} E: {E}, R: {R_est}, t: {t_est}')

    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(k1, d1, k2, d2, frame2.shape[::-1], R_est, t_est)

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

        stereoLeftIn = xLinks["right"]["node"].out
        stereoRightIn = xLinks["left"]["node"].out
        
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

        if rgbEnabled:
            rgb_k = calibration_handler.getCameraIntrinsics(rgb_camera, rgb_img_shape[0], rgb_img_shape[1])
            rgb_k = np.array(rgb_k)
            rgb_d = calibration_handler.getDistortionCoefficients(rgb_camera)
            rgb_d = np.array(rgb_d)

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

                if debug:
                    cv2.imshow('left', left_frame)
                    cv2.imshow('right', right_frame)

                if len(left_frame.shape) != 2:
                    left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                if len(right_frame.shape) != 2:
                    right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
                left_mean = np.mean(left_frame)
                right_mean = np.mean(right_frame)

                if left_mean > 200 or right_mean > 200:
                    print(f'left mean: {left_mean}, right mean: {right_mean}')
                    cv2.waitKey(1)
                    continue

                estimate_counts += 1
                if estimate_counts < 10:
                    R, T, R1, R2, P1, P2, Q = calculate_Rt_from_frames(left_frame, right_frame, left_k, right_k, left_d, right_d)
                    estimates["lr"].append((R, T, R1, R2, P1, P2, Q))
                    if rgbEnabled:
                        rgb_frame = rgb_camera_queue.get().getCvFrame()
                        R, T, R1, R2, P1, P2, Q = calculate_Rt_from_frames(rgb_frame, right_frame, rgb_k, right_k, rgb_d, right_d)
                        # rgbR = np.linalg.inv(rgbR) #right to rgb rotation
                        estimates["lr"].append((R, T, R1, R2, P1, P2, Q))
                else:
                    img_shape = left_rect_frame.shape
                    M1 = left_k
                    M2 = right_k
                    d1 = left_d
                    d2 = right_d

                    for i in range(len(estimates["lr"])):
                        R, T, R1, R2, P1, P2, Q = estimates["lr"][i]
                        mapx_l, mapy_l = cv2.initUndistortRectifyMap(M1, d1, R1, M2, img_shape, cv2.CV_32FC1)
                        mapx_r, mapy_r = cv2.initUndistortRectifyMap(M2, d2, R2, M2, img_shape, cv2.CV_32FC1)

                        img_l = cv2.remap(left_frame, mapx_l, mapy_l, cv2.INTER_LINEAR)
                        img_r = cv2.remap(right_frame, mapx_r, mapy_r, cv2.INTER_LINEAR)

                        stereo_epipolar, disp_image = feature_helper.calculate_epipolar_error(img_l, img_r)
                        if i not in epipolar_list["lr"]:
                            epipolar_list["lr"][i] = []
                        epipolar_list["lr"][i].append(stereo_epipolar)
                        if debug:
                            cv2.imshow("lr", disp_image)
                        cv2.waitKey(1)
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
                if estimate_counts >= 20:
                    break
            except Exception as e:
                print(e)
                cv2.waitKey(1)
                tb = traceback.format_exc()
                error_line = tb.split("\n")[-3]
                print(f"Error: {e}")
                print(error_line)
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

