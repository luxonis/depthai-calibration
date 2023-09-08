import math
import numpy as np
import cv2
from pathlib import Path
import depthai as dai
from feature_helper import FeaturesHelper, keypoint_to_point2f
from scipy.spatial.transform import Rotation


feature_helper = FeaturesHelper(0.5, 1)


def getDevice(calib):

    device = dai.Device()
    if not calib:
        calibHandler = device.readCalibration()

    pipeline = dai.Pipeline()

    cams = device.getConnectedCameras()
    sensorNames = device.getCameraSensorNames()

    if not dai.CameraBoardSocket.CAM_B in cams and dai.CameraBoardSocket.CAM_C in cams:
        raise RuntimeError("Left and right cameras are not available for epipolar check")

    for cam in cams:
        if cam == dai.CameraBoardSocket.CAM_B:
            name = sensorNames[dai.CameraBoardSocket.CAM_B]
            camLeft = None
            print('Name of left camera: ', name)
            if name == 'OV9282':
                camLeft = pipeline.create(dai.node.MonoCamera)
                camLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
                camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
            xoutLeft = pipeline.create(dai.node.XLinkOut)
            xoutLeft.setStreamName("left")
            camLeft.out.link(xoutLeft.input)
        elif cam == dai.CameraBoardSocket.CAM_C:
            name = sensorNames[dai.CameraBoardSocket.CAM_C]
            camRight = None
            if name == 'OV9282':
                camRight = pipeline.create(dai.node.MonoCamera)
                camRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
                camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
            xoutRight = pipeline.create(dai.node.XLinkOut)
            xoutRight.setStreamName("right")
            camRight.out.link(xoutRight.input)

    device.startPipeline(pipeline)
    return device, calibHandler

def evaluateDevice(device, calibHandler):
    left_queue = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    right_queue = device.getOutputQueue(name="right", maxSize=4, blocking=False)

    left_k, w, h = calibHandler.getDefaultIntrinsics(dai.CameraBoardSocket.CAM_B)
    right_k, _, _ = calibHandler.getDefaultIntrinsics(dai.CameraBoardSocket.CAM_C)
    left_k = np.array(left_k)
    right_k = np.array(right_k)
    left_d = np.array(calibHandler.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B))
    right_d = np.array(calibHandler.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C))

    left_r = np.array(calibHandler.getStereoLeftRectificationRotation())
    right_r = np.array(calibHandler.getStereoRightRectificationRotation())
    t = np.array(calibHandler.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C, False))
    r = t[:3, :3]
    trans = t[:3, 3]
    print(f'transformation matrix is {t}')
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(left_k, left_d, right_k, right_d, (w, h), r, trans)

    left_mapx, left_mapy = cv2.initUndistortRectifyMap(left_k, left_d, R1, right_k, (w, h), cv2.CV_32FC1)
    right_mapx, right_mapy = cv2.initUndistortRectifyMap(right_k, right_d, R2, right_k, (w, h), cv2.CV_32FC1)
    width = w
    height = h
    print(f'Width = {width},\n Height = {height}')
    print(f'left K = {left_k},\n right K = {right_k}' )
    print(f'left D = {left_d},\n right D = {right_d}' )
    print(f'Left R = {left_r},\n Right R = {right_r}' )
    print(f'Left R2 = {R1},\n Right P = {R2}' )
    # exit(1)
    left_image = None
    right_image = None

    hor_epipolar_list = []
    while not device.isClosed():
        left_image = left_queue.get().getCvFrame()
        right_image = right_queue.get().getCvFrame()
        print(f'left_image shape: {left_image.shape}, right_image shape: {right_image.shape}')
        cv2.imshow("left_image", left_image)
        cv2.imshow("right_image", right_image)

        left_hor_undistorted = cv2.remap(left_image, left_mapx, left_mapy, cv2.INTER_CUBIC)
        right_hor_undistorted = cv2.remap(right_image, right_mapx, right_mapy, cv2.INTER_CUBIC)

        # cv2.imshow("left_hor_undistorted", left_hor_undistorted)
        # cv2.imshow("right_hor_undistorted", right_hor_undistorted)        
        
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

        epipolar_error, stacked_image = feature_helper.calculate_epipolar_error(left_hor_undistorted, right_hor_undistorted)
        hor_epipolar_list.append(epipolar_error)

        print(f' average epipolar error per frame: {epipolar_error}')
        cv2.imshow("stacked image", stacked_image)
    print(f'Average hor Epiploar error across {len(hor_epipolar_list)} frames is : { sum(hor_epipolar_list) / len(hor_epipolar_list)}')



if __name__ == '__main__':
    calib = None
    device, calib = getDevice(calib)
    evaluateDevice(device, calib)


