import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path
from .types import *
import numpy as np
import glob
import cv2

# Helper functions

def scale_intrinsics(intrinsics, originalShape, destShape):
  scale = destShape[1] / originalShape[1] # scale on width
  scale_mat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
  scaled_intrinsics = np.matmul(scale_mat, intrinsics)
  """ print("Scaled height offset : {}".format(
    (originalShape[0] * scale - destShape[0]) / 2))
  print("Scaled width offset : {}".format(
    (originalShape[1] * scale - destShape[1]) / 2)) """
  scaled_intrinsics[1][2] -= (originalShape[0]    # c_y - along height of the image
                * scale - destShape[0]) / 2
  scaled_intrinsics[0][2] -= (originalShape[1]   # c_x width of the image
                * scale - destShape[1]) / 2
  #if self.traceLevel == 3 or self.traceLevel == 10:
  print('original_intrinsics')
  print(intrinsics)
  print('scaled_intrinsics')
  print(scaled_intrinsics)

  return scaled_intrinsics

def scale_image(img, scaled_res):
  expected_height = img.shape[0]*(scaled_res[1]/img.shape[1])
  #if self.traceLevel == 2 or self.traceLevel == 10:
  print("Expected Height: {}".format(expected_height))

  if not (img.shape[0] == scaled_res[0] and img.shape[1] == scaled_res[1]):
    if int(expected_height) == scaled_res[0]:
      # resizing to have both stereo and rgb to have same
      # resolution to capture extrinsics of the rgb-right camera
      img = cv2.resize(img, (scaled_res[1], scaled_res[0]),
                interpolation=cv2.INTER_CUBIC)
      return img
    else:
      # resizing and cropping to have both stereo and rgb to have same resolution
      # to calculate extrinsics of the rgb-right camera
      scale_width = scaled_res[1]/img.shape[1]
      dest_res = (
        int(img.shape[1] * scale_width), int(img.shape[0] * scale_width))
      img = cv2.resize(
        img, dest_res, interpolation=cv2.INTER_CUBIC)
      if img.shape[0] < scaled_res[0]:
        raise RuntimeError("resizeed height of rgb is smaller than required. {0} < {1}".format(
          img.shape[0], scaled_res[0]))
      # print(gray.shape[0] - req_resolution[0])
      del_height = (img.shape[0] - scaled_res[0]) // 2
      # gray = gray[: req_resolution[0], :]
      img = img[del_height: del_height + scaled_res[0], :]
      return img
  else:
    return img

def detect_charuco_board(config: CharucoBoard, image: np.array):
  arucoParams = cv2.aruco.DetectorParameters_create()
  arucoParams.minMarkerDistanceRate = 0.01
  corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, config.dictionary, parameters=arucoParams)  # First, detect markers
  marker_corners, marker_ids, refusd, recoverd = cv2.aruco.refineDetectedMarkers(image, config.board, corners, ids, rejectedCorners=rejectedImgPoints)
  # If found, add object points, image points (after refining them)
  if len(marker_corners) > 0:
    ret, corners, ids = cv2.aruco.interpolateCornersCharuco(marker_corners,marker_ids,image, config.board, minMarkers = 1)
    return ret, corners, ids, marker_corners, marker_ids
  else:
    return None, None, None, None, None

def display_rectification(image_data_pairs, images_corners_l, images_corners_r, image_epipolar_color, isHorizontal):
  print("Displaying Stereo Pair for visual inspection. Press the [ESC] key to exit.")
  colors = [(0, 255 , 0), (0, 0, 255)]
  for idx, image_data_pair in enumerate(image_data_pairs):
    if isHorizontal:
      img_concat = cv2.hconcat(
        [image_data_pair[0], image_data_pair[1]])
      for left_pt, right_pt, colorMode in zip(images_corners_l[idx], images_corners_r[idx], image_epipolar_color[idx]):
        cv2.line(img_concat,
                 (int(left_pt[0][0]), int(left_pt[0][1])), (int(right_pt[0][0]) + image_data_pair[0].shape[1], int(right_pt[0][1])),
                 colors[colorMode], 1)
    else:
      img_concat = cv2.vconcat(
        [image_data_pair[0], image_data_pair[1]])
      for left_pt, right_pt, colorMode in zip(images_corners_l[idx], images_corners_r[idx], image_epipolar_color[idx]):
        cv2.line(img_concat,
               (int(left_pt[0][0]), int(left_pt[0][1])), (int(right_pt[0][0]), int(right_pt[0][1])  + image_data_pair[0].shape[0]),
               colors[colorMode], 1)

    img_concat = cv2.resize(img_concat, (0, 0), fx=0.8, fy=0.8)

    # show image
    cv2.imshow('Stereo Pair', img_concat)
    while True:
      k = cv2.waitKey(1)
      if k == 27:  # Esc key to stop
        break

  cv2.destroyWindow('Stereo Pair')

# Debugging functions

def debug_epipolar_charuco(left_cam_info: CameraData, right_cam_info: CameraData, left_board: CharucoBoard, right_board: CharucoBoard, t, r_l, r_r):
  images_left = glob.glob(left_cam_info['images_path'] + '/*.png')
  images_right = glob.glob(right_cam_info['images_path'] + '/*.png')
  images_left.sort()
  print(images_left)
  images_right.sort()
  assert len(images_left) != 0, "ERROR: Images not read correctly"
  assert len(images_right) != 0, "ERROR: Images not read correctly"
  isHorizontal = np.absolute(t[0]) > np.absolute(t[1])

  scale = None
  scale_req = False
  frame_left_shape = cv2.imread(images_left[0], 0).shape
  frame_right_shape = cv2.imread(images_right[0], 0).shape
  scalable_res = frame_left_shape
  scaled_res = frame_right_shape
  if frame_right_shape[0] < frame_left_shape[0] and frame_right_shape[1] < frame_left_shape[1]:
    scale_req = True
    scale = frame_right_shape[1] / frame_left_shape[1]
  elif frame_right_shape[0] > frame_left_shape[0] and frame_right_shape[1] > frame_left_shape[1]:
    scale_req = True
    scale = frame_left_shape[1] / frame_right_shape[1]
    scalable_res = frame_right_shape
    scaled_res = frame_left_shape

  if scale_req:
    scaled_height = scale * scalable_res[0]
    diff = scaled_height - scaled_res[0]
    if diff < 0:
      scaled_res = (int(scaled_height), scaled_res[1])
  #if self.traceLevel == 3 or self.traceLevel == 10:
  print(
    f'Is scale Req: {scale_req}\n scale value: {scale} \n scalable Res: {scalable_res} \n scale Res: {scaled_res}')
  print("Original res Left :{}".format(frame_left_shape))
  print("Original res Right :{}".format(frame_right_shape))
  # print("Scale res :{}".format(scaled_res))

  M_l = left_cam_info['intrinsics']
  M_r = right_cam_info['intrinsics']
  M_lp = scale_intrinsics(M_l, frame_left_shape, scaled_res)
  M_rp = scale_intrinsics(M_r, frame_right_shape, scaled_res)

  criteria = (cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.00001)

  # TODO(Sachin): Observe Images by adding visualization 
  # TODO(Sachin): Check if the stetch is only in calibration Images
  print('Original intrinsics ....')
  print(f"L {M_lp}")
  print(f"R: {M_rp}")
  #if self.traceLevel == 3 or self.traceLevel == 10:
  print(f'Width and height is {scaled_res[::-1]}')
  # kScaledL, _ = cv2.getOptimalNewCameraMatrix(M_r, d_r, scaled_res[::-1], 0)
  # kScaledL, _ = cv2.getOptimalNewCameraMatrix(M_r, d_l, scaled_res[::-1], 0)
  # kScaledR, _ = cv2.getOptimalNewCameraMatrix(M_r, d_r, scaled_res[::-1], 0)
  kScaledR = kScaledL = M_rp

  # if self.cameraModel != 'perspective':
  #   kScaledR = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(M_r, d_r, scaled_res[::-1], np.eye(3), fov_scale=1.1)
  #   kScaledL = kScaledR

    
  print('Intrinsics from the getOptimalNewCameraMatrix/Original ....')
  print(f"L: {kScaledL}")
  print(f"R: {kScaledR}")
  oldEpipolarError = None
  epQueue = deque()
  movePos = True

  left_calib_model = left_cam_info['calib_model']
  right_calib_model = right_cam_info['calib_model']
  d_l = left_cam_info['dist_coeff']
  d_r = left_cam_info['dist_coeff']
  print(left_calib_model, right_calib_model)
  if left_calib_model == 'perspective':
    mapx_l, mapy_l = cv2.initUndistortRectifyMap(
      M_lp, d_l, r_l, kScaledL, scaled_res[::-1], cv2.CV_32FC1)
  else:
    mapx_l, mapy_l = cv2.fisheye.initUndistortRectifyMap(
      M_lp, d_l, r_l, kScaledL, scaled_res[::-1], cv2.CV_32FC1)
  if right_calib_model == "perspective":
    mapx_r, mapy_r = cv2.initUndistortRectifyMap(
      M_rp, d_r, r_r, kScaledR, scaled_res[::-1], cv2.CV_32FC1)
  else:
    mapx_r, mapy_r = cv2.fisheye.initUndistortRectifyMap(
      M_rp, d_r, r_r, kScaledR, scaled_res[::-1], cv2.CV_32FC1)

  image_data_pairs = []
  for image_left, image_right in zip(images_left, images_right):
    # read images
    img_l = cv2.imread(image_left, 0)
    img_r = cv2.imread(image_right, 0)

    img_l = scale_image(img_l, scaled_res)
    img_r = scale_image(img_r, scaled_res)
    # print(img_l.shape)
    # print(img_r.shape)

    # warp right image
    # img_l = cv2.warpPerspective(img_l, self.H1, img_l.shape[::-1],
    #               cv2.INTER_CUBIC +
    #               cv2.WARP_FILL_OUTLIERS +
    #               cv2.WARP_INVERSE_MAP)

    # img_r = cv2.warpPerspective(img_r, self.H2, img_r.shape[::-1],
    #               cv2.INTER_CUBIC +
    #               cv2.WARP_FILL_OUTLIERS +
    #               cv2.WARP_INVERSE_MAP)

    img_l = cv2.remap(img_l, mapx_l, mapy_l, cv2.INTER_LINEAR)
    img_r = cv2.remap(img_r, mapx_r, mapy_r, cv2.INTER_LINEAR)

    image_data_pairs.append((img_l, img_r))
    
    #if self.traceLevel == 4 or self.traceLevel == 5 or self.traceLevel == 10:
      #cv2.imshow("undistorted-Left", img_l)
      #cv2.imshow("undistorted-right", img_r)
      # print(f'image path - {im}')
      # print(f'Image Undistorted Size {undistorted_img.shape}')
    #  k = cv2.waitKey(0)
    #  if k == 27:  # Esc key to stop
    #    break
  #if self.traceLevel == 4 or self.traceLevel == 5 or self.traceLevel == 10:
  #  cv2.destroyWindow("undistorted-Left")
  #  cv2.destroyWindow("undistorted-right")  
  # compute metrics
  imgpoints_r = []
  imgpoints_l = []
  image_epipolar_color = []
  # new_imagePairs = [])
  for i, image_data_pair in enumerate(image_data_pairs):
    res2_l = detect_charuco_board(left_board, image_data_pair[0])
    res2_r = detect_charuco_board(right_board, image_data_pair[1])
    
    # if self.traceLevel == 2 or self.traceLevel == 4 or self.traceLevel == 10:
    

    img_concat = cv2.hconcat([image_data_pair[0], image_data_pair[1]])
    img_concat = cv2.cvtColor(img_concat, cv2.COLOR_GRAY2RGB)
    line_row = 0
    while line_row < img_concat.shape[0]:
      cv2.line(img_concat,
           (0, line_row), (img_concat.shape[1], line_row),
           (0, 255, 0), 1)
      line_row += 30

    cv2.imshow('Stereo Pair', img_concat)
    while True:
      k = cv2.waitKey(1)
      if k == 27:  # Esc key to stop
        break
    
    if res2_l[1] is not None and res2_r[2] is not None and len(res2_l[1]) > 3 and len(res2_r[1]) > 3:

      cv2.cornerSubPix(image_data_pair[0], res2_l[1],
               winSize=(5, 5),
               zeroZone=(-1, -1),
               criteria=criteria)
      cv2.cornerSubPix(image_data_pair[1], res2_r[1],
               winSize=(5, 5),
               zeroZone=(-1, -1),
               criteria=criteria)

      # termination criteria
      img_pth_right = Path(images_right[i])
      img_pth_left = Path(images_left[i])
      org = (100, 50)
      # cv2.imshow('ltext', lText)
      # cv2.waitKey(0)
      localError = 0
      corners_l = []
      corners_r = []
      for j in range(len(res2_l[2])):
        idx = np.where(res2_r[2] == res2_l[2][j])
        if idx[0].size == 0:
          continue
        corners_l.append(res2_l[1][j])
        corners_r.append(res2_r[1][idx])
      #if len(corners_l) == 0 or len(corners_r) == 0:
        #continue

      imgpoints_l.append(corners_l)
      imgpoints_r.append(corners_r)
      epi_error_sum = 0
      corner_epipolar_color = []
      for l_pt, r_pt in zip(corners_l, corners_r):
        if isHorizontal:
          curr_epipolar_error = abs(l_pt[0][1] - r_pt[0][1])
        else:
          curr_epipolar_error = abs(l_pt[0][0] - r_pt[0][0])
        if curr_epipolar_error >= 1:
          corner_epipolar_color.append(1)
        else:
          corner_epipolar_color.append(0)
        epi_error_sum += curr_epipolar_error
      localError = epi_error_sum / len(corners_l)
      image_epipolar_color.append(corner_epipolar_color)
      #if self.traceLevel == 2 or self.traceLevel == 3 or self.traceLevel == 4 or self.traceLevel == 10:
      print("Epipolar Error per image on host in " + img_pth_right.name + " : " +
          str(localError))
    else:
      print('Numer of corners is in left -> and right ->')
      imgpoints_l.append([])
      imgpoints_r.append([])
      continue
      return -1
    lText = cv2.putText(cv2.cvtColor(image_data_pair[0],cv2.COLOR_GRAY2RGB), img_pth_left.name, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (2, 0, 255), 2, cv2.LINE_AA)
    rText = cv2.putText(cv2.cvtColor(image_data_pair[1],cv2.COLOR_GRAY2RGB), img_pth_right.name + " Error: " + str(localError), org, cv2.FONT_HERSHEY_SIMPLEX, 1, (2, 0, 255), 2, cv2.LINE_AA)
    image_data_pairs[i] = (lText, rText)


  epi_error_sum = 0
  total_corners = 0
  for corners_l, corners_r in zip(imgpoints_l, imgpoints_r):
    total_corners += len(corners_l)
    for l_pt, r_pt in zip(corners_l, corners_r):
      if isHorizontal:
        epi_error_sum += abs(l_pt[0][1] - r_pt[0][1])
      else:
        epi_error_sum += abs(l_pt[0][0] - r_pt[0][0])

  avg_epipolar = epi_error_sum / total_corners
  print("Average Epipolar Error is : " + str(avg_epipolar))

  if True or enable_rectification_disp:
    display_rectification(image_data_pairs, imgpoints_l, imgpoints_r, image_epipolar_color, isHorizontal)

  return avg_epipolar
