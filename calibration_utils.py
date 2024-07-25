#!/usr/bin/env python3

import cv2
import glob
import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
import time
import json
import cv2.aruco as aruco
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from pathlib import Path
from functools import reduce
from collections import deque
from typing import Optional
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
plt.rcParams.update({'font.size': 16})
import matplotlib.colors as colors
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
per_ccm = True
extrinsic_per_ccm = False
cdict = {'red':  ((0.0, 0.0, 0.0),   # no red at 0
          (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
          (1.0, 0.8, 0.8)),  # set to 0.8 so its not too bright at 1
'green': ((0.0, 0.8, 0.8),   # set to 0.8 so its not too bright at 0
          (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
          (1.0, 0.0, 0.0)),  # no green at 1
'blue':  ((0.0, 0.0, 0.0),   # no blue at 0
          (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
          (1.0, 0.0, 0.0))   # no blue at 1
}
# Create the colormap using the dictionary
GnRd = colors.LinearSegmentedColormap('GnRd', cdict)
def get_quadrant_coordinates(width, height, nx, ny):
    quadrant_width = width // nx
    quadrant_height = height // ny
    quadrant_coords = []
    
    for i in range(int(nx)):
        for j in range(int(ny)):
            left = i * quadrant_width
            upper = j * quadrant_height
            right = left + quadrant_width
            bottom = upper + quadrant_height
            quadrant_coords.append((left, upper, right, bottom))
    
    return quadrant_coords

def sort_points_into_quadrants(points, width, height, error, nx = 4, ny = 4):
    quadrant_coords = get_quadrant_coordinates(width, height, nx, ny)
    quadrants = {i: [] for i in range(int(nx*ny))}  # Create a dictionary to store points by quadrant index

    for x, y in points:
        # Find the correct quadrant for each point
        for index, (left, upper, right, bottom) in enumerate(quadrant_coords):
            if left <= x < right and upper <= y < bottom:
                quadrants[index].append(error[index])
                break
            
    return quadrants, quadrant_coords

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
# Creates a set of 13 polygon coordinates
rectProjectionMode = 0

colors = [(0, 255 , 0), (0, 0, 255)]
def setPolygonCoordinates(height, width):
    horizontal_shift = width//4
    vertical_shift = height//4

    margin = 60
    slope = 150

    p_coordinates = [
        [[margin, margin], [margin, height-margin],
            [width-margin, height-margin], [width-margin, margin]],

        [[margin, 0], [margin, height], [width//2, height-slope], [width//2, slope]],
        [[horizontal_shift, 0], [horizontal_shift, height], [
            width//2 + horizontal_shift, height-slope], [width//2 + horizontal_shift, slope]],
        [[horizontal_shift*2-margin, 0], [horizontal_shift*2-margin, height], [width//2 +
                                                                               horizontal_shift*2-margin, height-slope], [width//2 + horizontal_shift*2-margin, slope]],

        [[width-margin, 0], [width-margin, height],
            [width//2, height-slope], [width//2, slope]],
        [[width-horizontal_shift, 0], [width-horizontal_shift, height], [width //
                                                                         2-horizontal_shift, height-slope], [width//2-horizontal_shift, slope]],
        [[width-horizontal_shift*2+margin, 0], [width-horizontal_shift*2+margin, height], [width //
                                                                                           2-horizontal_shift*2+margin, height-slope], [width//2-horizontal_shift*2+margin, slope]],

        [[0, margin], [width, margin], [
            width-slope, height//2], [slope, height//2]],
        [[0, vertical_shift], [width, vertical_shift], [width-slope,
                                                        height//2+vertical_shift], [slope, height//2+vertical_shift]],
        [[0, vertical_shift*2-margin], [width, vertical_shift*2-margin], [width-slope,
                                                                          height//2+vertical_shift*2-margin], [slope, height//2+vertical_shift*2-margin]],

        [[0, height-margin], [width, height-margin],
         [width-slope, height//2], [slope, height//2]],
        [[0, height-vertical_shift], [width, height-vertical_shift], [width -
                                                                      slope, height//2-vertical_shift], [slope, height//2-vertical_shift]],
        [[0, height-vertical_shift*2+margin], [width, height-vertical_shift*2+margin], [width -
                                                                                        slope, height//2-vertical_shift*2+margin], [slope, height//2-vertical_shift*2+margin]]
    ]
    return p_coordinates


def getPolygonCoordinates(idx, p_coordinates):
    return p_coordinates[idx]


def getNumOfPolygons(p_coordinates):
    return len(p_coordinates)

# Filters polygons to just those at the given indexes.


def select_polygon_coords(p_coordinates, indexes):
    if indexes == None:
        # The default
        return p_coordinates
    else:
        print("Filtering polygons to those at indexes=", indexes)
        return [p_coordinates[i] for i in indexes]


def image_filename(polygon_index, total_num_of_captured_images):
    return "p{polygon_index}_{total_num_of_captured_images}.png".format(polygon_index=polygon_index, total_num_of_captured_images=total_num_of_captured_images)


def polygon_from_image_name(image_name):
    """Returns the polygon index from an image name (ex: "left_p10_0.png" => 10)"""
    return int(re.findall("p(\d+)", image_name)[0])

class StereoExceptions(Exception):
    def __init__(self, message, stage, path=None, *args, **kwargs) -> None:
        self.stage = stage
        self.path = path
        super().__init__(message, *args, **kwargs)

    @property
    def summary(self) -> str:
        """
        Returns a more comprehensive summary of the exception
        """
        return f"'{self.args[0]}' (occured during stage '{self.stage}')"


class StereoCalibration(object):
    """Class to Calculate Calibration and Rectify a Stereo Camera."""

    def __init__(self, traceLevel: float = 1.0, outputScaleFactor: float = 0.5, disableCamera: list = [], model = None,distortion_model = {}, filtering_enable = False, initial_max_threshold = 15, initial_min_filtered = 0.05, calibration_max_threshold = 10):
        self.filtering_enable = filtering_enable
        self.ccm_model = distortion_model
        self.model = model
        self.output_scale_factor = outputScaleFactor
        self.disableCamera = disableCamera
        self.errors = {}
        self.initial_max_threshold = initial_max_threshold
        self.initial_min_filtered = initial_min_filtered
        self.calibration_max_threshold = calibration_max_threshold
        self.calibration_min_filtered = initial_min_filtered

        """Class to Calculate Calibration and Rectify a Stereo Camera."""

    def calibrate(self, board_config, filepath, square_size, mrk_size, squaresX, squaresY, camera_model, enable_disp_rectify, charucos = {}, intrinsic_img = {}, extrinsic_img = []):
        """Function to calculate calibration for stereo camera."""
        start_time = time.time()
        # init object data
        if intrinsic_img != {}:
            for cam in intrinsic_img:
                intrinsic_img[cam].sort(reverse=True)
        if extrinsic_img != {}:
            for cam in extrinsic_img:
                extrinsic_img[cam].sort(reverse=True)
        self.intrinsic_img = intrinsic_img
        self.extrinsic_img = extrinsic_img
        self.cameraIntrinsics = {}
        self.cameraDistortion = {}
        self.distortion_model = {}
        self.errors = {}
        self._enable_rectification_disp = True
        self._cameraModel = camera_model
        self._data_path = filepath
        self._aruco_dictionary = aruco.Dictionary_get(aruco.DICT_4X4_1000)
        self._board = aruco.CharucoBoard_create(
            # 22, 16,
            squaresX, squaresY,
            square_size,
            mrk_size,
            self._aruco_dictionary)

        self.cams = []
        # parameters = aruco.DetectorParameters_create()
        combinedCoverageImage = None
        resizeWidth, resizeHeight = 1280, 800
        assert mrk_size != None,  "ERROR: marker size not set"
        calibModels = {} # Still needs to be passed to stereo calibration
        for camera in board_config['cameras'].keys():
            cam_info = board_config['cameras'][camera]
            self.id = cam_info["name"]
            if cam_info["name"] in self.disableCamera:
                continue

            images_path = filepath + '/' + cam_info['name']
            image_files = glob.glob(images_path + "/*")
            image_files.sort()
            for im in image_files:
                frame = cv2.imread(im)
                height, width, _ = frame.shape
                widthRatio = resizeWidth / width
                heightRatio = resizeHeight / height
                if (widthRatio > 0.8 and heightRatio > 0.8 and widthRatio <= 1.0 and heightRatio <= 1.0) or (widthRatio > 1.2 and heightRatio > 1.2) or (resizeHeight == 0):
                    resizeWidth = width
                    resizeHeight = height
                break

            print(
                '<------------Calibrating {} ------------>'.format(cam_info['name']))
            images_path = filepath + '/' + cam_info['name']
            if "calib_model" in cam_info.keys():
                self.cameraModel_ccm, self.model_ccm = cam_info["calib_model"].split("_")
                if self.cameraModel_ccm == "fisheye":
                    self.model_ccm == None
                calib_model = self.cameraModel_ccm
                self.distortion_model[cam_info["name"]] = self.model_ccm
            else:
                calib_model = self._cameraModel
                if cam_info["name"] in self.ccm_model:
                    self.distortion_model[cam_info["name"]] = self.ccm_model[cam_info["name"]]
                else:
                    self.distortion_model[cam_info["name"]] = self.model
            calibModels[cam_info['name']] = calib_model


            features = None
            self.img_path = glob.glob(images_path + "/*")
            if charucos == {}:
                self.img_path = sorted(self.img_path, key=lambda x: int(x.split('_')[1]))
            else:
                self.img_path.sort()
            cam_info["img_path"] = self.img_path
            self.name = cam_info["name"]
            if per_ccm:
                all_features, all_ids, imsize = self.getting_features(images_path, cam_info["name"], width, height, features=features, charucos=charucos)
                if isinstance(all_features, str) and all_ids is None:
                    if cam_info["name"] not in self.errors.keys():
                        self.errors[cam_info["name"]] = []
                    self.errors[cam_info["name"]].append(all_features)
                    continue
                cam_info["imsize"] = imsize

                f = imsize[0] / (2 * np.tan(np.deg2rad(cam_info["hfov"]/2)))
                print("INTRINSIC CALIBRATION")
                cameraMatrixInit = np.array([[f,    0.0,      imsize[0]/2],
                                             [0.0,     f,      imsize[1]/2],
                                             [0.0,   0.0,        1.0]])
                if cam_info["name"] not in self.cameraIntrinsics.keys():
                    self.cameraIntrinsics[cam_info["name"]] = cameraMatrixInit

                distCoeffsInit = np.zeros((12, 1))
                if cam_info["name"] not in self.cameraDistortion:
                    self.cameraDistortion[cam_info["name"]] = distCoeffsInit

                if cam_info["name"] in self.intrinsic_img:
                    all_features, all_ids, filtered_images = self.remove_features(filtered_features, filtered_ids, self.intrinsic_img[cam_info["name"]], image_files)
                else:
                    filtered_images = images_path
                current_time = time.time()
                if self._cameraModel != "fisheye":
                    print("Filtering corners")
                    removed_features, filtered_features, filtered_ids = self.filtering_features(all_features, all_ids, cam_info["name"],imsize,cam_info["hfov"], cameraMatrixInit, distCoeffsInit)

                    if filtered_features is None:
                        if cam_info["name"] not in self.errors.keys():
                            self.errors[cam_info["name"]] = []
                        self.errors[cam_info["name"]].append(removed_features)
                        continue

                    print(f"Filtering takes: {time.time()-current_time}")
                else:
                    filtered_features = all_features
                    filtered_ids = all_ids

                cam_info['filtered_ids'] = filtered_ids
                cam_info['filtered_corners'] = filtered_features

                ret, intrinsics, dist_coeff, _, _, filtered_ids, filtered_corners, size, coverageImage, all_corners, all_ids = self.calibrate_wf_intrinsics(cam_info["name"], all_features, all_ids, filtered_features, filtered_ids, cam_info["imsize"], cam_info["hfov"], features, filtered_images, charucos, calib_model)
                if isinstance(ret, str) and all_ids is None:
                    if cam_info["name"] not in self.errors.keys():
                        self.errors[cam_info["name"]] = []
                    self.errors[cam_info["name"]].append(ret)
                    continue
            else:
                ret, intrinsics, dist_coeff, _, _, filtered_ids, filtered_corners, size, coverageImage, all_corners, all_ids = self.calibrate_intrinsics(
                    images_path, cam_info['hfov'], cam_info["name"], charucos, width, height, calib_model)
                cam_info['filtered_ids'] = filtered_ids
                cam_info['filtered_corners'] = filtered_corners
            self.cameraIntrinsics[cam_info["name"]] = intrinsics
            self.cameraDistortion[cam_info["name"]] = dist_coeff
            cam_info['intrinsics'] = intrinsics
            cam_info['dist_coeff'] = dist_coeff
            cam_info['size'] = size # (Width, height)
            cam_info['reprojection_error'] = ret
            print("Reprojection error of {0}: {1}".format(
                cam_info['name'], ret))
            
            coverage_name = cam_info['name']
            print_text = f'Coverage Image of {coverage_name} with reprojection error of {round(ret,5)}'
            height, width, _ = coverageImage.shape

            if width > resizeWidth and height > resizeHeight:
                coverageImage = cv2.resize(
                coverageImage, (0, 0), fx= resizeWidth / width, fy= resizeWidth / width)

            height, width, _ = coverageImage.shape
            if height > resizeHeight:
                height_offset = (height - resizeHeight)//2
                coverageImage = coverageImage[height_offset:height_offset+resizeHeight, :]
            
            height, width, _ = coverageImage.shape
            height_offset = (resizeHeight - height)//2
            width_offset = (resizeWidth - width)//2
            subImage = np.pad(coverageImage, ((height_offset, height_offset), (width_offset, width_offset), (0, 0)), 'constant', constant_values=0)
            cv2.putText(subImage, print_text, (50, 50+height_offset), cv2.FONT_HERSHEY_SIMPLEX, 2*coverageImage.shape[0]/1750, (0, 0, 0), 2)
            if combinedCoverageImage is None:
                combinedCoverageImage = subImage
            else:
                combinedCoverageImage = np.hstack((combinedCoverageImage, subImage))
            coverage_file_path = filepath + '/' + coverage_name + '_coverage.png'
            
            cv2.imwrite(coverage_file_path, subImage)
        if self.errors != {}:
            string = ""
            for key in self.errors.keys():
                string += self.errors[key][0] + "\n"
            raise StereoExceptions(message=string, stage="intrinsic")

        combinedCoverageImage = cv2.resize(combinedCoverageImage, (0, 0), fx=self.output_scale_factor, fy=self.output_scale_factor)
        if enable_disp_rectify:
            # cv2.imshow('coverage image', combinedCoverageImage)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
        
        for camera in board_config['cameras'].keys():
            left_cam_info = board_config['cameras'][camera]
            if str(left_cam_info["name"]) not in self.disableCamera:
                if 'extrinsics' in left_cam_info:
                    if 'to_cam' in left_cam_info['extrinsics']:
                        left_cam = camera
                        right_cam = left_cam_info['extrinsics']['to_cam']
                        left_path = filepath + '/' + left_cam_info['name']
    
                        right_cam_info = board_config['cameras'][left_cam_info['extrinsics']['to_cam']]
                        if str(right_cam_info["name"]) not in self.disableCamera:
                            right_path = filepath + '/' + right_cam_info['name']
                            print('<-------------Extrinsics calibration of {} and {} ------------>'.format(
                                left_cam_info['name'], right_cam_info['name']))
    
                            specTranslation = left_cam_info['extrinsics']['specTranslation']
                            rot = left_cam_info['extrinsics']['rotation']
    
                            translation = np.array(
                                [specTranslation['x'], specTranslation['y'], specTranslation['z']], dtype=np.float32)
                            rotation = Rotation.from_euler(
                                'xyz', [rot['r'], rot['p'], rot['y']], degrees=True).as_matrix().astype(np.float32)
                            if per_ccm and extrinsic_per_ccm:
                                if left_cam_info["name"] in self.extrinsic_img or right_cam_info["name"] in self.extrinsic_img:
                                    if left_cam_info["name"] in self.extrinsic_img:
                                        array = self.extrinsic_img[left_cam_info["name"]]
                                    elif right_cam_info["name"] in self.extrinsic_img:
                                        array = self.extrinsic_img[left_cam_info["name"]]
                                    left_cam_info['filtered_corners'], left_cam_info['filtered_ids'], filtered_images = self.remove_features(left_cam_info['filtered_corners'], left_cam_info['filtered_ids'], array)
                                    right_cam_info['filtered_corners'], right_cam_info['filtered_ids'], filtered_images = self.remove_features(right_cam_info['filtered_corners'], right_cam_info['filtered_ids'], array)
                                    removed_features, left_cam_info['filtered_corners'], left_cam_info['filtered_ids'] = self.filtering_features(left_cam_info['filtered_corners'], left_cam_info['filtered_ids'], left_cam_info["name"],left_cam_info["imsize"],left_cam_info["hfov"], self.cameraIntrinsics["name"], self.cameraDistortion["name"])
                                    removed_features, right_cam_info['filtered_corners'], right_cam_info['filtered_ids'] = self.filtering_features(right_cam_info['filtered_corners'], right_cam_info['filtered_ids'], right_cam_info["name"], right_cam_info["imsize"], right_cam_info["hfov"], self.cameraIntrinsics["name"], self.cameraDistortion["name"])
                            
                            extrinsics = self.calibrate_stereo(left_cam_info['name'], right_cam_info['name'], left_cam_info['filtered_ids'], left_cam_info['filtered_corners'], right_cam_info['filtered_ids'], right_cam_info['filtered_corners'], left_cam_info['intrinsics'], left_cam_info[
                                                                   'dist_coeff'], right_cam_info['intrinsics'], right_cam_info['dist_coeff'], translation, rotation, features, calibModels[left_cam_info['name']], calibModels[right_cam_info['name']])
                            if extrinsics[0] == -1:
                                return -1, extrinsics[1]
    
                            if board_config['stereo_config']['left_cam'] == left_cam and board_config['stereo_config']['right_cam'] == right_cam:
                                board_config['stereo_config']['rectification_left'] = extrinsics[3]
                                board_config['stereo_config']['rectification_right'] = extrinsics[4]

                            elif board_config['stereo_config']['left_cam'] == right_cam and board_config['stereo_config']['right_cam'] == left_cam:
                                board_config['stereo_config']['rectification_left'] = extrinsics[4]
                                board_config['stereo_config']['rectification_right'] = extrinsics[3]
    
                            """ for stereoObj in board_config['stereo_config']:
    
                                if stereoObj['left_cam'] == left_cam and stereoObj['right_cam'] == right_cam and stereoObj['main'] == 1:
                                    stereoObj['rectification_left'] = extrinsics[3]
                                    stereoObj['rectification_right'] = extrinsics[4] """
    
                            print('<-------------Epipolar error of {} and {} ------------>'.format(
                                left_cam_info['name'], right_cam_info['name']))
                            #print(f"dist {left_cam_info['name']}: {left_cam_info['dist_coeff']}")
                            #print(f"dist {right_cam_info['name']}: {right_cam_info['dist_coeff']}")
                            if left_cam_info['intrinsics'][0][0] < right_cam_info['intrinsics'][0][0]:
                                scale = right_cam_info['intrinsics'][0][0]
                            else:
                                scale = left_cam_info['intrinsics'][0][0]
                            if per_ccm and extrinsic_per_ccm:
                                scale = ((left_cam_info['intrinsics'][0][0]*right_cam_info['intrinsics'][0][0] + left_cam_info['intrinsics'][1][1]*right_cam_info['intrinsics'][1][1])/2)
                                print(f"Epipolar error {extrinsics[0]*np.sqrt(scale)}")
                                left_cam_info['extrinsics']['epipolar_error'] = extrinsics[0]*np.sqrt(scale)
                                left_cam_info['extrinsics']['stereo_error'] = extrinsics[0]*np.sqrt(scale)
                            else:
                                print(f"Epipolar error {extrinsics[0]}")
                                left_cam_info['extrinsics']['epipolar_error'] = extrinsics[0]
                                left_cam_info['extrinsics']['stereo_error'] = extrinsics[0]
                            """self.test_epipolar_charuco(left_cam_info['name'], 
                                                        right_cam_info['name'],
                                                        left_path, 
                                                        right_path, 
                                                        left_cam_info['intrinsics'], 
                                                        left_cam_info['dist_coeff'], 
                                                        right_cam_info['intrinsics'], 
                                                        right_cam_info['dist_coeff'], 
                                                        extrinsics[2], # Translation between left and right Cameras
                                                        extrinsics[3], # Left Rectification rotation 
                                                        extrinsics[4], # Right Rectification rotation
                                                        calibModels[left_cam_info['name']], calibModels[right_cam_info['name']]
                                                        )"""
                                                                                            
    
                            left_cam_info['extrinsics']['rotation_matrix'] = extrinsics[1]
                            left_cam_info['extrinsics']['translation'] = extrinsics[2]
    
        return 1, board_config

    def getting_features(self, img_path, name, width, height, features = None, charucos=None):
        if charucos != {}:
            allCorners = []
            allIds = []
            for index, charuco_img in enumerate(charucos[name]):
                ids, charucos = charuco_img
                allCorners.append(charucos)
                allIds.append(ids)
            imsize = (width, height)
            return allCorners, allIds, imsize

        elif features == None or features == "charucos":
            allCorners, allIds, _, _, imsize, _ = self.analyze_charuco(self.img_path)
            return allCorners, allIds, imsize

        if features == "checker_board":
            allCorners, allIds, _, _, imsize, _ = self.analyze_charuco(self.img_path)
            return allCorners, allIds, imsize
        ###### ADD HERE WHAT IT IS NEEDED ######

    def filtering_features(self,allCorners, allIds, name,imsize, hfov, cameraMatrixInit, distCoeffsInit):

         # check if there are any suspicious corners with high reprojection error
        rvecs = []
        tvecs = []
        index = 0
        self.index = 0
        max_threshold = 75 + self.initial_max_threshold * (hfov / 30 + imsize[1] / 800 * 0.2)
        threshold_stepper = int(1.5 * (hfov / 30 + imsize[1] / 800))
        if threshold_stepper < 1:
            threshold_stepper = 1
        print(threshold_stepper)
        min_inlier = 1 - self.initial_min_filtered * (hfov / 60 + imsize[1] / 800 * 0.2)
        overall_pose = time.time()
        for index, corners in enumerate(allCorners):
            if len(corners) < 4:
                return f"Less than 4 corners detected on {index} image.", None, None
        for corners, ids in zip(allCorners, allIds):
            current = time.time()
            self.index = index
            objpts = self.charuco_ids_to_objpoints(ids)
            rvec, tvec, newids = self.camera_pose_charuco(objpts, corners, ids, cameraMatrixInit, distCoeffsInit, max_threshold = max_threshold, min_inliers=min_inlier, ini_threshold = 5, threshold_stepper=threshold_stepper)
            #allCorners[index] = np.array([corners[id[0]-1] for id in newids])
            #allIds[index] = np.array([ids[id[0]-1] for id in newids])
            tvecs.append(tvec)
            rvecs.append(rvec)
            print(f"Pose estimation {index}, {time.time() -current}s")
            index += 1
        print(f"Overall pose estimation {time.time() - overall_pose}s")

        # Here we need to get initialK and parameters for each camera ready and fill them inside reconstructed reprojection error per point
        ret = 0.0
        distortion_flags = self.get_distortion_flags(name)
        flags = cv2.CALIB_USE_INTRINSIC_GUESS + distortion_flags
        current = time.time()
        filtered_corners, filtered_ids,all_error, removed_corners, removed_ids, removed_error = self.features_filtering_function(rvecs, tvecs, cameraMatrixInit, distCoeffsInit, ret, allCorners, allIds, camera = name)
        corner_detector = filtered_corners.copy()
        for index, corners in enumerate(filtered_corners):
            if len(corners) < 4:
                corner_detector.remove(corners)
        if len(corner_detector) < int(len(self.img_path)*0.75):
            return f"More than 1/4 of images has less than 4 corners for {name}", None, None

                
        print(f"Filtering {time.time() -current}s")
        try:
            (ret, camera_matrix, distortion_coefficients,
                     rotation_vectors, translation_vectors,
                     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
                     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                        charucoCorners=filtered_corners,
                        charucoIds=filtered_ids,
                        board=self._board,
                        imageSize=imsize,
                        cameraMatrix=cameraMatrixInit,
                        distCoeffs=distCoeffsInit,
                        flags=flags,
                        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 1000, 1e-6))
        except:
            return f"First intrisic calibration failed for {name}", None, None

        self.cameraIntrinsics[name] = camera_matrix
        self.cameraDistortion[name] = distortion_coefficients
        return removed_corners, filtered_corners, filtered_ids
    
    def remove_features(self, allCorners, allIds, array, img_files = None):
        filteredCorners = allCorners.copy()
        filteredIds = allIds.copy()
        if img_files is not None:
            img_path = img_files.copy()

        for index in array:
            filteredCorners.pop(index)
            filteredIds.pop(index)
            if img_files is not None:
                img_path.pop(index)

        return filteredCorners, filteredIds, img_path

    def get_distortion_flags(self,name):
        def is_binary_string(s: str) -> bool:
        # Check if all characters in the string are '0' or '1'
            return all(char in '01' for char in s)
        if self.distortion_model[name] == None:
            print("Use DEFAULT model")
            flags = cv2.CALIB_RATIONAL_MODEL
        elif is_binary_string(self.distortion_model[name]):
            flags = cv2.CALIB_RATIONAL_MODEL
            flags += cv2.CALIB_TILTED_MODEL
            flags += cv2.CALIB_THIN_PRISM_MODEL
            binary_number = int(self.distortion_model[name], 2)
            # Print the results
            if binary_number == 0:
                clauses_status = [True, True,True, True, True, True, True, True, True]
            else:
                clauses_status = [(binary_number & (1 << i)) != 0 for i in range(len(self.distortion_model[name]))]
                clauses_status = clauses_status[::-1]
            if clauses_status[0]:
                print("FIX_K1")
                flags += cv2.CALIB_FIX_K1
            if clauses_status[1]:
                print("FIX_K2")
                flags += cv2.CALIB_FIX_K2
            if clauses_status[2]:
                print("FIX_K3")
                flags += cv2.CALIB_FIX_K3
            if clauses_status[3]:
                print("FIX_K4")
                flags += cv2.CALIB_FIX_K4
            if clauses_status[4]:
                print("FIX_K5")
                flags += cv2.CALIB_FIX_K5
            if clauses_status[5]:
                print("FIX_K6")
                flags += cv2.CALIB_FIX_K6
            if clauses_status[6]:
                print("FIX_TANGENT_DISTORTION")
                flags += cv2.CALIB_ZERO_TANGENT_DIST
            if clauses_status[7]:
                print("FIX_TILTED_DISTORTION")
                flags += cv2.CALIB_FIX_TAUX_TAUY
            if clauses_status[8]:
                print("FIX_PRISM_DISTORTION")
                flags += cv2.CALIB_FIX_S1_S2_S3_S4

        elif isinstance(self.distortion_model[name], str):
            if self.distortion_model[name] == "NORMAL":
                print("Using NORMAL model")
                flags = cv2.CALIB_RATIONAL_MODEL
                flags += cv2.CALIB_TILTED_MODEL

            elif self.distortion_model[name] == "TILTED":
                print("Using TILTED model")
                flags = cv2.CALIB_RATIONAL_MODEL
                flags += cv2.CALIB_TILTED_MODEL

            elif self.distortion_model[name] == "PRISM":
                print("Using PRISM model")
                flags = cv2.CALIB_RATIONAL_MODEL
                flags += cv2.CALIB_TILTED_MODEL
                flags += cv2.CALIB_THIN_PRISM_MODEL

            elif self.distortion_model[name] == "THERMAL":
                print("Using THERMAL model")
                flags = cv2.CALIB_RATIONAL_MODEL
                flags += cv2.CALIB_FIX_K3
                flags += cv2.CALIB_FIX_K5
                flags += cv2.CALIB_FIX_K6

        elif isinstance(self.distortion_model[name], int):
            print("Using CUSTOM flags")
            flags = self.distortion_model[name]
        return flags

    def calibrate_wf_intrinsics(self, name, all_Features, all_features_Ids, allCorners, allIds, imsize, hfov, features, image_files, charucos, calib_model):
        image_files = glob.glob(image_files + "/*")
        image_files.sort()
        coverageImage = np.ones(imsize[::-1], np.uint8) * 255
        coverageImage = cv2.cvtColor(coverageImage, cv2.COLOR_GRAY2BGR)
        coverageImage = self.draw_corners(allCorners, coverageImage)
        if calib_model == 'perspective':
            if features == None or features == "charucos":
                distortion_flags = self.get_distortion_flags(name)
                ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors, filtered_ids, filtered_corners, allCorners, allIds  = self.calibrate_camera_charuco(
                    all_Features, all_features_Ids,allCorners, allIds, imsize, hfov, name, distortion_flags)
            if charucos == {}:
                self.undistort_visualization(
                    image_files, camera_matrix, distortion_coefficients, imsize, name)

                return ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors, filtered_ids, filtered_corners, imsize, coverageImage, allCorners, allIds
            else:
                return ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors, filtered_ids, filtered_corners, imsize, coverageImage, allCorners, allIds
            #### ADD ADDITIONAL FEATURES CALIBRATION ####
        else:
            if features == None or features == "charucos":
                print('Fisheye--------------------------------------------------')
                ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors, filtered_ids, filtered_corners = self.calibrate_fisheye(
                    allCorners, allIds, imsize, hfov, name)
                self.undistort_visualization(
                        image_files, camera_matrix, distortion_coefficients, imsize, name)
                print('Fisheye rotation vector', rotation_vectors[0])
                print('Fisheye translation vector', translation_vectors[0])

                # (Height, width)
                return ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors, filtered_ids, filtered_corners, imsize, coverageImage, allCorners, allIds

    def draw_corners(self, charuco_corners, displayframe):
        for corners in charuco_corners:
            color = (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))
            for corner in corners:
                corner_int = (int(corner[0][0]), int(corner[0][1]))
                cv2.circle(displayframe, corner_int, 4, color, -1)
        height, width = displayframe.shape[:2]
        start_point = (0, 0)  # top of the image
        end_point = (0, height)

        color = (0, 0, 0)  # blue in BGR
        thickness = 4

        # Draw the line on the image
        cv2.line(displayframe, start_point, end_point, color, thickness)
        return displayframe

    def features_filtering_function(self,rvecs, tvecs, cameraMatrix, distCoeffs, reprojection, filtered_corners,filtered_id, camera, display = True, threshold = None, draw_quadrants = False, nx = 4, ny = 4):
        whole_error = []
        all_points = []
        all_corners = []
        all_error = []
        all_ids = []
        removed_corners = []
        removed_points = []
        removed_ids = []
        removed_error = []
        display_corners = []
        display_points = []
        circle_size = 0
        reported_error = []
        for i, (corners, ids, frame_path) in enumerate(zip(filtered_corners, filtered_id, self.img_path)):
            frame = cv2.imread(frame_path)
            if ids is not None and corners.size > 0:
                ids = ids.flatten()  # Flatten the IDs from 2D to 1D
                objPoints = np.array([self._board.chessboardCorners[id] for id in ids], dtype=np.float32)
                imgpoints2, _ = cv2.projectPoints(objPoints, rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
                corners2 = corners.reshape(-1, 2)
                imgpoints2 = imgpoints2.reshape(-1, 2)

                errors = np.linalg.norm(corners2 - imgpoints2, axis=1)
                if threshold == None:
                    threshold = max(2*np.median(errors), 150)
                valid_mask = errors <= threshold
                removed_mask = ~valid_mask

                # Collect valid IDs in the original format (array of arrays)
                valid_ids = ids[valid_mask]
                all_ids.append(valid_ids.reshape(-1, 1).astype(np.int32))  # Reshape and store as array of arrays

                # Collect data for valid points
                reported_error.extend(errors)
                all_error.extend(errors[valid_mask])
                display_corners.extend(corners2)
                display_points.extend(imgpoints2[valid_mask])
                all_points.append(imgpoints2[valid_mask])  # Collect valid points for calibration
                all_corners.append(corners2[valid_mask].reshape(-1, 1, 2))   # Collect valid corners for calibration

                removed_corners.extend(corners2[removed_mask])
                removed_points.extend(imgpoints2[removed_mask])
                removed_ids.extend(ids[removed_mask])
                removed_error.extend(errors[removed_mask])

                total_error_squared = np.sum(errors[valid_mask]**2)
                total_points = len(objPoints[valid_mask])
                rms_error = np.sqrt(total_error_squared / total_points if total_points else 0)
                whole_error.append(rms_error)


            total_error_squared = 0
            total_points = 0

        return all_corners ,all_ids, all_error, removed_corners, removed_ids, removed_error

    def detect_charuco_board(self, image: np.array):
        arucoParams = cv2.aruco.DetectorParameters_create()
        arucoParams.minMarkerDistanceRate = 0.01
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, self._aruco_dictionary, parameters=arucoParams)  # First, detect markers
        marker_corners, marker_ids, refusd, recoverd = cv2.aruco.refineDetectedMarkers(image, self._board, corners, ids, rejectedCorners=rejectedImgPoints)
        # If found, add object points, image points (after refining them)
        if len(marker_corners) > 0:
            ret, corners, ids = cv2.aruco.interpolateCornersCharuco(marker_corners,marker_ids,image, self._board, minMarkers = 1)
            return ret, corners, ids, marker_corners, marker_ids
        else:
            return None, None, None, None, None

    def camera_pose_charuco(self, objpoints: np.array, corners: np.array, ids: np.array, K: np.array, d: np.array, ini_threshold = 2, min_inliers = 0.95, threshold_stepper = 1, max_threshold = 50):
        objects = []
        all_objects = []
        index = 0
        start_time = time.time()
        while len(objects) < len(objpoints[:,0,0]) * min_inliers:
            if ini_threshold > max_threshold:
                break
            ret, rvec, tvec, objects  = cv2.solvePnPRansac(objpoints, corners, K, d, flags = cv2.SOLVEPNP_P3P, reprojectionError = ini_threshold,  iterationsCount = 10000, confidence = 0.9)
            all_objects.append(objects)
            imgpoints2 = objpoints.copy()

            all_corners = corners.copy()
            all_corners = np.array([all_corners[id[0]] for id in objects])
            imgpoints2 = np.array([imgpoints2[id[0]] for id in objects])

            ret, rvec, tvec = cv2.solvePnP(imgpoints2, all_corners, K, d)
            imgpoints2, _ = cv2.projectPoints(imgpoints2, rvec, tvec, self.cameraIntrinsics[self.name], self.cameraDistortion[self.name])
            
            ini_threshold += threshold_stepper
            index += 1
        if ret:
            return rvec, tvec, objects
        else:
            return None
        
    def compute_reprojection_errors(self, obj_pts: np.array, img_pts: np.array, K: np.array, dist: np.array, rvec: np.array, tvec: np.array, fisheye = False):
        if fisheye:
            proj_pts, _ = cv2.fisheye.projectPoints(obj_pts, rvec, tvec, K, dist)
        else:
            proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        errs = np.linalg.norm(np.squeeze(proj_pts) - np.squeeze(img_pts), axis = 1)
        return errs 
    
    def charuco_ids_to_objpoints(self, ids):
        one_pts = self._board.chessboardCorners
        objpts = []
        for j in range(len(ids)):   
            objpts.append(one_pts[ids[j]])
        return np.array(objpts)


    def analyze_charuco(self, images, scale_req=False, req_resolution=(800, 1280)):
        """
        Charuco base pose estimation.
        """
        # print("POSE ESTIMATION STARTS:")
        allCorners = []
        allIds = []
        all_marker_corners = []
        all_marker_ids = []
        all_recovered = []
        # decimator = 0
        # SUB PIXEL CORNER DETECTION CRITERION
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.00001)
        count = 0
        skip_vis = False
        for im in images:
            img_pth = Path(im)
            frame = cv2.imread(im)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            expected_height = gray.shape[0]*(req_resolution[1]/gray.shape[1])

            if scale_req and not (gray.shape[0] == req_resolution[0] and gray.shape[1] == req_resolution[1]):
                if int(expected_height) == req_resolution[0]:
                    # resizing to have both stereo and rgb to have same
                    # resolution to capture extrinsics of the rgb-right camera
                    gray = cv2.resize(gray, req_resolution[::-1],
                                      interpolation=cv2.INTER_CUBIC)
                else:
                    # resizing and cropping to have both stereo and rgb to have same resolution
                    # to calculate extrinsics of the rgb-right camera
                    scale_width = req_resolution[1]/gray.shape[1]
                    dest_res = (
                        int(gray.shape[1] * scale_width), int(gray.shape[0] * scale_width))
                    gray = cv2.resize(
                        gray, dest_res, interpolation=cv2.INTER_CUBIC)
                    if gray.shape[0] < req_resolution[0]:
                        raise RuntimeError("resizeed height of rgb is smaller than required. {0} < {1}".format(
                            gray.shape[0], req_resolution[0]))
                    # print(gray.shape[0] - req_resolution[0])
                    del_height = (gray.shape[0] - req_resolution[0]) // 2
                    # gray = gray[: req_resolution[0], :]
                    gray = gray[del_height: del_height + req_resolution[0], :]

                count += 1
            
            ret, charuco_corners, charuco_ids, marker_corners, marker_ids  = self.detect_charuco_board(gray)


            if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:

                charuco_corners = cv2.cornerSubPix(gray, charuco_corners,
                                    winSize=(5, 5),
                                    zeroZone=(-1, -1),
                                    criteria=criteria)
                allCorners.append(charuco_corners)  # Charco chess corners
                allIds.append(charuco_ids)  # charuco chess corner id's
                all_marker_corners.append(marker_corners)
                all_marker_ids.append(marker_ids)
            else:
                print(im)
                return f'Failed to detect more than 3 markers on image {im}', None, None, None, None, None

        # imsize = gray.shape[::-1]
        return allCorners, allIds, all_marker_corners, all_marker_ids, gray.shape[::-1], all_recovered

    def calibrate_intrinsics(self, image_files, hfov, name, charucos, width, height, calib_model):
        image_files = glob.glob(image_files + "/*")
        image_files.sort()
        assert len(
            image_files) != 0, "ERROR: Images not read correctly, check directory"
        if charucos == {}:
            allCorners, allIds, _, _, imsize, _ = self.analyze_charuco(image_files)
        else:
            allCorners = []
            allIds = []
            for index, charuco_img in enumerate(charucos[name]):
                ids, charucos = charuco_img
                allCorners.append(charucos)
                allIds.append(ids)
            imsize = (height, width)

        coverageImage = np.ones(imsize[::-1], np.uint8) * 255
        coverageImage = cv2.cvtColor(coverageImage, cv2.COLOR_GRAY2BGR)
        coverageImage = self.draw_corners(allCorners, coverageImage)
        if calib_model == 'perspective':
            distortion_flags = self.get_distortion_flags(name)
            ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors, filtered_ids, filtered_corners, allCorners, allIds  = self.calibrate_camera_charuco(
                allCorners, allIds, imsize, hfov, name, distortion_flags)
            self.undistort_visualization(
                image_files, camera_matrix, distortion_coefficients, imsize, name)

            return ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors, filtered_ids, filtered_corners, imsize, coverageImage, allCorners, allIds
        else:
            print('Fisheye--------------------------------------------------')
            ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors, filtered_ids, filtered_corners = self.calibrate_fisheye(
                allCorners, allIds, imsize, hfov, name)
            self.undistort_visualization(
                    image_files, camera_matrix, distortion_coefficients, imsize, name)
            print('Fisheye rotation vector', rotation_vectors[0])
            print('Fisheye translation vector', translation_vectors[0])

            # (Height, width)
            return ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors, filtered_ids, filtered_corners, imsize, coverageImage, allCorners, allIds


    def scale_intrinsics(self, intrinsics, originalShape, destShape):
        scale = destShape[1] / originalShape[1] # scale on width
        scale_mat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        scaled_intrinsics = np.matmul(scale_mat, intrinsics)
        """ print("Scaled height offset : {}".format(
            (originalShape[0] * scale - destShape[0]) / 2))
        print("Scaled width offset : {}".format(
            (originalShape[1] * scale - destShape[1]) / 2)) """
        scaled_intrinsics[1][2] -= (originalShape[0]      # c_y - along height of the image
                                    * scale - destShape[0]) / 2
        scaled_intrinsics[0][2] -= (originalShape[1]     # c_x width of the image
                                    * scale - destShape[1]) / 2

        return scaled_intrinsics

    def undistort_visualization(self, img_list, K, D, img_size, name):
        for index, im in enumerate(img_list):
            # print(im)
            img = cv2.imread(im)
            # h, w = img.shape[:2]
            if self._cameraModel == 'perspective':
                kScaled, _ = cv2.getOptimalNewCameraMatrix(K, D, img_size, 0)
                # print(f'K scaled is \n {kScaled} and size is \n {img_size}')
                # print(f'D Value is \n {D}')
                map1, map2 = cv2.initUndistortRectifyMap(
                    K, D, np.eye(3), kScaled, img_size, cv2.CV_32FC1)
            else:
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, D, np.eye(3), K, img_size, cv2.CV_32FC1)

            undistorted_img = cv2.remap(
                img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            
            if index == 0:
                undistorted_file_path = self.data_path + '/' + name + f'_undistorted.png'
                cv2.imwrite(undistorted_file_path, undistorted_img)


    def filter_corner_outliers(self, allIds, allCorners, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors):
        corners_removed = False
        for i in range(len(allIds)):
            corners = allCorners[i]
            ids = allIds[i]
            objpts = self.charuco_ids_to_objpoints(ids)
            if self._cameraModel == "fisheye":
                errs = self.compute_reprojection_errors(objpts, corners, camera_matrix, distortion_coefficients, rotation_vectors[i], translation_vectors[i], fisheye = True)
            else:
                errs = self.compute_reprojection_errors(objpts, corners, camera_matrix, distortion_coefficients, rotation_vectors[i], translation_vectors[i])
            suspicious_err_thr = max(2*np.median(errs), 100)
            n_offending_pts = np.sum(errs > suspicious_err_thr)
            offending_pts_idxs = np.where(errs > suspicious_err_thr)[0]
            # check if there are offending points and if they form a minority
            if n_offending_pts > 0 and n_offending_pts < len(corners)/5:
                print(f"removing {n_offending_pts} offending points with errs {errs[offending_pts_idxs]}")
                corners_removed = True
                #remove the offending points
                offset = 0
                allCorners[i] = np.delete(allCorners[i],offending_pts_idxs, axis = 0)
                allIds[i] = np.delete(allIds[i],offending_pts_idxs, axis = 0)
        return corners_removed, allIds, allCorners


    def calibrate_camera_charuco(self, all_Features, all_features_Ids, allCorners, allIds, imsize, hfov, name, distortion_flags):
        """
        Calibrates the camera using the dected corners.
        """
        f = imsize[0] / (2 * np.tan(np.deg2rad(hfov/2)))

        if name not in self.cameraIntrinsics:
            cameraMatrixInit = np.array([[f,    0.0,      imsize[0]/2],
                                     [0.0,     f,      imsize[1]/2],
                                     [0.0,   0.0,        1.0]])
            threshold = 20 * imsize[1]/800.0
        else:
            cameraMatrixInit = self.cameraIntrinsics[name]
            threshold = 2 * imsize[1]/800.0
       
        if name not in self.cameraDistortion:
            distCoeffsInit = np.zeros((5, 1))
        else:
            distCoeffsInit = self.cameraDistortion[name]
         # check if there are any suspicious corners with high reprojection error
        rvecs = []
        tvecs = []
        self.index = 0
        index = 0
        max_threshold = 10 + self.initial_max_threshold * (hfov / 30 + imsize[1] / 800 * 0.2)
        min_inlier = 1 - self.initial_min_filtered * (hfov / 60 + imsize[1] / 800 * 0.2)
        for corners, ids in zip(allCorners, allIds):
            self.index = index
            objpts = self.charuco_ids_to_objpoints(ids)
            rvec, tvec, newids = self.camera_pose_charuco(objpts, corners, ids, cameraMatrixInit, distCoeffsInit)
            tvecs.append(tvec)
            rvecs.append(rvec)
            index += 1

        # Here we need to get initialK and parameters for each camera ready and fill them inside reconstructed reprojection error per point
        ret = 0.0
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        flags += distortion_flags

        #     flags = (cv2.CALIB_RATIONAL_MODEL)
        reprojection = []
        removed_errors = []
        num_corners = []
        num_threshold = []
        iterations_array = []
        intrinsic_array = {"f_x": [], "f_y": [], "c_x": [],"c_y": []}
        distortion_array = {}
        index = 0
        camera_matrix = cameraMatrixInit
        distortion_coefficients = distCoeffsInit
        rotation_vectors = rvecs
        translation_vectors = tvecs
        translation_array_x = []
        translation_array_y = []
        translation_array_z = []
        corner_checker = 0
        previous_ids = []
        import time
        try:
            whole = time.time()
            while True:
                intrinsic_array['f_x'].append(camera_matrix[0][0])
                intrinsic_array['f_y'].append(camera_matrix[1][1])
                intrinsic_array['c_x'].append(camera_matrix[0][2])
                intrinsic_array['c_y'].append(camera_matrix[1][2])
                num_threshold.append(threshold)

                translation_array_x.append(np.mean(np.array(translation_vectors).T[0][0]))
                translation_array_y.append(np.mean(np.array(translation_vectors).T[0][1]))
                translation_array_z.append(np.mean(np.array(translation_vectors).T[0][2]))


                start = time.time()
                filtered_corners, filtered_ids, all_error, removed_corners, removed_ids, removed_error = self.features_filtering_function(rotation_vectors, translation_vectors, camera_matrix, distortion_coefficients, ret, allCorners, allIds, camera = name, threshold = threshold)
                num_corners.append(len(removed_corners))
                iterations_array.append(index)
                reprojection.append(ret)
                for i in range(len(distortion_coefficients)):
                    if i not in distortion_array.keys():
                        distortion_array[i] = []
                    distortion_array[i].append(distortion_coefficients[i][0])
                print(f"Each filtering {time.time() - start}")
                start = time.time()
                try:
                    (ret, camera_matrix, distortion_coefficients,
                     rotation_vectors, translation_vectors,
                     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
                     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                        charucoCorners=filtered_corners,
                        charucoIds=filtered_ids,
                        board=self._board,
                        imageSize=imsize,
                        cameraMatrix=cameraMatrixInit,
                        distCoeffs=distCoeffsInit,
                        flags=flags,
                        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 50000, 1e-9))
                except:
                    raise StereoExceptions(message="Intrisic calibration failed", stage="intrinsic_calibration", element=name, id=self.id)
                cameraMatrixInit = camera_matrix
                distCoeffsInit = distortion_coefficients
                threshold = 5 * imsize[1]/800.0
                print(f"Each calibration {time.time()-start}")
                index += 1
                if  index > 5 or (previous_ids == removed_ids and len(previous_ids) >= len(removed_ids) and index > 2):
                    print(f"Whole procedure: {time.time() - whole}")
                    break
                previous_ids = removed_ids
        except:
            return f"Failed to calibrate camera {name}", None, None, None, None, None, None, None ,None , None
        return ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors, filtered_ids, filtered_corners, allCorners, allIds

    def calibrate_fisheye(self, allCorners, allIds, imsize, hfov, name):
        one_pts = self._board.chessboardCorners
        obj_points = []
        for i in range(len(allIds)):
            obj_points.append(self.charuco_ids_to_objpoints(allIds[i]))

        f_init = imsize[0]/np.deg2rad(hfov)*1.15

        cameraMatrixInit = np.array([[f_init, 0.          , imsize[0]/2],
                                     [0.          , f_init, imsize[1]/2],
                                     [0.          , 0.          , 1.          ]])
        distCoeffsInit = np.zeros((4,1))
         # check if there are any suspicious corners with high reprojection error
        rvecs = []
        tvecs = []
        for corners, ids in zip(allCorners, allIds):
            objpts = self.charuco_ids_to_objpoints(ids)
            corners_undist = cv2.fisheye.undistortPoints(corners, cameraMatrixInit, distCoeffsInit)
            rvec, tvec, new_ids = self.camera_pose_charuco(objpts, corners_undist,ids, np.eye(3), np.array((0.0,0,0,0)))
            tvecs.append(tvec)
            rvecs.append(rvec)
        corners_removed, filtered_ids, filtered_corners = self.filter_corner_outliers(allIds, allCorners, cameraMatrixInit, distCoeffsInit, rvecs, tvecs)
        if corners_removed:
            obj_points = []
            for i in range(len(filtered_ids)):
                obj_points.append(self.charuco_ids_to_objpoints(filtered_ids[i]))
 
        print("Camera Matrix initialization.............")
        print(cameraMatrixInit)
        flags = 0
        flags |= cv2.fisheye.CALIB_CHECK_COND 
        flags |= cv2.fisheye.CALIB_USE_INTRINSIC_GUESS 
        flags |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        flags |= cv2.fisheye.CALIB_FIX_SKEW
        

        term_criteria = (cv2.TERM_CRITERIA_COUNT +
                         cv2.TERM_CRITERIA_EPS, 30, 1e-9)
        try:
            res, K, d, rvecs, tvecs =  cv2.fisheye.calibrate(obj_points, filtered_corners, None, cameraMatrixInit, distCoeffsInit, flags=flags, criteria=term_criteria)
        except:
            # calibration failed for full FOV, let's try to limit the corners to smaller part of FOV first to find initial parameters
            success = False
            crop = 0.95
            while not success:
                print(f"trying crop factor {crop}")
                obj_points_limited = []
                corners_limited = []
                for obj_pts, corners in zip(obj_points, filtered_corners):
                    obj_points_tmp = []
                    corners_tmp = []
                    for obj_pt, corner in zip(obj_pts, corners):
                        check_x = corner[0,0] > imsize[0]*(1-crop) and corner[0,0] < imsize[0]*crop
                        check_y = corner[0,1] > imsize[1]*(1-crop) and corner[0,1] < imsize[1]*crop
                        if check_x and check_y:
                            obj_points_tmp.append(obj_pt)
                            corners_tmp.append(corner)
                    obj_points_limited.append(np.array(obj_points_tmp))
                    corners_limited.append(np.array(corners_tmp))
                try:
                    res, K, d, rvecs, tvecs = cv2.fisheye.calibrate(obj_points_limited, corners_limited, None, cameraMatrixInit, distCoeffsInit, flags=flags, criteria=term_criteria)
                    print(f"success with crop factor {crop}")
                    success = True
                    break
                except:
                    print(f"failed with crop factor {crop}")
                    if crop > 0.7:
                        crop -= 0.05
                    else:
                        raise Exception("Calibration failed: Tried maximum crop factor and still no success")
            if success:
                # trying the full FOV once more with better initial K
                print(f"new K init {K}")
                print(f"new d_init {d}")
                try:
                    res, K, d, rvecs, tvecs =  cv2.fisheye.calibrate(obj_points, filtered_corners, imsize, K, distCoeffsInit, flags=flags, criteria=term_criteria) 
                except:
                    print(f"Failed the full res calib, using calibration with crop factor {crop}")
        
        return res, K, d, rvecs, tvecs, filtered_ids, filtered_corners


    def calibrate_stereo(self, left_name, right_name, allIds_l, allCorners_l, allIds_r, allCorners_r, cameraMatrix_l, distCoeff_l, cameraMatrix_r, distCoeff_r, t_in, r_in, left_calib_model, right_calib_model, features = None):
        left_corners_sampled = []
        right_corners_sampled = []
        left_ids_sampled = []
        obj_pts = []
        one_pts = self._board.chessboardCorners

        for i in range(len(allIds_l)):
            left_sub_corners = []
            right_sub_corners = []
            obj_pts_sub = []
            #if len(allIds_l[i]) < 70 or len(allIds_r[i]) < 70:
            #      continue
            for j in range(len(allIds_l[i])):
                idx = np.where(allIds_r[i] == allIds_l[i][j])
                if idx[0].size == 0:
                    continue
                left_sub_corners.append(allCorners_l[i][j])
                right_sub_corners.append(allCorners_r[i][idx])
                obj_pts_sub.append(one_pts[allIds_l[i][j]])
            if len(left_sub_corners) > 3 and len(right_sub_corners) > 3:
                obj_pts.append(np.array(obj_pts_sub, dtype=np.float32))
                left_corners_sampled.append(
                    np.array(left_sub_corners, dtype=np.float32))
                left_ids_sampled.append(np.array(allIds_l[i], dtype=np.int32))
                right_corners_sampled.append(
                    np.array(right_sub_corners, dtype=np.float32))
            else:
                return -1, "Stereo Calib failed due to less common features"

        stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT +
                                cv2.TERM_CRITERIA_EPS, 300, 1e-9)
        if per_ccm and extrinsic_per_ccm:
            for i in range(len(left_corners_sampled)):
                if left_calib_model == "perspective":
                    left_corners_sampled[i] = cv2.undistortPoints(np.array(left_corners_sampled[i]), cameraMatrix_l, distCoeff_l, P=cameraMatrix_l)
                    #left_corners_sampled[i] = cv2.undistortPoints(np.array(left_corners_sampled[i]), cameraMatrix_l, None)

                else:
                    left_corners_sampled[i] = cv2.fisheye.undistortPoints(np.array(left_corners_sampled[i]), cameraMatrix_l, distCoeff_l, P=cameraMatrix_l)
                    #left_corners_sampled[i] = cv2.fisheye.undistortPoints(np.array(left_corners_sampled[i]), cameraMatrix_l, None)
            for i in range(len(right_corners_sampled)):
                if right_calib_model == "perspective":
                    right_corners_sampled[i] = cv2.undistortPoints(np.array(right_corners_sampled[i]), cameraMatrix_r, distCoeff_r, P=cameraMatrix_r)
                    #right_corners_sampled[i] = cv2.undistortPoints(np.array(right_corners_sampled[i]), cameraMatrix_r, None)
                else:
                    right_corners_sampled[i] = cv2.fisheye.undistortPoints(np.array(right_corners_sampled[i]), cameraMatrix_r, distCoeff_r, P=cameraMatrix_r)
                    #right_corners_sampled[i] = cv2.fisheye.undistortPoints(np.array(right_corners_sampled[i]), cameraMatrix_r, None)

            if features == None or features == "charucos": 
                flags = cv2.CALIB_FIX_INTRINSIC
                ret, M1, d1, M2, d2, R, T, E, F, _ = cv2.stereoCalibrateExtended(
                obj_pts, left_corners_sampled, right_corners_sampled,
                np.eye(3), np.zeros(12), np.eye(3), np.zeros(12), None,
                R=r_in, T=t_in, criteria=stereocalib_criteria , flags=flags)

                r_euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
                scale = ((cameraMatrix_l[0][0]*cameraMatrix_r[0][0]))
                print(f'Epipolar error without scale: {ret}')
                print(f'Epipolar error with scale: {ret*np.sqrt(scale)}')
                print('Printing Extrinsics res...')
                print(R)
                print(T)
                print(f'Euler angles in XYZ {r_euler} degs')    
                R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
                    cameraMatrix_l,
                    distCoeff_l,
                    cameraMatrix_r,
                    distCoeff_r,
                    None, R, T) # , alpha=0.1
                # self.P_l = P_l
                # self.P_r = P_r
                r_euler = Rotation.from_matrix(R_l).as_euler('xyz', degrees=True)
                r_euler = Rotation.from_matrix(R_r).as_euler('xyz', degrees=True)

                # print(f'P_l is \n {P_l}')
                # print(f'P_r is \n {P_r}') 
                return [ret, R, T, R_l, R_r, P_l, P_r]
            #### ADD OTHER CALIBRATION METHODS ###
        else:
            if self._cameraModel == 'perspective':
                flags = 0
                # flags |= cv2.CALIB_USE_EXTRINSIC_GUESS
                # print(flags)
                flags = cv2.CALIB_FIX_INTRINSIC
                distortion_flags = self.get_distortion_flags(left_name)
                flags += distortion_flags
                # print(flags)
                ret, M1, d1, M2, d2, R, T, E, F, _ = cv2.stereoCalibrateExtended(
                obj_pts, left_corners_sampled, right_corners_sampled,
                cameraMatrix_l, distCoeff_l, cameraMatrix_r, distCoeff_r, None,
                R=r_in, T=t_in, criteria=stereocalib_criteria , flags=flags)

                r_euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
                print(f'Epipolar error is {ret}')
                print('Printing Extrinsics res...')
                print(R)
                print(T)
                print(f'Euler angles in XYZ {r_euler} degs')


                R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
                    cameraMatrix_l,
                    distCoeff_l,
                    cameraMatrix_r,
                    distCoeff_r,
                    None, R, T) # , alpha=0.1
                # self.P_l = P_l
                # self.P_r = P_r
                r_euler = Rotation.from_matrix(R_l).as_euler('xyz', degrees=True)
                r_euler = Rotation.from_matrix(R_r).as_euler('xyz', degrees=True)

                return [ret, R, T, R_l, R_r, P_l, P_r]
            elif self._cameraModel == 'fisheye':
                # make sure all images have the same *number of* points
                min_num_points = min([len(pts) for pts in obj_pts])
                obj_pts_truncated = [pts[:min_num_points] for pts in obj_pts]
                left_corners_truncated = [pts[:min_num_points] for pts in left_corners_sampled]
                right_corners_truncated = [pts[:min_num_points] for pts in right_corners_sampled]

                flags = 0
                # flags |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                # flags |= cv2.fisheye.CALIB_CHECK_COND
                # flags |= cv2.fisheye.CALIB_FIX_SKEW
                flags |= cv2.fisheye.CALIB_FIX_INTRINSIC
                # flags |= cv2.fisheye.CALIB_FIX_K1
                # flags |= cv2.fisheye.CALIB_FIX_K2
                # flags |= cv2.fisheye.CALIB_FIX_K3 
                # flags |= cv2.fisheye.CALIB_FIX_K4
                # flags |= cv2.CALIB_RATIONAL_MODEL
                # TODO(sACHIN): Try without intrinsic guess
                # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
                # TODO(sACHIN): Try without intrinsic guess
                # flags |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                # flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
                (ret, M1, d1, M2, d2, R, T), E, F = cv2.fisheye.stereoCalibrate(
                    obj_pts_truncated, left_corners_truncated, right_corners_truncated,
                    cameraMatrix_l, distCoeff_l, cameraMatrix_r, distCoeff_r, None,
                    flags=flags, criteria=stereocalib_criteria), None, None
                r_euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
                print(f'Reprojection error is {ret}')
                isHorizontal = np.absolute(T[0]) > np.absolute(T[1])

                R_l, R_r, P_l, P_r, Q = cv2.fisheye.stereoRectify(
                    cameraMatrix_l,
                    distCoeff_l,
                    cameraMatrix_r,
                    distCoeff_r,
                    None, R, T, flags=0)

                r_euler = Rotation.from_matrix(R_l).as_euler('xyz', degrees=True)
                r_euler = Rotation.from_matrix(R_r).as_euler('xyz', degrees=True)

                return [ret, R, T, R_l, R_r, P_l, P_r]

    def display_rectification(self, image_data_pairs, images_corners_l, images_corners_r, image_epipolar_color, isHorizontal):
        print(
            "Displaying Stereo Pair for visual inspection. Press the [ESC] key to exit.")
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

            img_concat = cv2.resize(
                img_concat, (0, 0), fx=0.8, fy=0.8)

            # show image
            cv2.imshow('Stereo Pair', img_concat)
            k = cv2.waitKey(1)
            if k == 27:  # Esc key to stop
                break

                # os._exit(0)
                # raise SystemExit()

        cv2.destroyWindow('Stereo Pair')

    def scale_image(self, img, scaled_res):
        expected_height = img.shape[0]*(scaled_res[1]/img.shape[1])

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
    
    def sgdEpipolar(self, images_left, images_right, M_lp, d_l, M_rp, d_r, r_l, r_r, kScaledL, kScaledR, scaled_res, isHorizontal):
        if self._cameraModel == 'perspective':
            mapx_l, mapy_l = cv2.initUndistortRectifyMap(
                M_lp, d_l, r_l, kScaledL, scaled_res[::-1], cv2.CV_32FC1)
            mapx_r, mapy_r = cv2.initUndistortRectifyMap(
                M_rp, d_r, r_r, kScaledR, scaled_res[::-1], cv2.CV_32FC1)
        else:
            mapx_l, mapy_l = cv2.fisheye.initUndistortRectifyMap(
                M_lp, d_l, r_l, kScaledL, scaled_res[::-1], cv2.CV_32FC1)
            mapx_r, mapy_r = cv2.fisheye.initUndistortRectifyMap(
                M_rp, d_r, r_r, kScaledR, scaled_res[::-1], cv2.CV_32FC1)

        
        image_data_pairs = []
        imagesCount = 0

        for image_left, image_right in zip(images_left, images_right):
            # read images
            imagesCount += 1
            # print(imagesCount)
            img_l = cv2.imread(image_left, 0)
            img_r = cv2.imread(image_right, 0)

            img_l = self.scale_image(img_l, scaled_res)
            img_r = self.scale_image(img_r, scaled_res)

            # warp right image
            # img_l = cv2.warpPerspective(img_l, self.H1, img_l.shape[::-1],
            #                             cv2.INTER_CUBIC +
            #                             cv2.WARP_FILL_OUTLIERS +
            #                             cv2.WARP_INVERSE_MAP)

            # img_r = cv2.warpPerspective(img_r, self.H2, img_r.shape[::-1],
            #                             cv2.INTER_CUBIC +
            #                             cv2.WARP_FILL_OUTLIERS +
            #                             cv2.WARP_INVERSE_MAP)

            img_l = cv2.remap(img_l, mapx_l, mapy_l, cv2.INTER_LINEAR)
            img_r = cv2.remap(img_r, mapx_r, mapy_r, cv2.INTER_LINEAR)

            image_data_pairs.append((img_l, img_r))

        imgpoints_r = []
        imgpoints_l = []
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.00001)
            
        for i, image_data_pair in enumerate(image_data_pairs):
            res2_l = self.detect_charuco_board(image_data_pair[0])
            res2_r = self.detect_charuco_board(image_data_pair[1])

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

                imgpoints_l.extend(corners_l)
                imgpoints_r.extend(corners_r)
                epi_error_sum = 0
                for l_pt, r_pt in zip(corners_l, corners_r):
                    if isHorizontal:
                        epi_error_sum += abs(l_pt[0][1] - r_pt[0][1])
                    else:
                        epi_error_sum += abs(l_pt[0][0] - r_pt[0][0])
                # localError = epi_error_sum / len(corners_l)

                # print("Average Epipolar in test Error per image on host in " + img_pth_right.name + " : " +
                #       str(localError))
                raise SystemExit(1)

        epi_error_sum = 0
        for l_pt, r_pt in zip(imgpoints_l, imgpoints_r):
            if isHorizontal:
                epi_error_sum += abs(l_pt[0][1] - r_pt[0][1])
            else:
                epi_error_sum += abs(l_pt[0][0] - r_pt[0][0])

        avg_epipolar = epi_error_sum / len(imgpoints_r)
        print("Average Epipolar Error in test is : " + str(avg_epipolar))
        return avg_epipolar


    def test_epipolar_charuco(self, left_name, right_name, left_img_pth, right_img_pth, M_l, d_l, M_r, d_r, t, r_l, r_r, left_calib_model, right_calib_model):
        images_left = glob.glob(left_img_pth + '/*.png')
        images_right = glob.glob(right_img_pth + '/*.png')
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
        # print("Scale res :{}".format(scaled_res))

        M_lp = self.scale_intrinsics(M_l, frame_left_shape, scaled_res)
        M_rp = self.scale_intrinsics(M_r, frame_right_shape, scaled_res)

        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.00001)

        # TODO(Sachin): Observe Images by adding visualization 
        # TODO(Sachin): Check if the stetch is only in calibration Images
        print('Original intrinsics ....')
        print(f"L {M_lp}")
        print(f"R: {M_rp}")
        # kScaledL, _ = cv2.getOptimalNewCameraMatrix(M_r, d_r, scaled_res[::-1], 0)
        # kScaledL, _ = cv2.getOptimalNewCameraMatrix(M_r, d_l, scaled_res[::-1], 0)
        # kScaledR, _ = cv2.getOptimalNewCameraMatrix(M_r, d_r, scaled_res[::-1], 0)
        kScaledR = kScaledL = M_rp

        # if self.cameraModel != 'perspective':
        #     kScaledR = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(M_r, d_r, scaled_res[::-1], np.eye(3), fov_scale=1.1)
        #     kScaledL = kScaledR

            
        print('Intrinsics from the getOptimalNewCameraMatrix/Original ....')
        print(f"L: {kScaledL}")
        print(f"R: {kScaledR}")
        oldEpipolarError = None
        epQueue = deque()
        movePos = True


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

            img_l = self.scale_image(img_l, scaled_res)
            img_r = self.scale_image(img_r, scaled_res)
            # print(img_l.shape)
            # print(img_r.shape)

            # warp right image
            # img_l = cv2.warpPerspective(img_l, self.H1, img_l.shape[::-1],
            #                             cv2.INTER_CUBIC +
            #                             cv2.WARP_FILL_OUTLIERS +
            #                             cv2.WARP_INVERSE_MAP)

            # img_r = cv2.warpPerspective(img_r, self.H2, img_r.shape[::-1],
            #                             cv2.INTER_CUBIC +
            #                             cv2.WARP_FILL_OUTLIERS +
            #                             cv2.WARP_INVERSE_MAP)

            img_l = cv2.remap(img_l, mapx_l, mapy_l, cv2.INTER_LINEAR)
            img_r = cv2.remap(img_r, mapx_r, mapy_r, cv2.INTER_LINEAR)

            image_data_pairs.append((img_l, img_r))
            
        # compute metrics
        imgpoints_r = []
        imgpoints_l = []
        image_epipolar_color = []
        # new_imagePairs = [])
        for i, image_data_pair in enumerate(image_data_pairs):
            res2_l = self.detect_charuco_board(image_data_pair[0])
            res2_r = self.detect_charuco_board(image_data_pair[1])

            img_concat = cv2.hconcat([image_data_pair[0], image_data_pair[1]])
            img_concat = cv2.cvtColor(img_concat, cv2.COLOR_GRAY2RGB)
            line_row = 0
            while line_row < img_concat.shape[0]:
                cv2.line(img_concat,
                         (0, line_row), (img_concat.shape[1], line_row),
                         (0, 255, 0), 1)
                line_row += 30

            cv2.imshow('Stereo Pair', img_concat)
            k = cv2.waitKey(1)
            
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
            else:
                print('Numer of corners is in left -> and right ->')
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

        if self._enable_rectification_disp:
            self.display_rectification(image_data_pairs, imgpoints_l, imgpoints_r, image_epipolar_color, isHorizontal)

        return avg_epipolar

    def create_save_mesh(self):  # , output_path):

        curr_path = Path(__file__).parent.resolve()
        print("Mesh path")
        print(curr_path)

        if self._cameraModel == "perspective":
            map_x_l, map_y_l = cv2.initUndistortRectifyMap(
                self.M1, self.d1, self.R1, self.M2, self.img_shape, cv2.CV_32FC1)
            map_x_r, map_y_r = cv2.initUndistortRectifyMap(
                self.M2, self.d2, self.R2, self.M2, self.img_shape, cv2.CV_32FC1)
        else:    
            map_x_l, map_y_l = cv2.fisheye.initUndistortRectifyMap(
                self.M1, self.d1, self.R1, self.M2, self.img_shape, cv2.CV_32FC1)
            map_x_r, map_y_r = cv2.fisheye.initUndistortRectifyMap(
                self.M2, self.d2, self.R2, self.M2, self.img_shape, cv2.CV_32FC1)

        """ 
        map_x_l_fp32 = map_x_l.astype(np.float32)
        map_y_l_fp32 = map_y_l.astype(np.float32)
        map_x_r_fp32 = map_x_r.astype(np.float32)
        map_y_r_fp32 = map_y_r.astype(np.float32)
        
                
        print("shape of maps")
        print(map_x_l.shape)
        print(map_y_l.shape)
        print(map_x_r.shape)
        print(map_y_r.shape) """

        meshCellSize = 16
        mesh_left = []
        mesh_right = []

        for y in range(map_x_l.shape[0] + 1):
            if y % meshCellSize == 0:
                row_left = []
                row_right = []
                for x in range(map_x_l.shape[1] + 1):
                    if x % meshCellSize == 0:
                        if y == map_x_l.shape[0] and x == map_x_l.shape[1]:
                            row_left.append(map_y_l[y - 1, x - 1])
                            row_left.append(map_x_l[y - 1, x - 1])
                            row_right.append(map_y_r[y - 1, x - 1])
                            row_right.append(map_x_r[y - 1, x - 1])
                        elif y == map_x_l.shape[0]:
                            row_left.append(map_y_l[y - 1, x])
                            row_left.append(map_x_l[y - 1, x])
                            row_right.append(map_y_r[y - 1, x])
                            row_right.append(map_x_r[y - 1, x])
                        elif x == map_x_l.shape[1]:
                            row_left.append(map_y_l[y, x - 1])
                            row_left.append(map_x_l[y, x - 1])
                            row_right.append(map_y_r[y, x - 1])
                            row_right.append(map_x_r[y, x - 1])
                        else:
                            row_left.append(map_y_l[y, x])
                            row_left.append(map_x_l[y, x])
                            row_right.append(map_y_r[y, x])
                            row_right.append(map_x_r[y, x])
                if (map_x_l.shape[1] % meshCellSize) % 2 != 0:
                    row_left.append(0)
                    row_left.append(0)
                    row_right.append(0)
                    row_right.append(0)

                mesh_left.append(row_left)
                mesh_right.append(row_right)

        mesh_left = np.array(mesh_left)
        mesh_right = np.array(mesh_right)
        left_mesh_fpath = str(curr_path) + '/../resources/left_mesh.calib'
        right_mesh_fpath = str(curr_path) + '/../resources/right_mesh.calib'
        mesh_left.tofile(left_mesh_fpath)
        mesh_right.tofile(right_mesh_fpath)
