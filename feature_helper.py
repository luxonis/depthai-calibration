import numpy as np
import cv2
import sys

ransacMethod = cv2.RANSAC
if cv2.__version__ >= "4.5.4":
    ransacMethod = cv2.USAC_MAGSAC
    print('Using MAGSAC')


debug = False

class FeaturesHelper:
    def __init__(self, filter_ratio = 0.6, reprojection_threshold = 3, detectPlanar = False):
        self.sift = cv2.SIFT_create()
        self.configMatcher()
        self.reprojection_threshold = reprojection_threshold
        self.filter_ratio = filter_ratio
        self.detectPlanar = detectPlanar

    def configMatcher(self, flann_index = 1, tree_count = 5, search_count = 50):
        index_params = dict(algorithm=flann_index, trees=tree_count)
        search_params = dict(checks=search_count)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def getFeatures(self, image):
        kps, des = self.sift.detectAndCompute(image, None)
            
        strengths = [kp.response for kp in kps]

        percentile_threshold = np.percentile(strengths, 70)
        kp_res = [kps[i] for i, response in enumerate(strengths) if response >= percentile_threshold]
        des_res = des[np.array(strengths) >= percentile_threshold, :]

        return kp_res, des_res

    def getMatchedFeatures(self, left_image, right_image):
        kp_left, des_left = self.getFeatures(left_image)
        kp_right, des_right = self.getFeatures(right_image)

        kp_left_filtered,\
        kp_right_filtered,\
        des_left_filtered, \
        des_right_filtered = self.filterMatches(kp_left, 
                                                kp_right,
                                                des_left,
                                                des_right)

        return kp_left_filtered, kp_right_filtered, des_left_filtered, des_right_filtered

    def getMatchedFeaturesWithPlanes(self, left_image, right_image):
        kp_left_filtered,\
        kp_right_filtered,\
        des_left_filtered,\
        des_right_filtered = self.getMatchedFeatures(left_image, right_image)

        if len(kp_left_filtered) < 25 or len(kp_right_filtered) < 25:
            return kp_left_filtered, kp_right_filtered, des_left_filtered, des_right_filtered

        kp_left_filtered,\
        kp_right_filtered,\
        des_left_filtered,\
        des_right_filtered = self.filterMatchesWithPlanes(kp_left_filtered, kp_right_filtered, des_left_filtered, des_right_filtered)

        return kp_left_filtered, kp_right_filtered, des_left_filtered, des_right_filtered
    
    def getMatchedFeaturesWithNonPlanar(self, left_image, right_image):
        kp_left_filtered,\
        kp_right_filtered,\
        des_left_filtered,\
        des_right_filtered = self.getMatchedFeatures(left_image, right_image)

        if len(kp_left_filtered) < 25 or len(kp_right_filtered) < 25:
            return kp_left_filtered, kp_right_filtered, des_left_filtered, des_right_filtered

        kp_left_filtered,\
        kp_right_filtered,\
        des_left_filtered,\
        des_right_filtered = self.filterMatchesWithNonPlanar(kp_left_filtered, kp_right_filtered, des_left_filtered, des_right_filtered)

        return kp_left_filtered, kp_right_filtered, des_left_filtered, des_right_filtered

    def filterMatches(self, kp_left, kp_right, des_left, des_right):
        pts_left_filtered = []
        pts_right_filtered = []
        kp_left_filtered = []
        kp_right_filtered = []
        des_left_filtered =  []
        des_right_filtered = []

        matches = self.flann.knnMatch(des_left, des_right, k=2)
        for m,n in matches:
            if m.distance < self.filter_ratio * n.distance:
                pts_left_filtered.append(kp_left[m.queryIdx].pt)
                kp_left_filtered.append(kp_left[m.queryIdx])
                des_left_filtered.append(des_left[m.queryIdx])

                pts_right_filtered.append(kp_right[m.trainIdx].pt)
                kp_right_filtered.append(kp_right[m.trainIdx])
                des_right_filtered.append(des_right[m.trainIdx])

        return kp_left_filtered, kp_right_filtered, des_left_filtered, des_right_filtered

    def filterMatchesWithPlanes(self, kp_left_filtered, kp_right_filtered, des_left_filtered, des_right_filtered):
        pts_left_filtered = keypoint_to_point2f(kp_left_filtered)
        pts_right_filtered = keypoint_to_point2f(kp_right_filtered)

        M, mask = cv2.findHomography(pts_left_filtered, pts_right_filtered, method=cv2.RANSAC, ransacReprojThreshold=self.reprojection_threshold)
        matchesMask = mask.ravel().tolist()
        for i in reversed(range(len(pts_left_filtered))):
            if not matchesMask[i]:
                del kp_left_filtered[i]
                del kp_right_filtered[i]
                del des_left_filtered[i]
                del des_right_filtered[i]
        return kp_left_filtered, kp_right_filtered, des_left_filtered, des_right_filtered


    def filterMatchesWithNonPlanar(self, kp_left_filtered, kp_right_filtered, des_left_filtered, des_right_filtered):
        pts_left_filtered = keypoint_to_point2f(kp_left_filtered)
        pts_right_filtered = keypoint_to_point2f(kp_right_filtered)
        F, mask = cv2.findFundamentalMat(pts_left_filtered, pts_right_filtered, ransacMethod, self.reprojection_threshold)
        matchesMask = mask.ravel().tolist()
        for i in reversed(range(len(pts_left_filtered))):
            if not matchesMask[i]:
                del kp_left_filtered[i]
                del kp_right_filtered[i]
                del des_left_filtered[i]
                del des_right_filtered[i]
        return kp_left_filtered, kp_right_filtered, des_left_filtered, des_right_filtered


    def draw_features(self, left_image, right_image, kp_left, kp_right):
        if len(left_image.shape) < 3:
            left_image = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
        if len(right_image.shape) < 3:
            right_image = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)
        horStack = np.hstack((left_image, right_image))

        green = (0, 255, 0)
        red = (0, 0, 255)
        # blue = (255, 0, 0)
        radius = 2
        thickness = 1
        size = left_image.shape[:2]
        for i in range(len(kp_left)):
            left_pt = kp_left[i].pt
            right_pt = kp_right[i].pt
            left_pt_i = (int(left_pt[0]), int(left_pt[1]))
            right_pt_i = (size[1] + int(right_pt[0]), int(right_pt[1]))
            
            epiploar_error = abs(left_pt[1] - right_pt[1])
            cv2.circle(horStack, left_pt_i, radius, red, thickness)
            cv2.circle(horStack, right_pt_i, radius, red, thickness)
            curr_color = green if epiploar_error < 0.7 else red
            
            horStack = cv2.line(horStack, left_pt_i, right_pt_i, curr_color, thickness)

        dest = cv2.resize(horStack, (0, 0), fx = 0.5, fy= 0.5, interpolation=cv2.INTER_AREA)
        return dest

    def calculate_epipolar_error(self, left_undistorted, right_undistorted):
        kp_left_filtered, kp_right_filtered, _, _ = self.getMatchedFeatures(left_undistorted, right_undistorted)

        # print(size)
        epiploar_error = 0

        for i in range(len(kp_left_filtered)):
            left_pt = kp_left_filtered[i].pt
            right_pt = kp_right_filtered[i].pt

            epiploar_error += abs(left_pt[1] - right_pt[1])
        
        marked_image = self.draw_features(left_undistorted, right_undistorted, kp_left_filtered, kp_right_filtered)
        if len(kp_left_filtered) < 20:
            return -(sys.maxsize - 1), marked_image
        epiploar_error /= len(kp_left_filtered)
        return epiploar_error, marked_image


def keypoint_to_point2f(kp):
    return np.array([kp[i].pt for i in range(len(kp))])

