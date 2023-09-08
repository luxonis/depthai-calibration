import numpy as np
import cv2
import sys

debug = False

class FeaturesHelper:
    def __init__(self, filter_ratio = 0.6, reprojection_threshold = 3):
        self.sift = cv2.SIFT_create()
        self.configMatcher()
        self.reprojection_threshold = reprojection_threshold
        self.filter_ratio = filter_ratio

    def configMatcher(self, flann_index = 1, tree_count = 5, search_count = 50):
        index_params = dict(algorithm=flann_index, trees=tree_count)
        search_params = dict(checks=search_count)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def getMatchedFeatures(self, left_image, right_image):
        kp_left, des_left = self.sift.detectAndCompute(left_image, None)
        kp_right, des_right = self.sift.detectAndCompute(right_image, None)

        # sorted_indices = [i[0] for i in sorted(enumerate(kp_left), key=lambda x: x[1].response, reverse=True)]
        # kp_left = [kp_left[i] for i in sorted_indices]
        # # des_left = [des_left[i] for i in sorted_indices]
        # des_left = des_left[sorted_indices, :]

        # sorted_indices = [i[0] for i in sorted(enumerate(kp_right), key=lambda x: x[1].response, reverse=True)]
        # kp_right = [kp_right[i] for i in sorted_indices]
        # des_right = des_right[sorted_indices, :]
    
        strengths_left = [kp.response for kp in kp_left]
        strengths_right = [kp.response for kp in kp_right]

        percentile_threshold = np.percentile(strengths_left, 70)
        kp_left = [kp_left[i] for i, response in enumerate(strengths_left) if response >= percentile_threshold]
        des_left = des_left[np.array(strengths_left) >= percentile_threshold, :]

        percentile_threshold = np.percentile(strengths_right, 70)
        kp_right = [kp_right[i] for i, response in enumerate(strengths_right) if response >= percentile_threshold]
        des_right = des_right[np.array(strengths_right) >= percentile_threshold, :]

        if debug:
            print(f'SIze of kps is {len(kp_right)} and {len(kp_right)}')
            print(f'SIze of des is {len(des_left)} and {len(des_right)}')

            print(f'Median response ----------> {np.median(strengths_left)} --> {np.median(strengths_right)}')
            print(f'Mean response ----------> {np.mean(strengths_left)} --> {np.mean(strengths_right)}')
            print(f'Max response ----------> {np.max(strengths_left)} --> {np.max(strengths_right)}')
            print(f'Min response ----------> {np.min(strengths_left)} --> {np.min(strengths_right)}')

            print(f'percentile threshold value is  -------------- {percentile_threshold}')


        kp_left_filtered,\
        kp_right_filtered,\
        des_left_filtered, \
        des_right_filtered = self.filterMatches(kp_left, 
                                                kp_right,
                                                des_left,
                                                des_right)

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

        if len(kp_left_filtered) < 25 or len(kp_right_filtered) < 25:
            return kp_left_filtered, kp_right_filtered, np.array(des_left_filtered), np.array(des_right_filtered)

        pts_left_filtered = np.float32(pts_left_filtered)
        pts_right_filtered = np.float32(pts_right_filtered)


        # this is just to get inliers
        # TODO(saching12): Check if this keeps only the points on the plane ? Since it is a perspective transform validator
        # If that is how it is doing. that means we are filtering the planar points.
        M, mask = cv2.findHomography(pts_left_filtered, pts_right_filtered, method=cv2.RANSAC, ransacReprojThreshold=self.reprojection_threshold)
        matchesMask = mask.ravel().tolist()
        for i in reversed(range(len(pts_left_filtered))):
            if not matchesMask[i]:
                del kp_left_filtered[i]
                del kp_right_filtered[i]
                del des_left_filtered[i]
                del des_right_filtered[i]
        return kp_left_filtered, kp_right_filtered, np.array(des_left_filtered), np.array(des_right_filtered)
    
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

