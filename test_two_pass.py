"""Numerical regression tests. No device or controller imports."""
import copy
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from . import calibration_utils as u
from .lens_validation import validate_perspective_lens, dependent_lens_fallback


class LensTests(unittest.TestCase):
    def test_valid_pinhole_includes_sensor_corners(self):
        K = np.array([[570.,0,640],[0,570.,400],[0,0,1]])
        audit = validate_perspective_lens(K, np.zeros(14), (1280,800))
        self.assertTrue(audit['valid'])
        self.assertEqual(audit['samples'], 101*161)

    def test_folded_lens_cannot_pass_with_good_center(self):
        K = np.array([[570.,0,640],[0,570.,400],[0,0,1]])
        audit = validate_perspective_lens(K, [-.5,0,0,0,0], (1280,800))
        self.assertFalse(audit['valid'])
        self.assertGreater(audit['roundtripFailures'], 0)


class LensFallbackTests(unittest.TestCase):
    def setUp(self):
        self.config={'cameras':{'A':{'name':'color','calib_model':'perspective_TILTED'},
            'B':{'name':'left','calib_model':'perspective_TILTED'},
            'C':{'name':'right','calib_model':'perspective_TILTED'}},
            'stereo_config':{'left_cam':'B','right_cam':'C'}}

    def test_only_invalid_dependent_model_is_simplified(self):
        changed, log=dependent_lens_fallback(self.config,{'A':{'valid':False},'B':{'valid':True},'C':{'valid':True}})
        self.assertEqual(changed['cameras']['A']['calib_model'],'perspective_RATIONAL')
        self.assertEqual(changed['cameras']['B'],self.config['cameras']['B'])
        self.assertEqual(changed['cameras']['C'],self.config['cameras']['C'])
        self.assertEqual(self.config['cameras']['A']['calib_model'],'perspective_TILTED')
        self.assertEqual(log[0]['camera'],'color')

    def test_invalid_primary_model_never_chooses_a_depth_tuned_fallback(self):
        changed, log=dependent_lens_fallback(self.config,{'B':{'valid':False}})
        self.assertIsNone(changed)
        self.assertEqual(log,[])

    def test_rational_retry_failure_is_not_silently_accepted(self):
        self.config['cameras']['A']['calib_model']='perspective_RATIONAL'
        changed,log=dependent_lens_fallback(self.config,{'A':{'valid':False}})
        self.assertIsNone(changed)
        self.assertEqual(log,[])


class CornerTests(unittest.TestCase):
    def test_second_pass_uses_first_fit_and_keeps_image_order(self):
        board = u.CharucoBoard()
        images = [np.full((80,100), value, np.uint8) for value in [10,20]]
        dataset = u.Dataset('right', board, images=u.Dataset.Images(images))
        K = np.eye(3); D = np.zeros(14)
        seen = []
        def interpolate(markers, ids, image, b, **kwargs):
            self.assertEqual(kwargs['minMarkers'], 2)
            np.testing.assert_array_equal(kwargs['cameraMatrix'], K)
            np.testing.assert_array_equal(kwargs['distCoeffs'], D)
            seen.append(int(image[0,0]))
            return 12, np.full((12,1,2), image[0,0], np.float32), np.arange(12).reshape(-1,1)
        with patch.object(cv2.aruco, 'detectMarkers', return_value=([], np.array([[0]]), [])), \
             patch.object(cv2.aruco, 'refineDetectedMarkers', return_value=([], np.array([[0]]), [], [])), \
             patch.object(cv2.aruco, 'interpolateCornersCharuco', side_effect=interpolate), \
             patch.object(cv2, 'cornerSubPix', side_effect=lambda image,c,*a: c):
            corners, ids = u.redetect_calibrated_corners({'intrinsics':K,'dist_coeff':D},dataset)
        self.assertEqual(seen, [10,20])
        self.assertEqual([float(c[0,0,0]) for c in corners], [10,20])

    def test_missing_second_pass_view_aborts_instead_of_reindexing_pairs(self):
        dataset = u.Dataset('right',u.CharucoBoard(),images=u.Dataset.Images([np.zeros((80,100),np.uint8)]))
        with patch.object(cv2.aruco,'detectMarkers',return_value=([],None,[])), \
             patch.object(cv2.aruco,'refineDetectedMarkers',return_value=([],None,[],[])):
            with self.assertRaisesRegex(RuntimeError,'capture 0'):
                u.redetect_calibrated_corners({'intrinsics':np.eye(3),'dist_coeff':np.zeros(14)},dataset)


class StereoTests(unittest.TestCase):
    def test_dependent_color_extrinsics_use_final_mono_intrinsics(self):
        K = np.array([[580.,0,640],[0,580.,400],[0,0,1]])
        board = u.CharucoBoard(8,6,5.,3.7).board.chessboardCorners
        objects=[]; mono=[]; color=[]
        R = cv2.Rodrigues(np.array([.01,-.02,.005]))[0]; T=np.array([[3.75],[0.],[0.]])
        for n in range(8):
            rv=np.array([.08*n-.25,.06*n-.2,.02*n]);t=np.array([[-15.+n],[2.-n],[95.+3*n]])
            r=cv2.Rodrigues(rv)[0]
            objects.append(board.copy())
            mono.append(cv2.projectPoints(board,rv,t,K,np.zeros(5))[0].reshape(-1,2))
            color.append(cv2.projectPoints(board,cv2.Rodrigues(R@r)[0],R@t+T,K,np.zeros(5))[0].reshape(-1,2))
        cam={'intrinsics':K.copy(),'dist_coeff':np.zeros((5,1)),'size':(1280,800),
             'distortion_model':'NORMAL','extrinsics':{'specTranslation':{'x':3.75,'y':0.,'z':0.},'rotation':{'r':0,'p':0,'y':0}}}
        config=u.CalibrationConfig(15,.05,'perspective',(3,300,1e-9))
        result,left,right=u.calibrate_stereo_perspective(config,objects,mono,color,copy.deepcopy(cam),copy.deepcopy(cam),False)
        np.testing.assert_array_equal(left['intrinsics'],K)
        np.testing.assert_array_equal(right['intrinsics'],K)
        np.testing.assert_allclose(result[1],R,atol=1e-5)
        np.testing.assert_allclose(result[2],T,atol=1e-3)

if __name__=='__main__': unittest.main()
