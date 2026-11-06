"""Stereo uniformity changes: numerical and dependency tests, no hardware."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from . import calibration_utils as u


class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[580.,0,640],[0,580.,400],[0,0,1]])
        self.board = u.CharucoBoard(8, 6, 5., 3.7)
        self.objects = self.board.board.chessboardCorners
        self.R = cv2.Rodrigues(np.array([.01,-.02,.005]))[0]
        self.T = np.array([[3.75],[0.],[0.]])
        self.object_views=[];self.mono=[];self.color=[]
        rng = np.random.default_rng(29)
        for n in range(16):
            rv=rng.uniform([-.4,-.5,-.2],[.4,.5,.2]);t=np.array([[rng.uniform(-18,-3)],[rng.uniform(-14,3)],[rng.uniform(80,160)]])
            r=cv2.Rodrigues(rv)[0]
            self.object_views.append(self.objects.copy())
            self.mono.append(cv2.projectPoints(self.objects,rv,t,self.K,np.zeros(5))[0].reshape(-1,2))
            self.color.append(cv2.projectPoints(self.objects,cv2.Rodrigues(self.R@r)[0],self.R@t+self.T,self.K,np.zeros(5))[0].reshape(-1,2))
        self.camera={'intrinsics':self.K.copy(),'dist_coeff':np.zeros((5,1)),'size':(1280,800),
            'distortion_model':'NORMAL','extrinsics':{'specTranslation':{'x':3.75,'y':0.,'z':0.},'rotation':{'r':0,'p':0,'y':0}}}
        self.config=u.CalibrationConfig(15,.05,'perspective',(3,300,1e-9))

    def test_dependent_rgb_pose_keeps_final_lens_models_fixed(self):
        result,left,right=u.calibrate_stereo_perspective(self.config,self.object_views,self.mono,self.color,
                                                       copy.deepcopy(self.camera),copy.deepcopy(self.camera),False)
        np.testing.assert_array_equal(left['intrinsics'], self.K)
        np.testing.assert_array_equal(right['intrinsics'], self.K)
        np.testing.assert_allclose(result[1], self.R, atol=1e-5)
        np.testing.assert_allclose(result[2], self.T, atol=1e-3)

    def test_joint_fit_returns_corrected_models_for_both_worker_outputs(self):
        left=copy.deepcopy(self.camera);right=copy.deepcopy(self.camera)
        # Fix zero distortion so principal point and free lens tilt cannot trade off.
        left['distortion_model']=right['distortion_model']='111111111'
        left['intrinsics'][0,0]*=1.03;left['intrinsics'][1,1]*=1.03
        right['intrinsics'][0,0]*=.97;right['intrinsics'][1,1]*=.97
        result,left,right=u.calibrate_stereo_perspective(self.config,self.object_views,self.mono,self.color,left,right,True)
        np.testing.assert_allclose(left['intrinsics'],self.K,atol=.03)
        np.testing.assert_allclose(right['intrinsics'],self.K,atol=.03)
        np.testing.assert_allclose(result[2],self.T,atol=1e-3)

    def test_quality_metric_includes_mono_points_absent_from_stereo_subset(self):
        camera=dict(self.camera,name='left')
        dataset=SimpleNamespace(name='left',board=self.board)
        ids=[np.arange(len(self.objects),dtype=np.int32).reshape(-1,1) for _ in self.mono]
        baseline=u.update_mono_reprojection_error(camera,self.mono,ids,dataset)['reprojection_error']
        corrupted=copy.deepcopy(self.mono)
        corrupted[0][0,0]+=20  # This mono observation must not disappear behind stereo RMS.
        measured=u.update_mono_reprojection_error(camera,corrupted,ids,dataset)['reprojection_error']
        self.assertLess(baseline,1e-3)
        self.assertGreater(measured,.5)


class PipelineScopeTests(unittest.TestCase):
    def run_pipeline(self, enabled, single=False, filtered=True):
        board=u.CharucoBoard()
        names=['left'] if single else ['left','right','color']
        datasets={n:SimpleNamespace(name=n,id=i,board=board,enableFiltering=filtered,
            imageSize=(1280,800),allCorners=[np.zeros((12,1,2),np.float32)],
            allIds=[np.arange(12).reshape(-1,1)]) for i,n in enumerate(names)}
        cameras={n:{'name':n,'calib_model':'perspective_NORMAL','intrinsics':np.eye(3),
                    'dist_coeff':np.zeros(5),'reprojection_error':.1} for n in names}
        config={'cameras':cameras,'stereo_config':{'left_cam':'left','right_cam':'right'}}
        pairs=[] if single else [(datasets['right'],datasets['color']),(datasets['left'],datasets['right'])]
        settings=u.CalibrationConfig();settings.stereoRefinement=enabled
        relocalized=[];calls=[]
        def stereo(cfg,obj,l,r,left,right,refine):
            calls.append((left['name'],right['name'],refine))
            if refine:
                left=copy.deepcopy(left);right=copy.deepcopy(right)
                left['intrinsics'][0,0]=777;right['intrinsics'][0,0]=777
            elif enabled and filtered:
                self.assertEqual(left['intrinsics'][0,0],777)
            return [0,np.eye(3),np.zeros(3),np.eye(3),np.eye(3),None,None],left,right
        with patch.object(u,'get_features',side_effect=lambda cfg,cam:copy.deepcopy(cam)), \
             patch.object(u,'estimate_pose_and_filter',side_effect=lambda cam,c,i,board:(c,i,None)), \
             patch.object(u,'calibrate_charuco',side_effect=lambda cam,c,i,data:cam), \
             patch.object(u,'calibrate_ccm_intrinsics_per_ccm',side_effect=lambda cfg,cam,data:(cam,data.allCorners,data.allIds)), \
             patch.object(u,'drop_indices',side_effect=lambda *args:args), \
             patch.object(u,'find_stereo_common_features',return_value=([],[],[])), \
             patch.object(u,'calibrate_stereo_perspective',side_effect=stereo), \
             patch.object(u,'update_mono_reprojection_error',side_effect=lambda camera,*args:dict(camera,reprojection_error=.25)), \
             patch.object(u,'calculate_epipolar_error',side_effect=lambda l,r,*a:(l,None)):
            output=u.calibrate_camera(settings,config,'perspective',list(datasets.values()),pairs)
        return relocalized,calls,output[0]['stereo_refinement']

    def test_primary_pair_is_refined_before_existing_rgb_pose_solve(self):
        corners,calls,trace=self.run_pipeline(True)
        self.assertEqual(corners,[])
        self.assertEqual(calls,[('left','right',True),('right','color',False)])
        self.assertEqual(trace['primaryCameras'],['left','right'])
        self.assertFalse(trace['dependentIntrinsicsRefined'])

    def test_disabled_option_keeps_original_pair_order_and_single_pass(self):
        corners,calls,trace=self.run_pipeline(False)
        self.assertEqual(corners,[])
        self.assertEqual(calls,[('right','color',False),('left','right',False)])
        self.assertFalse(trace['applied'])

    def test_single_camera_and_unfiltered_datasets_do_not_enable_refinement(self):
        for single,filtered in [(True,True),(False,False)]:
            corners,calls,trace=self.run_pipeline(True,single=single,filtered=filtered)
            self.assertEqual(corners,[])
            self.assertFalse(trace['applied'])


if __name__=='__main__':unittest.main()
