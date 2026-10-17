import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from space_target_detection.v2.exact.contracts import GaussianState
from space_target_detection.v2.geometry_errors import GeometryError
from space_target_detection.v2.tracking.model import predict_gaussian
from space_target_detection.v2.tracking import online_factors as factors
from space_target_detection.v2.efficient.solver import GraphInferencePolicy
from space_target_detection.v2.reporting.artifacts import XStyleReporter


class AssociationStabilityTests(unittest.TestCase):
    def setUp(self):
        self.state=GaussianState((0.,0.,.01,.02),tuple(map(tuple,np.eye(4)*.1)))
        self.detections=(SimpleNamespace(key=(1,'a'),x_px=0.,y_px=0.),)

    def test_gap_covariance_uses_incremental_physical_time(self):
        direct=predict_gaussian(self.state,8.,3e-8)
        repeated=self.state
        for dt in (2.,4.,2.):repeated=predict_gaussian(repeated,dt,3e-8)
        np.testing.assert_allclose(direct.mean,repeated.mean,rtol=1e-14)
        np.testing.assert_allclose(direct.covariance,repeated.covariance,rtol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(direct.covariance).min(),0.)

    def request(self,index):
        return ('T:'+str(index),'T',self.state,.8,index,.9,np.eye(2))

    def projection(self,state,plane,*args):
        if plane%2:raise GeometryError('camera_singularity','outside')
        return np.zeros(2),np.column_stack((np.eye(2),np.zeros((2,2)))),np.eye(2),np.eye(2)*1.1,.8

    def test_mixed_batch_matches_scalar_and_keeps_order(self):
        requests=tuple(self.request(i) for i in range(18))
        with patch.object(factors,'_affine_projection',side_effect=self.projection):
            batch=factors.affine_factors_many(requests,None,None,None,self.detections,.99)
            scalar=tuple(factors.affine_factors(*r[:5],None,None,None,self.detections,*r[5:],.99) for r in requests)
        for a,b in zip(batch,scalar):
            self.assertEqual(a[0],b[0]);self.assertEqual(a[1],b[1]);self.assertEqual(a[2],b[2])

    def test_all_invisible_batch_with_detections(self):
        with patch.object(factors,'_affine_projection',side_effect=GeometryError('tangent_domain','outside')):
            result=factors.affine_factors_many(tuple(self.request(i) for i in range(16)),None,None,None,self.detections,.99)
        for factor,states,geometry in result:
            self.assertEqual(factor.pd_bar,0.);self.assertEqual(factor.log_eta,(-math.inf,))
            self.assertEqual(states,{None:self.state});self.assertIsNone(geometry)

    def test_invalid_geometry_is_not_silenced(self):
        with patch.object(factors,'_affine_projection',side_effect=GeometryError('distortion_numeric','invalid')):
            with self.assertRaises(GeometryError):
                factors.affine_factors_many(tuple(self.request(i) for i in range(16)),None,None,None,self.detections,.99)

    def test_compact_audit_reports_conditional_discarded_mass(self):
        policy=GraphInferencePolicy(SimpleNamespace(top_k=2),None,detail='compact')
        keep=policy.select([(None,SimpleNamespace(hypothesis_id=str(i))) for i in range(3)],np.log([.5,.3,.2]))
        self.assertEqual(keep,(0,1))
        self.assertAlmostEqual(policy.audit['discarded_mass_within_evaluated_candidates'],.2)
        self.assertIn('UNKNOWN',policy.audit['omitted_mass_status'])

    def test_online_identity_is_not_reported_as_final_identity(self):
        reporter=object.__new__(XStyleReporter)
        reporter.frame_info={1:SimpleNamespace(timestamp_s=2.)}
        reporter.detections={1:[SimpleNamespace(detection_id='a',key=(1,'a'),x_px=2.,y_px=3.)]}
        reporter.label_ids={};reporter.tracks={};reporter.points={};reporter.frames=[];reporter.events=[]
        reporter.reliable=.7;reporter.export_seconds=0.
        reporter._detection_row=lambda *args:dict(Final_Canonical_ID=None)
        reporter._frame_row=lambda *args:{};reporter._save_frame=lambda *args:None
        reporter.progress=SimpleNamespace(frame=lambda *args:None)
        marginal=dict(Track_Label='label',Track_Maturity='CONFIRMED',Track_Lifecycle='ACTIVE',
                      Existence_Probability=.95,Canonical_Track_ID='canonical',Management_Reason='test',
                      Measured_Or_Predicted='Measured_Update_Mixture',Miss_Association_Probability=.2)
        result=dict(frame=1,track_marginals=[marginal],points=[dict(Detection_ID='a',Association_Marginals=[('T:label',.8)])])
        reporter.consume(result)
        row=reporter.points[(1,'a')]
        self.assertEqual(row['Identity_Posterior'],.8)
        self.assertEqual(row['Identity_Status'],'ONLINE_ASSOCIATION_ONLY')
        self.assertIsNone(row['Final_Canonical_ID'])
        self.assertEqual(reporter.tracks['label']['rows'][0]['posterior_update_kind'],'Measured_Update_Mixture')


if __name__=='__main__':unittest.main()
