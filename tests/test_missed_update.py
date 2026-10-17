import unittest
import json
from pathlib import Path
import numpy as np
from space_target_detection.v2.exact.contracts import GaussianState,ObservationWindow
from space_target_detection.v2.exact.numerics import gaussian_window_mass,gaussian_hit_updates
from space_target_detection.v2.tracking.missed_update import miss_conditioned_gaussian


class MissMomentTests(unittest.TestCase):
    def test_joseph_roundoff_regression_at_real_frame_328(self):
        r=json.loads(Path(__file__).with_name('joseph_roundoff_case.json').read_text())
        state=GaussianState(tuple(r['mean']),tuple(map(tuple,r['covariance'])))
        values=gaussian_hit_updates(state,np.array(r['H']),np.array(r['R']),r['residuals'])
        for value in values:
            C=np.asarray(value.covariance)
            np.testing.assert_array_equal(C,C.T)
            self.assertGreater(np.linalg.eigvalsh(C).min(),0.)

    def run_update(self,P,mu,window,pd=.9):
        if len(P)==2:
            full=np.eye(4);full[:2,:2]=P;P=full
        state=GaussianState(tuple(np.zeros(len(P))),tuple(map(tuple,P)))
        H=np.eye(len(P))[:2];R=np.eye(2)*.4;S=H@P@H.T+R
        mass,_=gaussian_window_mass(np.array(mu),S,window)
        return state,miss_conditioned_gaussian(state,H,R,np.array(mu),S,window,pd,mass),mass,S

    def test_half_plane_analytic_state_and_velocity_moments(self):
        P=np.eye(4);P[0,2]=P[2,0]=.3
        state,updated,mass,S=self.run_update(P,(0.,0.),ObservationWindow((0.,100.,-100.,100.)))
        self.assertAlmostEqual(mass,.5)
        expected=-.9/(1-.9*.5)*P[:,0]/np.sqrt(2*np.pi*S[0,0])
        np.testing.assert_allclose(updated.mean,expected,atol=1e-10)
        np.testing.assert_allclose(updated.covariance,P-np.outer(expected,expected),atol=1e-10)

    def test_symmetric_visible_window_can_increase_covariance(self):
        _,updated,_,_=self.run_update(np.eye(2),(0.,0.),ObservationWindow((-.5,.5,-.5,.5)))
        np.testing.assert_allclose(updated.mean,0.,atol=1e-10)
        self.assertGreater(updated.covariance[0][0],1.)

    def test_correlated_window_against_independent_monte_carlo(self):
        P=np.array(((1.,.45),(.45,1.5)));mu=np.array((.2,-.3));window=ObservationWindow((-.3,1.,-.7,.4))
        _,updated,mass,_=self.run_update(P,mu,window)
        rng=np.random.default_rng(914)
        x=rng.multivariate_normal(np.zeros(2),P,600000)
        z=mu+x+rng.normal(0,np.sqrt(.4),x.shape)
        inside=(z[:,0]>=-.3)&(z[:,0]<1.)&(z[:,1]>=-.7)&(z[:,1]<.4)
        weights=1-.9*inside;mean=np.average(x,axis=0,weights=weights)
        covariance=((x-mean).T*weights)@(x-mean)/weights.sum()
        self.assertAlmostEqual(mass,inside.mean(),delta=.002)
        np.testing.assert_allclose(updated.mean[:2],mean,atol=.006)
        np.testing.assert_allclose(np.asarray(updated.covariance)[:2,:2],covariance,atol=.01)

    def test_zero_detection_probability_retains_prior(self):
        state,updated,_,_=self.run_update(np.eye(2),(0.,0.),ObservationWindow((0.,1.,0.,1.)),pd=0.)
        self.assertEqual(state,updated)


if __name__=='__main__':unittest.main()
