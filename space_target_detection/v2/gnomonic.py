"""Fixed gnomonic observation coordinates; dimensionless ratios, geometry1.0."""
from dataclasses import dataclass
from typing import Iterable, Tuple
from .numerical_policy import BASIS_EPS_MULTIPLIER, UNIT_VECTOR_EPS_MULTIPLIER
import numpy as np
from .contracts import finite
from .geometry import unit
from .geometry_errors import GeometryError


@dataclass(frozen=True)
class GnomonicPlane:
    reference_id: str
    center: tuple
    max_angle_rad: float
    e1: tuple = None
    e2: tuple = None

    def __post_init__(self):
        if not isinstance(self.reference_id,str) or not self.reference_id.strip():
            raise ValueError('tangent reference ID must be nonempty')
        center=finite(self.center,'tangent center')
        # Preserve a canonical plane exactly across checkpoint reconstruction.
        # Re-normalizing an already-unit vector can change its last bits and
        # rotate the restored coordinate system relative to its saved states.
        c=(center if center.shape==(3,) and np.isclose(center @ center,1.,rtol=0,
            atol=UNIT_VECTOR_EPS_MULTIPLIER*np.finfo(float).eps) else unit(center))
        finite(self.max_angle_rad, 'tangent angular domain')
        if not 0 < self.max_angle_rad < np.pi/2:
            raise ValueError('tangent domain must be in (0,pi/2) radians')
        if (self.e1 is None) != (self.e2 is None):
            raise ValueError('provide both tangent basis vectors or neither')
        if self.e1 is None:
            axis = np.eye(3)[int(np.argmin(np.abs(c)))]
            e1 = unit(axis-c*(axis @ c))
            e2 = np.cross(c,e1)
        else:
            e1,e2 = finite(self.e1,'e1'),finite(self.e2,'e2')
            if e1.shape != (3,) or e2.shape != (3,):
                raise ValueError('tangent basis vectors must be 3D')
        basis = np.column_stack((e1,e2,c))
        tol = BASIS_EPS_MULTIPLIER*np.finfo(float).eps
        if not np.allclose(basis.T @ basis,np.eye(3),rtol=0,atol=tol) or abs(np.linalg.det(basis)-1) > tol:
            raise ValueError('tangent basis must be orthonormal and right handed')
        for name,value in (('center',c),('e1',e1),('e2',e2)):
            object.__setattr__(self,name,tuple(value))

    def basis(self) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        return np.asarray(self.center),np.asarray(self.e1),np.asarray(self.e2)

    def project(self, direction: Iterable[float]) -> Tuple[np.ndarray,np.ndarray]:
        """Ratios (2,), derivative (2,3) with respect to input direction.

        A non-unit input is allowed; homogeneity makes ratios identical, but
        the derivative correctly includes its norm. Validity uses unit LOS.
        """
        raw = finite(direction,'J2000 direction')
        u = unit(raw)
        c,e1,e2 = self.basis()
        d = float(c @ u)
        if d <= np.cos(self.max_angle_rad):
            raise GeometryError('tangent_domain','LOS outside declared gnomonic domain')
        position = np.array([e1 @ u,e2 @ u])/d
        scale = np.max(np.abs(raw))
        J = (np.vstack((e1-position[0]*c,e2-position[1]*c))/d/scale)/np.linalg.norm(raw/scale)
        return finite(position,'gnomonic position'),finite(J,'gnomonic Jacobian')

    def inverse(self, position: Iterable[float]) -> Tuple[np.ndarray,np.ndarray]:
        """Recover unit LOS (3,) and dLOS/dposition (3,2)."""
        p = finite(position,'gnomonic position')
        if p.shape != (2,):
            raise ValueError('gnomonic position must have dimension 2')
        c,e1,e2 = self.basis()
        if np.linalg.norm(p) >= np.tan(self.max_angle_rad):
            raise GeometryError('tangent_domain','gnomonic position outside declared domain')
        a = c+p[0]*e1+p[1]*e2
        norm = np.linalg.norm(a)
        u = a/norm
        J = (np.eye(3)-np.outer(u,u)) @ np.column_stack((e1,e2))/norm
        return u,J
