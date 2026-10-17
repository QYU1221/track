"""Visible catalog-star projection and shared direction-to-pixel geometry."""
from dataclasses import dataclass
import numpy as np
from ..contracts import FrameInfo
from ..camera_model import CameraModel
from ..geometry import quaternion_matrix
from ..geometry_errors import GeometryError
from ..observation_geometry import frozen_array
from ..numerical_policy import UNIT_VECTOR_EPS_MULTIPLIER
from .contracts import AffinePixelModel,InferenceError


@dataclass(frozen=True)
class ProjectedKnownStar:
    source_id: str
    model: AffinePixelModel


def project_direction_pixel(mean: tuple,plane,frame: FrameInfo,camera: CameraModel,*,rotation=None,inertial_geometry=None) -> tuple:
    """Shared 2-D static/4-D CV geometric projection; units px per ratio."""
    u,J_inverse = plane.inverse(mean) if inertial_geometry is None else inertial_geometry
    R = quaternion_matrix(frame.quaternion_xyzw) if rotation is None else rotation
    b = R.T @ u
    pixel = camera.camera_to_pixel(b)
    image = -b[:2]/b[2]
    _,J_D = camera.distort(image)
    J_image_body = np.array([[-1/b[2],0.,b[0]/b[2]**2],[0.,-1/b[2],b[1]/b[2]**2]])
    H_pos = camera.K[:2,:2] @ J_D @ J_image_body @ R.T @ J_inverse
    return pixel,H_pos


def _catalog_sources(stars,lookup,rotation,camera,window):
    """Yield only stars whose declared means are inside the observation window."""
    for spec in stars:
        if spec.known_star_id not in lookup:
            raise InferenceError('CATALOG_ID_MISSING','star prior must reference the public catalog')
        star = lookup[spec.known_star_id]
        try:
            mean = camera.camera_to_pixel(rotation.T @ star.direction_j2000)
        except GeometryError as error:
            if error.code not in ('camera_singularity','distortion_domain'):
                raise
            continue
        if not window.contains(mean):
            continue
        model = AffinePixelModel(frozen_array(mean),spec.measurement_covariance_px2,
                                 spec.p_detection,window)
        yield ProjectedKnownStar('S_K:'+spec.known_star_id,model)


class VisibleStarProjector:
    """Cache immutable catalog inputs and shortlist online star sources.

    The existing mean-visibility model gives outside stars PD=0 and a unit
    miss factor. Only those neutral sources are omitted. Visible stars with
    no matching detection retain their full finite-window miss likelihood.
    """

    def __init__(self):
        self._stars = self._catalog = None

    def sources(self,stars,catalog,frame,camera,window):
        if not frame.inertial_observation_available:
            raise InferenceError('OBSERVATION_UNAVAILABLE','valid input and attitude required')
        if stars is not self._stars or catalog is not self._catalog:
            lookup = {star.known_star_id:star for star in catalog}
            ordered = tuple(sorted(stars,key=lambda star:star.known_star_id))
            ids = tuple(star.known_star_id for star in ordered)
            if len(set(ids))!=len(ids):
                raise ValueError('duplicate target/star source ID')
            if any(identifier not in lookup for identifier in ids):
                raise InferenceError('CATALOG_ID_MISSING','star prior must reference the public catalog')
            directions = np.asarray([lookup[identifier].direction_j2000 for identifier in ids]).reshape(-1,3)
            directions.setflags(write=False)
            self._lookup,self._ordered,self._directions = lookup,ordered,directions
            self._stars,self._catalog = stars,catalog
        if not self._ordered:
            return ()
        rotation = quaternion_matrix(frame.quaternion_xyzw)
        if camera.distortion_model=='none':
            return self._none_sources(rotation,camera,window)
        # Brown distortion keeps the general scalar path so its domain and
        # invertibility failures remain observable rather than being culled.
        return tuple(_catalog_sources(self._ordered,self._lookup,rotation,camera,window))

    def _none_sources(self,rotation,camera,window):
        """Exact selection for the pinhole model, batched elementwise ops.

        Per-star matvec, max, norm and projection keep the identical scalar
        kernels as camera_to_pixel; only the elementwise normalization,
        masks and model construction are batched. Results are bit-identical
        to the scalar path, so boundary stars select identically.
        """
        count=len(self._directions)
        body=np.empty((count,3))
        for index,direction in enumerate(self._directions):
            body[index]=rotation.T @ np.asarray(direction)
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            for index in range(count):
                row=body[index]
                row=row/np.max(np.abs(row))
                body[index]=row/np.linalg.norm(row)
        front = body[:,2] > UNIT_VECTOR_EPS_MULTIPLIER*np.finfo(float).eps
        indices = np.flatnonzero(front)
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            image = -body[indices,:2]/body[indices,2:3]
        intrinsic = camera.K
        pixels=np.empty((len(indices),2));radius=np.empty(len(indices))
        for position,index in enumerate(indices):
            pixels[position]=intrinsic[:2,:2] @ image[index]+intrinsic[:2,2]
            radius[position]=np.linalg.norm(image[index])
        x0,x1,y0,y1 = window.bounds_px
        keep = (np.isfinite(pixels).all(axis=1) &
                (radius<=camera.max_normalized_radius) &
                (pixels[:,0]>=x0) & (pixels[:,0]<x1) &
                (pixels[:,1]>=y0) & (pixels[:,1]<y1))
        models=[]
        for row,pixel in zip(indices[keep],pixels[keep]):
            spec=self._ordered[row]
            models.append(ProjectedKnownStar('S_K:'+spec.known_star_id,
                AffinePixelModel(frozen_array(pixel),spec.measurement_covariance_px2,
                                 spec.p_detection,window)))
        return tuple(models)
