"""Dataset and per-frame detection input."""

from .detection_loader import DetectionLoader
from utils.catalog_loader import CatalogData, load_star_catalog

__all__ = ["DetectionLoader", "CatalogData", "load_star_catalog"]
