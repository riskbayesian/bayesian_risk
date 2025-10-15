"""
Open World Risk vision pipeline package.
"""

from .point_cloud_3D import PointCloudGenerator

__all__ = ['PointCloudGenerator']

# Optional imports
try:
    from .compute_obj_part_feature import FeatureExtractor
    __all__.append('FeatureExtractor')
except ImportError:
    print("===================================================")
    print("🚨🚨🚨 FeatureExtractor import failed, continuing without it... 🚨🚨🚨")
    print("===================================================")
    pass
