"""
Base class for image matching similarity computation.
"""

from abc import ABC, abstractmethod
import numpy as np


class Matcher(ABC):
    """
    Abstract base class for computing similarity between two images.
    
    Subclasses should implement the compute_similarity method to provide
    different matching algorithms (ORB features, SSIM, pHash, etc.).
    """
    
    def __init__(self):
        """Initialize the matcher."""
        self.hamming_distance = None
    
    @abstractmethod
    def compute_similarity(self, frame_bgr: np.ndarray) -> float:
        """
        Compute similarity score between the reference image and the given frame.
        
        Args:
            frame_bgr: Input frame in BGR format (numpy array)
        
        Returns:
            Similarity score in range [0.0, 1.0], where 1.0 means perfect match
        """
        pass
    
    def set_reference(self, ref_bgr: np.ndarray) -> None:
        """
        Set the reference image for comparison.
        
        Args:
            ref_bgr: Reference image in BGR format (numpy array)
        """
        self.ref_bgr = ref_bgr.copy() if ref_bgr is not None else None
