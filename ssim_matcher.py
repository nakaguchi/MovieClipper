"""
pHash + SSIM based image matcher.
First filters by pHash distance, then computes SSIM for matching frames.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from matcher_base import Matcher
from phash_matcher import phash_64, hamming_distance_64


def preprocess_for_ssim(frame_bgr: np.ndarray, size: int = 256) -> np.ndarray:
    """
    Preprocess image for SSIM computation.
    
    Args:
        frame_bgr: Input image in BGR format
        size: Target size for square resizing
        
    Returns:
        Preprocessed grayscale image
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return gray


def compute_ssim(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    """
    Compute SSIM between two grayscale images.
    
    Args:
        a_gray: First image in grayscale
        b_gray: Second image in grayscale
        
    Returns:
        SSIM score in range [-1, 1], typically [0, 1]
    """
    return float(ssim(a_gray, b_gray, data_range=255))


class SSIM_Matcher(Matcher):
    """
    Matcher using pHash + SSIM.
    
    Strategy:
    1. Filter frames using pHash Hamming distance (fast)
    2. Compute SSIM only for frames that pass pHash threshold (more expensive)
    3. Combined similarity = max(pHash-based, SSIM-based) * SSIM score
    """
    
    def __init__(self, hd_threshold: int = 20, ssim_size: int = 256):
        """
        Initialize SSIM matcher.
        
        Args:
            hd_threshold: pHash Hamming distance threshold
            ssim_size: Size for SSIM computation (square)
        """
        super().__init__()
        self.ref_hash = None
        self.ref_ssim_gray = None
        self.hd_threshold = hd_threshold
        self.ssim_size = ssim_size
    
    def set_reference(self, ref_bgr: np.ndarray) -> None:
        """
        Set the reference image and compute its pHash and SSIM representation.
        
        Args:
            ref_bgr: Reference image in BGR format
        """
        super().set_reference(ref_bgr)
        if self.ref_bgr is not None:
            self.ref_hash = phash_64(self.ref_bgr)
            self.ref_ssim_gray = preprocess_for_ssim(self.ref_bgr, size=self.ssim_size)
    
    def compute_similarity(self, frame_bgr: np.ndarray) -> float:
        """
        Compute similarity using pHash (fast filter) + SSIM (detailed matching).
        
        Args:
            frame_bgr: Input frame in BGR format
            
        Returns:
            Similarity score [0.0, 1.0]
        """
        if self.ref_hash is None or self.ref_ssim_gray is None:
            return 0.0
        
        try:
            # Step 1: pHash filtering
            frame_hash = phash_64(frame_bgr)
            self.hamming_distance = hamming_distance_64(frame_hash, self.ref_hash)
            
            # If pHash distance is too large, return low score
            if self.hamming_distance > self.hd_threshold:
                return 0.0
            
            # Step 2: SSIM computation (only if pHash passes)
            frame_ssim_gray = preprocess_for_ssim(frame_bgr, size=self.ssim_size)
            ssim_score = compute_ssim(frame_ssim_gray, self.ref_ssim_gray)
            
            # Normalize SSIM to [0, 1] range (SSIM can be -1 to 1, but typically 0 to 1)
            ssim_normalized = max(0.0, min(1.0, (ssim_score + 1.0) / 2.0))
            
            # Combine: heavily weighted on SSIM, with slight boost for good pHash match
            phash_score = 1.0 - (self.hamming_distance / 64.0)
            combined = 0.9 * ssim_normalized + 0.1 * phash_score
            
            return min(1.0, combined)
        
        except Exception:
            return 0.0
