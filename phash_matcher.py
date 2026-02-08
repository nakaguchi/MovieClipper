"""
pHash (Perceptual Hash) based image matcher.
Computes similarity using only pHash ハミング距離.
"""

import cv2
import numpy as np
from matcher_base import Matcher


def phash_64(gray_bgr_or_gray: np.ndarray) -> int:
    """
    Compute 64-bit perceptual hash.
    
    Args:
        gray_bgr_or_gray: Image in BGR or grayscale format
        
    Returns:
        64-bit hash as integer
    """
    if gray_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(gray_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_bgr_or_gray

    img = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(img)
    dct_low = dct[:8, :8].copy()
    dct_low[0, 0] = 0.0  # ignore DC

    med = np.median(dct_low)
    bits = (dct_low > med).astype(np.uint8).flatten()

    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming_distance_64(a: int, b: int) -> int:
    """
    Compute Hamming distance between two 64-bit hashes.
    
    Args:
        a: First hash
        b: Second hash
        
    Returns:
        Hamming distance (number of differing bits)
    """
    def bitcount(x: int) -> int:
        if hasattr(int, "bit_count"):
            return x.bit_count()
        return bin(x).count("1")
    
    return bitcount(a ^ b)


class pHash_Matcher(Matcher):
    """
    Matcher using pHash (Perceptual Hash) only.
    
    Computes similarity as inverse of normalized Hamming distance.
    Similarity = 1.0 - (hamming_distance / 64.0)
    """
    
    def __init__(self, hamming_dist_ubound: int = 50):
        """
        Initialize pHash matcher.
        
        Args:
            hamming_dist_ubound: Maximum hamming distance to consider as match.
                            Distances above this threshold result in similarity 0.0.
        """
        super().__init__()
        self.ref_hash = None
        self.hamming_dist_ubound = hamming_dist_ubound
    
    def set_reference(self, ref_bgr: np.ndarray) -> None:
        """
        Set the reference image and compute its pHash.
        
        Args:
            ref_bgr: Reference image in BGR format
        """
        super().set_reference(ref_bgr)
        if self.ref_bgr is not None:
            self.ref_hash = phash_64(self.ref_bgr)
    
    def compute_similarity(self, frame_bgr: np.ndarray) -> float:
        """
        Compute similarity using pHash Hamming distance.
        
        Args:
            frame_bgr: Input frame in BGR format
            
        Returns:
            Similarity score [0.0, 1.0]. Score = 1.0 - (hd / 64.0) if hd <= max_dist, else 0.0
        """
        if self.ref_hash is None:
            return 0.0
        
        try:
            frame_hash = phash_64(frame_bgr)
            self.hamming_distance = min(hamming_distance_64(frame_hash, self.ref_hash), \
                                        self.hamming_dist_ubound)
            
            # Normalize to [0, 1]: 1.0 = perfect match (distance 0)
            similarity = 1.0 - (self.hamming_distance / self.hamming_dist_ubound)
            return similarity
        except Exception:
            return 0.0
