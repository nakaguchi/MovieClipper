"""
ORB feature-based image matcher.
Detects and matches ORB features between reference and frame images.
"""

from typing import List, Optional
import cv2
import numpy as np

from matcher_base import Matcher


class ORB_Matcher(Matcher):
    """
    Matcher using ORB (Oriented FAST and Rotated BRIEF) features.
    
    Detects keypoints and descriptors in both reference and frame images,
    performs feature matching via RANSAC homography, and returns confidence score.
    """
    
    def __init__(
        self,
        nfeatures: int = 1500,
        feature_threshold: float = 0.7,
        angle_tol: float = 15.0,
        min_good_matches: int = 4,
        match_size: int = 320,
        visualize: bool = False,
    ):
        """
        Initialize ORB feature matcher.
        
        Args:
            nfeatures: Number of ORB features to detect
            feature_threshold: Lowe's ratio test threshold (0.0-1.0)
            angle_tol: Angle tolerance for keypoint orientation filtering (degrees)
            min_good_matches: Minimum good matches required for homography calculation
        """
        super().__init__()
        self.detector = cv2.ORB_create(nfeatures=nfeatures)
        self.kp_ref: Optional[List[cv2.KeyPoint]] = None
        self.desc_ref: Optional[np.ndarray] = None
        self.feature_threshold = feature_threshold
        self.angle_tol = angle_tol
        self.min_good_matches = min_good_matches
        self.match_size = (match_size, match_size//4*3)  # Maintain 4:3 aspect ratio
        self.num_good_matches = 0
        self.visualize = visualize
    
    def set_reference(self, ref_bgr: np.ndarray) -> None:
        """
        Set reference image and compute its ORB features.
        
        Args:
            ref_bgr: Reference image in BGR format
        """
        super().set_reference(ref_bgr)
        if self.ref_bgr is not None:
            try:
                ref_gray = cv2.cvtColor(self.ref_bgr, cv2.COLOR_BGR2GRAY)
                ref_gray_resized = cv2.resize(ref_gray, self.match_size, interpolation=cv2.INTER_AREA)
                self.kp_ref, self.desc_ref = self.detector.detectAndCompute(ref_gray_resized, None)
                self.ref_disp = cv2.resize(self.ref_bgr, self.match_size, interpolation=cv2.INTER_AREA)
                for kp in self.kp_ref:
                    cv2.circle(self.ref_disp, (int(kp.pt[0]), int(kp.pt[1])), 2, (0, 255, 0), -1)
            except Exception:
                self.kp_ref = None
                self.desc_ref = None
    
    def compute_similarity(self, frame_bgr: np.ndarray) -> float:
        """
        Compute similarity via ORB feature matching and RANSAC homography.
        
        Args:
            frame_bgr: Input frame in BGR format
            
        Returns:
            Similarity score [0.0, 1.0], where 1.0 means perfect match
        """
        if self.desc_ref is None or self.kp_ref is None:
            return 0.0
        
        if len(self.kp_ref) < self.min_good_matches:
            return 0.0
        
        try:
            # Detect features in frame
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frame_gray_resized = cv2.resize(frame_gray, self.match_size, interpolation=cv2.INTER_AREA)
            
            kp_frame, desc_frame = self.detector.detectAndCompute(frame_gray_resized, None)
            
            if desc_frame is None or len(kp_frame) < self.min_good_matches:
                return 0.0
            
            # BFMatcher for ORB (Hamming distance)
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            
            # KNN matching (k=2 for Lowe's ratio test)
            matches = matcher.knnMatch(self.desc_ref, desc_frame, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < self.feature_threshold * n.distance:
                        good_matches.append(m)
            
            # Filter by angle difference if angle_tol is set
            if len(good_matches) > 0 and self.angle_tol is not None and self.angle_tol >= 0.0:
                filtered = []
                for m in good_matches:
                    ang_ref = self.kp_ref[m.queryIdx].angle if m.queryIdx < len(self.kp_ref) else 0.0
                    ang_fr = kp_frame[m.trainIdx].angle if m.trainIdx < len(kp_frame) else 0.0
                    diff = abs(((ang_ref - ang_fr + 180.0) % 360.0) - 180.0)
                    if diff <= self.angle_tol:
                        filtered.append(m)
                good_matches = filtered

            # 可視化（マッチ表示）
            if self.visualize:
                try:
                    frame_disp = cv2.resize(frame_bgr, self.match_size, interpolation=cv2.INTER_AREA)
                    for kp in kp_frame:
                        cv2.circle(frame_disp, (int(kp.pt[0]), int(kp.pt[1])), 2, (0, 255, 0), -1)
                    img_matches = cv2.drawMatches(self.ref_disp, self.kp_ref, frame_disp, kp_frame, good_matches, None, flags=2)
                    # txt = f"score={score:.3f} inliers={int(inliers)}/{len(good_matches)}"
                    txt = f"good_matches={len(good_matches)}"
                    cv2.putText(img_matches, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("feature_matches", img_matches)
                    cv2.waitKey(1)
                except Exception:
                    pass

            # Check minimum matches for homography
            self.num_good_matches = len(good_matches)
            if self.num_good_matches < self.min_good_matches:
                return 0.0
            
            # # Compute homography via RANSAC
            # src_pts = np.float32([self.kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            # dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            # if H is None:
            #     return 0.0
            
            # # Score = inlier ratio
            # inliers = np.sum(mask)
            # score = float(inliers) / float(self.num_good_matches)
            score = 1.0

            return score
        
        except Exception:
            return 0.0
