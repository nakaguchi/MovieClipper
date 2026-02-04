"""
特徴ベースのテンプレートマッチングモジュール。
ORB特徴点を使用した部分一致検出を提供します。
"""

from typing import List

import cv2
import numpy as np

# 特徴点マッチング用のリサイズサイズ
FM_SIZE = (320, 240)


def match_template_by_features(
    ref_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    kp_ref: List[cv2.KeyPoint],
    desc_ref: np.ndarray,
    detector: object,
    feature_threshold: float = 0.7,
    visualize: bool = False,
    angle_tol: float = 15.0,
    min_good_matches: int = 4,
) -> float:
    """
    特徴点マッチングで参照フレームが入力フレーム内の一部に含まれるかを判定する。
    ORB 特徴点を使用し、ホモグラフィ行列の確立度をスコアとして返す。
    
    Args:
        ref_bgr: 参照フレーム（BGR）
        frame_bgr: 入力フレーム（BGR）
        kp_ref: 参照フレームのキーポイント
        desc_ref: 参照フレームの記述子
        detector: ORB検出器インスタンス
        feature_threshold: 特徴点マッチングの信頼度閾値 (デフォルト 0.7)
        visualize: マッチング画像を表示するかどうか (デフォルト False)
        angle_tol: 角度許容度（度数法、デフォルト 15.0）
        min_good_matches: 十分なマッチと判定する最小マッチ数 (デフォルト 4)
    
    Returns:
        マッチスコア（0.0～1.0）。高いほどマッチしている。
    """
    try:
        # フレーム側をグレースケールにして切り出す（参照記述子は事前計算済み）
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.resize(frame_gray, FM_SIZE, interpolation=cv2.INTER_AREA)  # crop to match ref frame size

        # 受け取った ORB 実装を使ってフレーム側のキーポイント・記述子を計算
        kp_frame, desc_frame = detector.detectAndCompute(frame_gray, None)
        
        # マッチング対象がない場合
        if desc_frame is None or len(kp_ref) < min_good_matches or len(kp_frame) < min_good_matches:
            return 0.0
        
        # ORB 用の BFMatcher（ハミング距離）を使用
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # knn マッチング（各特徴点に対して最も近い2つのマッチを取得）
        matches = matcher.knnMatch(desc_ref, desc_frame, k=2)
        
        # Lowe's ratio test で良いマッチのみ選別
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < feature_threshold * n.distance:
                    good_matches.append(m)
        
        # マッチ後に角度差でフィルタ（角度依存マッチング）
        if len(good_matches) > 0 and angle_tol is not None and angle_tol >= 0.0:
            filtered = []
            for m in good_matches:
                ang_ref = kp_ref[m.queryIdx].angle if kp_ref and m.queryIdx < len(kp_ref) else 0.0
                ang_fr = kp_frame[m.trainIdx].angle if kp_frame and m.trainIdx < len(kp_frame) else 0.0
                diff = abs(((ang_ref - ang_fr + 180.0) % 360.0) - 180.0)
                if diff <= angle_tol:
                    filtered.append(m)
            good_matches = filtered
        
        # 十分なマッチが見つからない場合
        if len(good_matches) < min_good_matches:
            return 0.0
        
        # マッチした特徴点を抽出
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # ホモグラフィ行列を計算（RANSACを使用）
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return 0.0
        
        # RANSAC でのインライアの比率をスコアとする
        inliers = np.sum(mask)
        score = float(inliers) / float(len(good_matches))

        # 可視化（マッチ表示）
        if visualize:
            try:
                ref_disp = cv2.resize(ref_bgr, FM_SIZE, interpolation=cv2.INTER_AREA)
                for kp in kp_ref:
                    cv2.circle(ref_disp, (int(kp.pt[0]), int(kp.pt[1])), 2, (0, 255, 0), -1)
                frame_disp = cv2.resize(frame_bgr, FM_SIZE, interpolation=cv2.INTER_AREA)
                for kp in kp_frame:
                    cv2.circle(frame_disp, (int(kp.pt[0]), int(kp.pt[1])), 2, (0, 255, 0), -1)
                img_matches = cv2.drawMatches(ref_disp, kp_ref, frame_disp, kp_frame, good_matches, None, flags=2)
                txt = f"score={score:.3f} inliers={int(inliers)}/{len(good_matches)}"
                cv2.putText(img_matches, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("feature_matches", img_matches)
                cv2.waitKey(1)
            except Exception:
                pass
        
        return score
        
    except Exception as e:
        # エラーが発生した場合は 0.0 を返す
        return 0.0
