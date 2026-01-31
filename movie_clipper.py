from __future__ import annotations

import argparse
import csv
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict


import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Version
__version__ = "1.3"
FM_SIZE = (640, 480)  # 特徴点マッチング用のリサイズサイズ

# -----------------------------
# small helpers
# -----------------------------
def replace_suffix(p: Path, new_suffix: str) -> Path:
    if p.suffix:
        return p.with_suffix(new_suffix)
    return Path(str(p) + new_suffix)


def bitcount(x: int) -> int:
    if hasattr(int, "bit_count"):
        return x.bit_count()
    return bin(x).count("1")


def find_executable(name: str) -> Optional[str]:
    path = shutil.which(name)
    if path:
        return path
    exe_dir = os.path.dirname(sys.executable)
    ext = ".exe" if platform.system() == "Windows" else ""
    candidate = os.path.join(exe_dir, name + ext)
    if os.path.isfile(candidate):
        return candidate
    return None


def run_cmd(cmd: List[str]) -> Tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def render_progress(current: float, total: Optional[float], width: int = 30) -> str:
    if total and total > 0:
        frac = min(max(current / total, 0.0), 1.0)
        filled = int(round(frac * width))
        bar = "#" * filled + "-" * (width - filled)
        return f"[{bar}] {frac*100:6.2f}%"
    else:
        # total unknown
        filled = int(current) % (width + 1)
        bar = "#" * filled + "-" * (width - filled)
        return f"[{bar}]"


# -----------------------------
# pHash (64-bit) and Hamming distance
# -----------------------------
def phash_64(gray_bgr_or_gray: np.ndarray) -> int:
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
    return bitcount(a ^ b)


# -----------------------------
# SSIM preprocessing
# -----------------------------
def preprocess_for_ssim(frame_bgr: np.ndarray, size: int = 256) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return gray


def compute_ssim(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    return float(ssim(a_gray, b_gray, data_range=255))


# -----------------------------
# 特徴点マッチング（部分一致検出用）
# -----------------------------
def match_template_by_features(
    ref_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    kp_ref: List[cv2.KeyPoint],
    desc_ref: np.ndarray,
    detector: object,
    feature_threshold: float = 0.7,
    visualize: bool = False,
    angle_tol: float = 15.0,
) -> float:
    """
    特徴点マッチングで参照フレームが入力フレーム内の一部に含まれるかを判定する。
    ORB 特徴点を使用し、ホモグラフィ行列の確立度をスコアとして返す。
    
    Args:
        ref_bgr: 参照フレーム（BGR）
        frame_bgr: 入力フレーム（BGR）
        feature_threshold: 特徴点マッチングの信頼度閾値
    
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
        if desc_frame is None or len(kp_ref) < 4 or len(kp_frame) < 4:
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
        if len(good_matches) < 4:
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


# -----------------------------
# ffprobe helpers
# -----------------------------
def probe_fps_ffprobe(input_path: str) -> Optional[float]:
    exec = find_executable("ffprobe")
    if not exec:
        return None
    cmd = [
        exec,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate",
        "-of", "default=nokey=1:noprint_wrappers=1",
        input_path,
    ]
    rc, out = run_cmd(cmd)
    if rc != 0:
        return None

    for line in out.strip().splitlines():
        if "/" in line:
            num, den = line.split("/", 1)
            try:
                numf = float(num)
                denf = float(den)
                if denf != 0:
                    fps = numf / denf
                    if fps > 1e-6:
                        return fps
            except ValueError:
                pass
    return None


def probe_duration_ffprobe(input_path: str) -> Optional[float]:
    exec = find_executable("ffprobe")
    if not exec:
        return None
    cmd = [
        exec,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        input_path,
    ]
    rc, out = run_cmd(cmd)
    if rc != 0:
        return None
    try:
        dur = float(out.strip())
        if dur > 0:
            return dur
    except ValueError:
        return None
    return None


def has_audio_stream(input_path: str) -> bool:
    exec = find_executable("ffprobe")
    if not exec:
        return False
    cmd = [
        exec,
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        input_path,
    ]
    rc, out = run_cmd(cmd)
    return (rc == 0) and (len(out.strip()) > 0)


# -----------------------------
# Segment logic
# -----------------------------
@dataclass
class Segment:
    start: float
    end: float


def merge_segments(segs: List[Segment], max_gap_sec: float) -> List[Segment]:
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: s.start)
    merged = [segs[0]]
    for s in segs[1:]:
        last = merged[-1]
        if s.start - last.end <= max_gap_sec:
            last.end = max(last.end, s.end)
        else:
            merged.append(s)
    return merged


def write_segments_csv(csv_path: Path, segs: List[Segment], fps: float, params: dict):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["segment_index", "start_sec", "end_sec", "duration_sec", "fps"] +
                   [f"param:{k}" for k in params.keys()])
        for i, s in enumerate(segs):
            w.writerow([i, f"{s.start:.6f}", f"{s.end:.6f}", f"{(s.end - s.start):.6f}", f"{fps:.6f}"] +
                       list(params.values()))


def analyze_segments(
    input_path: str,
    ref_image_path: str,
    fps_override: float,
    phash_maxdist: int,
    ssim_size: int,
    ssim_enter: float,
    ssim_exit: float,
    smooth_sec: float,
    preroll_sec: float,
    postroll_sec: float,
    min_segment_sec: float,
    max_gap_sec: float,
    frame_csv_path: Optional[Path],
    partial_match: bool = False,
    feature_threshold: float = 0.7,
    frame_skip: int = 0,
    visual: bool = False,
    progress_interval_sec: float = 0.2,
) -> Tuple[List[Segment], float]:
    ref_bgr = cv2.imread(ref_image_path, cv2.IMREAD_COLOR)
    if ref_bgr is None:
        raise FileNotFoundError(f"参照画像を読み込めません: {ref_image_path}")

    ref_hash = phash_64(ref_bgr)
    ref_ssim_gray = preprocess_for_ssim(ref_bgr, size=ssim_size)

    # 部分一致モード向け: 参照フレームの特徴量は最初に一度だけ計算する
    detector = None
    kp_ref = None
    desc_ref = None
    if partial_match:
        detector = cv2.ORB_create(nfeatures=1500)
        ref_gray_feat = cv2.resize(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY), FM_SIZE, interpolation=cv2.INTER_AREA)
        # 通常の detectAndCompute を使い、記述子は回転補正ありで計算する
        kp_ref, desc_ref = detector.detectAndCompute(ref_gray_feat, None)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"入力動画を開けません: {input_path}")

    # fps
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps_override > 0:
        fps = fps_override
    if fps <= 1e-6:
        probed = probe_fps_ffprobe(input_path)
        fps = probed if (probed and probed > 1e-6) else 30.0

    # total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        dur = probe_duration_ffprobe(input_path)
        total_frames = int(round(dur * fps)) if dur is not None else 0

    preroll_frames = max(0, int(round(preroll_sec * fps)))
    postroll_frames = max(0, int(round(postroll_sec * fps)))

    # EMA
    window = max(1, int(round(smooth_sec * fps)))
    alpha = 2.0 / (window + 1.0)

    ema = 0.0
    in_segment = False
    seg_start_frame = 0
    segs: List[Segment] = []

    frame_idx = 0
    last_frame_idx = -1

    # optional frame CSV
    fcsv = None
    writer = None
    if frame_csv_path is not None:
        fcsv = frame_csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(fcsv)
        if partial_match:
            writer.writerow([
                "frame_idx",
                "time_sec",
                "feature_match_score",
                "smoothed_score",
                "in_segment",
            ])
        else:
            writer.writerow([
                "frame_idx",
                "time_sec",
                "phash_hamming_dist",
                "ssim_score",
                "smoothed_score",
                "in_segment",
            ])

    last_progress_t = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            last_frame_idx = frame_idx

            # pHash
            h = phash_64(frame)
            hd = hamming_distance_64(h, ref_hash)
            phash_pass = (hd <= phash_maxdist)

            # SSIM or Feature Matching
            ssim_score: Optional[float] = None
            feature_match_score: Optional[float] = None
            score_for_smoothing = 0.0
            
            if partial_match:
                # 部分一致モード：特徴点マッチングを使用
                feature_match_score = match_template_by_features(
                    ref_bgr,
                    frame,
                    kp_ref,
                    desc_ref,
                    detector,
                    feature_threshold=feature_threshold,
                    visualize=visual,
                )
                score_for_smoothing = feature_match_score
            else:
                # 通常モード：pHash + SSIM を使用
                if phash_pass:
                    g = preprocess_for_ssim(frame, size=ssim_size)
                    ssim_score = compute_ssim(g, ref_ssim_gray)
                    score_for_smoothing = ssim_score
                    if visual:
                        try:
                            ref_disp = cv2.resize(ref_ssim_gray, (ssim_size, ssim_size))
                            cur_disp = cv2.resize(g, (ssim_size, ssim_size))
                            disp = np.stack([cur_disp, cur_disp, cur_disp], axis=2)
                            ref_color = np.stack([ref_disp, ref_disp, ref_disp], axis=2)
                            side = np.hstack([ref_color, disp])
                            cv2.putText(side, f"SSIM={ssim_score:.3f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.imshow("ssim", side)
                            cv2.waitKey(1)
                        except Exception:
                            pass

            # EMA smoothing
            ema = alpha * score_for_smoothing + (1.0 - alpha) * ema

            # hysteresis
            if (not in_segment) and (ema >= ssim_enter):
                in_segment = True
                seg_start_frame = max(0, frame_idx - preroll_frames)
            elif in_segment and (ema <= ssim_exit):
                in_segment = False
                seg_end_frame = frame_idx + postroll_frames
                start = seg_start_frame / fps
                end = seg_end_frame / fps
                if end - start >= min_segment_sec:
                    segs.append(Segment(start=start, end=end))

            if writer is not None:
                tsec = frame_idx / fps
                if partial_match:
                    writer.writerow([
                        frame_idx,
                        f"{tsec:.6f}",
                        "" if feature_match_score is None else f"{feature_match_score:.6f}",
                        f"{ema:.6f}",
                        int(in_segment),
                    ])
                else:
                    writer.writerow([
                        frame_idx,
                        f"{tsec:.6f}",
                        hd,
                        "" if ssim_score is None else f"{ssim_score:.6f}",
                        f"{ema:.6f}",
                        int(in_segment),
                    ])

            frame_idx += 1

            # フレームスキップ: 指定数だけフレームを grab() して読み飛ばす
            if frame_skip and frame_skip > 0:
                for _ in range(frame_skip):
                    ok_skip = cap.grab()
                    if not ok_skip:
                        break
                    frame_idx += 1

            # progress
            now = time.time()
            if (now - last_progress_t) >= progress_interval_sec:
                if total_frames > 0:
                    print("\r" + render_progress(frame_idx, total_frames) + f" (analyze {frame_idx}/{total_frames})",
                          end="", flush=True)
                else:
                    print("\r" + render_progress(frame_idx, None) + f" (analyze {frame_idx})",
                          end="", flush=True)
                last_progress_t = now

        # finalize progress line
        if total_frames > 0:
            print("\r" + render_progress(frame_idx, total_frames) + f" (analyze {frame_idx}/{total_frames})")
        else:
            print("\r" + render_progress(frame_idx, None) + f" (analyze {frame_idx})")

        # trailing segment
        if in_segment and last_frame_idx >= 0:
            seg_end_frame = last_frame_idx + 1
            start = seg_start_frame / fps
            end = seg_end_frame / fps
            if end - start >= min_segment_sec:
                segs.append(Segment(start=start, end=end))

    finally:
        cap.release()
        if fcsv is not None:
            fcsv.close()

    segs = merge_segments(segs, max_gap_sec=max_gap_sec)
    return segs, fps


# -----------------------------
# ffmpeg progress runner
# -----------------------------
def run_ffmpeg_with_progress(cmd: List[str], total_duration_sec: Optional[float], label: str = "encode"):
    """
    Runs ffmpeg with -progress pipe:1 and prints progress bar based on out_time_ms.
    total_duration_sec: expected OUTPUT duration (seconds). If None, progress is non-percentage.
    """
    if "-progress" not in cmd:
        # 念のため
        cmd = cmd[:-1] + ["-progress", "pipe:1", "-nostats"] + cmd[-1:]

    # print(f"Running ffmpeg command: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    last_print_t = time.time()
    out_time_ms = 0
    speed = None
    saw_end = False  # progress=end を見たか

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)

            if k == "out_time_ms":
                try:
                    out_time_ms = int(v)
                except ValueError:
                    pass
            elif k == "speed":
                speed = v
            elif k == "progress" and v == "end":
                saw_end = True

            now = time.time()
            if now - last_print_t >= 0.2:
                cur_sec = out_time_ms / 1_000_000.0
                bar = render_progress(cur_sec, total_duration_sec, width=30)
                if total_duration_sec and total_duration_sec > 0:
                    msg = f"{bar} ({label} {cur_sec:8.2f}/{total_duration_sec:8.2f}s)"
                else:
                    msg = f"{bar} ({label} {cur_sec:8.2f}s)"
                if speed:
                    msg += f" speed={speed}"
                print("\r" + msg, end="", flush=True)
                last_print_t = now

        rc = proc.wait()

        # ---- 最終表示：成功時は必ず 100% に揃える ----
        if rc == 0 and total_duration_sec and total_duration_sec > 0:
            # progress=end を見ていなくても、成功なら100%表示に揃える
            cur_sec = total_duration_sec
        else:
            cur_sec = out_time_ms / 1_000_000.0

        bar = render_progress(cur_sec, total_duration_sec, width=30)
        if total_duration_sec and total_duration_sec > 0:
            msg = f"{bar} ({label} {cur_sec:8.2f}/{total_duration_sec:8.2f}s)"
        else:
            msg = f"{bar} ({label} {cur_sec:8.2f}s)"
        if speed:
            msg += f" speed={speed}"
        print("\r" + msg)

        if rc != 0:
            raise RuntimeError(f"ffmpeg が失敗しました（return code={rc}）")

    finally:
        if proc.stdout:
            proc.stdout.close()


# -----------------------------
# ffmpeg concat with trim/atrim (+ optional deinterlace) + progress
# -----------------------------
def export_with_ffmpeg(
    input_path: str,
    output_path: str,
    segs: List[Segment],
    bitrate: str,
    preset: str,
    audio_bitrate: str,
    deinterlace: bool,
    yadif_args: str,
    total_duration_sec: Optional[float],
):
    exec = find_executable("ffmpeg")
    if not exec:
        raise RuntimeError("ffmpeg が見つかりません。PATH を確認してください。")

    if not segs:
        raise RuntimeError("抽出区間が0件です（閾値が厳しすぎる可能性があります）。")

    audio_present = has_audio_stream(input_path)

    segs = [s for s in segs if (s.end - s.start) > 1e-3]
    if not segs:
        raise RuntimeError("有効な抽出区間がありません（全て end<=start です）。")

    parts: List[str] = []
    n = len(segs)

    for i, s in enumerate(segs):
        v_filters = [f"trim=start={s.start:.6f}:end={s.end:.6f}"]
        if deinterlace:
            v_filters.append(f"yadif={yadif_args}")
        v_filters.append("setpts=PTS-STARTPTS")
        parts.append(f"[0:v]{','.join(v_filters)}[v{i}]")

        if audio_present:
            parts.append(
                f"[0:a]atrim=start={s.start:.6f}:end={s.end:.6f},"
                f"asetpts=PTS-STARTPTS,"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"aresample=48000[a{i}]"
            )

    if audio_present:
        concat_inputs = "".join([f"[v{i}][a{i}]" for i in range(n)])
        parts.append(f"{concat_inputs}concat=n={n}:v=1:a=1[vout][aout]")
    else:
        concat_inputs = "".join([f"[v{i}]" for i in range(n)])
        parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[vout]")

    filter_complex = ";".join(parts)

    # フィルタコンプレックスが長い場合、テンポラリファイルに保存して -filter_complex_script で渡す
    # これによりコマンドライン長制限を回避できる
    use_filter_script = len(filter_complex) > 4000

    filter_script_file = None
    try:
        if use_filter_script:
            # テンポラリファイルにフィルタを書き込む
            fd, filter_script_file = tempfile.mkstemp(suffix=".txt", prefix="ffmpeg_filter_", text=True)
            # print(f"Using temporary filter script file: {filter_script_file}")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(filter_complex)
            except:
                os.close(fd)
                raise

        cmd = [
            exec,
            "-y",
            "-i", input_path,
        ]

        # フィルタをコマンドに追加（方法1: スクリプトファイル、方法2: 直接）
        if use_filter_script:
            cmd += ["-filter_complex_script", filter_script_file]
        else:
            cmd += ["-filter_complex", filter_complex]

        cmd += [
            "-map", "[vout]",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", preset,
            "-b:v", bitrate,
        ]

        if audio_present:
            cmd += [
                "-map", "[aout]",
                "-c:a", "aac",
                "-b:a", audio_bitrate,
            ]
        else:
            cmd += ["-an"]

        cmd += [
            "-movflags", "+faststart",
            "-progress", "pipe:1",
            "-nostats",
            output_path,
        ]

        # 実出力は抽出後の合成なので、total_duration_sec は「入力尺」だと100%表示にズレます。
        # ただしユーザ要件は「バーと%」の表示なので、便宜上 input duration を使用します。
        run_ffmpeg_with_progress(cmd, total_duration_sec=total_duration_sec, label="output")

    finally:
        # テンポラリファイルを削除
        if filter_script_file and os.path.exists(filter_script_file):
            try:
                os.remove(filter_script_file)
            except:
                pass



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    parser.add_argument("input", help="入力動画ファイル")
    parser.add_argument("--ref", default="", help="参照フレーム画像（省略時: 入力拡張子を .jpg に置換）")
    parser.add_argument("--output", default="", help="出力 mp4（省略時: 入力に応じて自動決定）")

    parser.add_argument("--segment_csv", action="store_true", help="抽出区間CSVを出力（<入力名>_seg.csv）")
    parser.add_argument("--frame_csv", action="store_true", help="フレーム単位ログCSVを出力（<入力名>_frm.csv）")

    parser.add_argument("--phash_maxdist", type=int, default=12, help="pHash ハミング距離閾値。大きくすると候補が増える（SSIM計算が増える）。固定カメラなら 8〜14 付近から調整が無難です。規定値 12")
    parser.add_argument("--ssim_size", type=int, default=256, help="SSIM 計算用にリサイズするサイズ（正方形）。大きくすると精度が上がるが計算コストも増える。256〜512 程度が無難です。規定値 256")
    parser.add_argument("--ssim_enter", type=float, default=0.78, help="SSIM しきい値（入る側）。この値以上で抽出区間に入る。規定値 0.78")
    parser.add_argument("--ssim_exit", type=float, default=0.72, help="SSIM しきい値（出る側）。この値以下で抽出区間から出る。規定値 0.72")

    parser.add_argument("--smooth_sec", type=float, default=0.5, help="SSIM スコアの平滑化ウィンドウ時間（秒）。大きくするとノイズに強くなるが応答が遅くなる。規定値 0.5")
    parser.add_argument("--preroll_sec", type=float, default=0.2, help="抽出区間の前に追加する余裕時間（秒）。規定値 0.2")
    parser.add_argument("--postroll_sec", type=float, default=0.2, help="抽出区間の後に追加する余裕時間（秒）。規定値 0.2")
    parser.add_argument("--min_segment_sec", type=float, default=0.3, help="抽出区間の最小長さ（秒）。これ未満の区間は破棄される。規定値 0.3")
    parser.add_argument("--max_gap_sec", type=float, default=0.15, help="抽出区間の結合最大ギャップ時間（秒）。これ以下のギャップは結合される。規定値 0.15")

    parser.add_argument("--fps", type=float, default=0.0, help="入力動画のFPSを強制指定（0.0で自動検出）規定値 0.0")
    parser.add_argument("--bitrate", default="2000k", help="出力動画のビットレート設定（libx264）。例: 5000k, 8000k など。値が大きいほど高品質・大容量。一般的には3000k〜10000k程度が無難です。規定値 2000k")
    parser.add_argument("--preset", default="medium", help="出力動画のエンコードプリセット（libx264）。品質には影響しないが、速度と圧縮率に影響する。ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow から選択。規定値 medium")
    parser.add_argument("--audio_bitrate", default="192k", help="出力音声ビットレート（AAC）。規定値 192k")

    parser.add_argument("--no_deinterlace", action="store_true", help="デインターレース処理を無効化（デフォルトでは有効）")
    parser.add_argument("--yadif_args", default="mode=send_frame:parity=auto:deint=all", help="yadif フィルタの引数（デインターレース有効時）規定値 mode=send_frame:parity=auto:deint=all")

    parser.add_argument("--progress_interval", type=float, default=0.2, help="解析進捗表示の間隔（秒）")

    parser.add_argument("--visual", action="store_true", help="一致計算中の処理画像を表示する（デバッグ用）")

    parser.add_argument("--frame_skip", type=int, default=0, help="各処理フレームの後にスキップするフレーム数（0で無効）。例: 2 は各処理後に2フレームをスキップ）")

    parser.add_argument("--partial_match", action="store_true", help="部分一致モード: 参照フレームが入力フレーム内の一部に含まれる場合を検出（特徴点マッチング使用）")
    parser.add_argument("--feature_threshold", type=float, default=0.7, help="特徴点マッチングの信頼度閾値（0.0～1.0）。小さくすると検出が容易になる。規定値 0.7")

    args = parser.parse_args()

    deinterlace = not args.no_deinterlace

    in_path = Path(args.input)

    # defaults derived from input
    ref_path = Path(args.ref) if args.ref else replace_suffix(in_path, ".jpg")

    # Output default with condition:
    # - if input is mp4 => <no suffix>_out.mp4
    # - else => replace suffix to .mp4
    if args.output:
        out_path = Path(args.output)
    else:
        if in_path.suffix.lower() == ".mp4":
            out_path = Path(str(in_path.with_suffix("")) + "_out.mp4")
        else:
            out_path = replace_suffix(in_path, ".mp4")

    # CSV paths = remove suffix then add _seg.csv / _frm.csv
    stem_no_suffix = in_path.with_suffix("")
    seg_csv_path = (Path(str(stem_no_suffix) + "_seg.csv")) if args.segment_csv else None
    frm_csv_path = (Path(str(stem_no_suffix) + "_frm.csv")) if args.frame_csv else None

    if not ref_path.exists():
        raise FileNotFoundError(f"参照画像が見つかりません: {ref_path}（--ref で指定可能）")

    for exec_name in ["ffmpeg", "ffprobe"]:
        if not find_executable(exec_name):
            raise RuntimeError(f"{exec_name} が見つかりません。PATH を確認してください。")

    input_duration = probe_duration_ffprobe(str(in_path))  # for ffmpeg progress denominator

    print(f"Input : {in_path}")
    print(f"Ref   : {ref_path}")
    print(f"Output: {out_path}")
    if seg_csv_path:
        print(f"SegCSV: {seg_csv_path}")
    if frm_csv_path:
        print(f"FrmCSV: {frm_csv_path}")
    if input_duration:
        print(f"Duration (ffprobe): {input_duration:.2f}s")
    if args.partial_match:
        print(f"Mode: PARTIAL MATCH (feature-based detection)")

    segs, fps = analyze_segments(
        input_path=str(in_path),
        ref_image_path=str(ref_path),
        fps_override=args.fps,
        phash_maxdist=args.phash_maxdist,
        ssim_size=args.ssim_size,
        ssim_enter=args.ssim_enter,
        ssim_exit=args.ssim_exit,
        smooth_sec=args.smooth_sec,
        preroll_sec=args.preroll_sec,
        postroll_sec=args.postroll_sec,
        min_segment_sec=args.min_segment_sec,
        max_gap_sec=args.max_gap_sec,
        frame_csv_path=frm_csv_path,
        partial_match=args.partial_match,
        feature_threshold=args.feature_threshold,
        frame_skip=args.frame_skip,
        visual=args.visual,
        progress_interval_sec=args.progress_interval,
    )

    print(f"Detected segments: {len(segs)} (fps={fps:.3f})")

    if seg_csv_path is not None:
        params = {
            "partial_match": args.partial_match,
            "feature_threshold": args.feature_threshold if args.partial_match else "",
            "phash_maxdist": args.phash_maxdist if not args.partial_match else "",
            "ssim_enter": args.ssim_enter,
            "ssim_exit": args.ssim_exit,
            "smooth_sec": args.smooth_sec,
            "preroll_sec": args.preroll_sec,
            "postroll_sec": args.postroll_sec,
            "min_segment_sec": args.min_segment_sec,
            "max_gap_sec": args.max_gap_sec,
            "deinterlace": deinterlace,
            "yadif_args": args.yadif_args if deinterlace else "",
            "bitrate": args.bitrate,
            "preset": args.preset,
            "audio_bitrate": args.audio_bitrate,
        }
        write_segments_csv(seg_csv_path, segs, fps, params)
        print(f"Segments CSV written: {seg_csv_path}")

    expected_out_duration = sum(max(0.0, s.end - s.start) for s in segs)  # ←追加
    print(f"Expected output duration: {expected_out_duration:.2f}s")

    export_with_ffmpeg(
        input_path=str(in_path),
        output_path=str(out_path),
        segs=segs,
        bitrate=args.bitrate,
        preset=args.preset,
        audio_bitrate=args.audio_bitrate,
        deinterlace=deinterlace,
        yadif_args=args.yadif_args,
        total_duration_sec=expected_out_duration,  # ← input_duration ではなくこちら
    )

    if frm_csv_path is not None:
        print(f"Frame CSV written: {frm_csv_path}")

    print(f"Done: {out_path}")


if __name__ == "__main__":
    print(f"MovieClipper v{__version__} - Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    print(f"Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
