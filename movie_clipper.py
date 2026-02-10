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

from matcher_base import Matcher
from orb_matcher import ORB_Matcher
from ssim_matcher import SSIM_Matcher
from phash_matcher import pHash_Matcher

# Version
__version__ = "1.3"

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
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', text=True)
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


# Note: pHash, Hamming distance, and SSIM functions have been refactored
# into dedicated matcher classes (phash_matcher.py, ssim_matcher.py).
# Original placeholder functions can be removed once all code is transitioned to use Matcher classes.


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


# -----------------------------
# Segment processing
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
    phash_ubound: int,
    phash_threshold: int,
    match_size: int,
    match_enter: float,
    match_leave: float,
    smooth_sec: float,
    preroll_sec: float,
    postroll_sec: float,
    min_segment_sec: float,
    max_gap_sec: float,
    frame_csv_path: Optional[Path],
    matcher_type: str = "ssim",
    feature_threshold: float = 0.7,
    frame_skip: int = 0,
    visualize: bool = False,
    min_good_matches: int = 4,
    progress_interval_sec: float = 0.5,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> Tuple[List[Segment], float]:
    ref_bgr = cv2.imread(ref_image_path, cv2.IMREAD_COLOR)
    if ref_bgr is None:
        raise FileNotFoundError(f"参照画像を読み込めません: {ref_image_path}")

    # マッチャーを選択して初期化
    matcher: Matcher
    matcher_type = matcher_type.lower()
    if matcher_type == "orb":
        # ORB特徴点マッチング
        matcher = ORB_Matcher(
            nfeatures=1500,
            feature_threshold=feature_threshold,
            angle_tol=15.0,
            min_good_matches=min_good_matches,
            match_size=match_size,
            visualize=visualize,
        )
    elif matcher_type == "phash":
        # pHash マッチング
        matcher = pHash_Matcher(
            hamming_dist_ubound=phash_ubound,
            visualize=visualize,
        )
    else:  # "ssim" or default
        # pHash + SSIM マッチング
        matcher = SSIM_Matcher(
            hd_threshold=phash_threshold,
            ssim_size=match_size,
        )
    
    matcher.set_reference(ref_bgr)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"入力動画を開けません: {input_path}")

    # fps
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6:
        probed = probe_fps_ffprobe(input_path)
        fps = probed if (probed and probed > 1e-6) else 30.0

    # total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        dur = probe_duration_ffprobe(input_path)
        total_frames = int(round(dur * fps)) if dur is not None else 0

    # determine start/end frames (optional)
    start_frame = 0
    end_frame = (total_frames - 1) if total_frames > 0 else None
    if start_sec is not None and start_sec > 0:
        start_frame = max(0, int(round(start_sec * fps)))
    if end_sec is not None and end_sec > 0:
        ef = int(round(end_sec * fps))
        if end_frame is None:
            end_frame = ef
        else:
            end_frame = min(end_frame, ef)

    # seek to start frame if needed
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    preroll_frames = max(0, int(round(preroll_sec * fps)))
    postroll_frames = max(0, int(round(postroll_sec * fps)))

    # EMA
    window = max(1, int(round(smooth_sec * fps)))
    alpha = 2.0 / (window + 1.0)

    ema = 0.0
    in_segment = False
    seg_start_frame = 0
    segs: List[Segment] = []

    # optional frame CSV
    fcsv = None
    writer = None
    if frame_csv_path is not None:
        fcsv = frame_csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(fcsv)
        writer.writerow([
            "frame_idx",
            "time_sec",
            "num_matches" if matcher_type == "orb" else "pHash_hd",
            matcher_type+ "_score",
            "smoothed_score",
            "in_segment",
        ])

    frame_idx = start_frame
    last_frame_idx = -1
    proc_count = 0
    start_time = time.time()
    last_progress_t = time.time()
    
    # progress total uses the clipped range when available
    total_for_progress: Optional[int]
    if (end_frame is not None) and (end_frame >= start_frame):
        total_for_progress = end_frame - start_frame + 1
    else:
        total_for_progress = total_frames if total_frames > 0 else None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            last_frame_idx = frame_idx

            # 新しいマッチャークラスを使用して一致度を計算
            score_for_smoothing = matcher.compute_similarity(frame)

            # EMA smoothing
            ema = alpha * score_for_smoothing + (1.0 - alpha) * ema

            # hysteresis
            if (not in_segment) and (ema >= match_enter):
                in_segment = True
                seg_start_frame = max(0, frame_idx - preroll_frames)
            elif in_segment and (ema <= match_leave):
                in_segment = False
                seg_end_frame = frame_idx + postroll_frames
                start = seg_start_frame / fps
                end = seg_end_frame / fps
                if end - start >= min_segment_sec:
                    segs.append(Segment(start=start, end=end))

            if writer is not None:
                tsec = frame_idx / fps
                writer.writerow([
                    frame_idx,
                    f"{tsec:.6f}",
                    f"{matcher.num_good_matches if hasattr(matcher, 'num_good_matches') else matcher.hamming_distance}",
                    f"{score_for_smoothing:.6f}",
                    f"{ema:.6f}",
                    int(in_segment),
                ])

            # check if we've reached the end frame
            if (end_frame is not None) and (frame_idx >= end_frame):
                break

            frame_idx += 1
            proc_count += 1

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
                if total_for_progress and total_for_progress > 0:
                    processed = frame_idx - start_frame
                    print("\r" + render_progress(processed, total_for_progress) + \
                            f" (analyze {processed}/{total_for_progress})", end="", flush=True)
                else:
                    print("\r" + render_progress(frame_idx, None) + f" (analyze {frame_idx})",
                            end="", flush=True)
                last_progress_t = now

        # finalize progress line
        if total_for_progress and total_for_progress > 0:
            processed = frame_idx - start_frame
            print("\r" + render_progress(processed, total_for_progress) \
                    + f" (analyze {processed}/{total_for_progress})")
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

    elapsed = time.time() - start_time
    print(f"Matching done: {proc_count} frames, elapsed time {elapsed:.2f} s, {proc_count/elapsed:.2f} fps")

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
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", text=True, bufsize=1)

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
    if not segs:
        raise RuntimeError("抽出区間が0件です（閾値が厳しすぎる可能性があります）。")
    segs = [s for s in segs if (s.end - s.start) > 1e-3]
    if not segs:
        raise RuntimeError("有効な抽出区間がありません（全て end<=start です）。")

    audio_present = has_audio_stream(input_path)
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
    # use_filter_script = len(filter_complex) > 4000
    filter_script_file = None
    try:
        # if use_filter_script:
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
        cmd += ["-filter_complex_script", filter_script_file]
        # if use_filter_script:
        #     cmd += ["-filter_complex_script", filter_script_file]
        # else:
        #     cmd += ["-filter_complex", filter_complex]

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
    # コマンドライン引数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    # 入力/出力設定
    parser.add_argument("input", help="入力動画ファイル")
    parser.add_argument("--ref_file", default="", help="参照フレーム画像（省略時: 入力拡張子を .jpg に置換）")
    parser.add_argument("--output", default="", help="出力 mp4（省略時: 入力に応じて自動決定）")
    parser.add_argument("--start", type=float, default=None, help="処理開始時間（秒）。省略または0で先頭から）")
    parser.add_argument("--end", type=float, default=None, help="処理終了時間（秒）。省略で最後まで処理）")
    parser.add_argument("--frame_skip", type=int, default=1, help="各処理フレームの後にスキップするフレーム数（0で無効）。例: 2 は各処理後に2フレームをスキップ）")
    parser.add_argument("--segment_csv", action="store_true", help="抽出区間CSVを出力（<入力名>_seg.csv）")
    parser.add_argument("--frame_csv", action="store_true", help="フレーム単位ログCSVを出力（<入力名>_frm.csv）")
    # マッチング設定
    parser.add_argument("--matcher", type=str, default="ssim", choices=["phash", "ssim", "orb"], help="マッチング手法: phash (pHashのみ), ssim (pHash+SSIM), orb (ORB特徴点) [既定値: ssim]")
    parser.add_argument("--match_size", type=int, default=256, help="一致度計算にリサイズするサイズ（正方形）。大きくすると精度が上がるが計算コストも増える。256〜512 程度が無難です。規定値 256")
    parser.add_argument("--match_enter", type=float, default=0.78, help="マッチ進入閾値（入る側）。この値以上で抽出区間に入る。規定値 0.78")
    parser.add_argument("--match_leave", type=float, default=0.72, help="マッチ退出閾値（出る側）。この値以下で抽出区間から出る。規定値 0.72")
    # マッチャー固有設定
    parser.add_argument("--phash_ubound", type=int, default=50, help="pHash識別用 ハミング距離閾値の上限。規定値 50")
    parser.add_argument("--phash_threshold", type=int, default=12, help="SSIM識別用 pHash ハミング距離閾値。大きくすると候補が増える（SSIM計算が増える）。固定カメラなら 8〜14 付近から調整が無難です。規定値 12")
    parser.add_argument("--feature_threshold", type=float, default=0.8, help="ORB特徴点マッチングの信頼度閾値（0.0～1.0）。小さくすると検出が容易になる。規定値 0.8")
    parser.add_argument("--min_good_matches", type=int, default=4, help="ORB特徴点マッチングのホモグラフィ計算に必要な最小マッチ数。4以上である必要があります。規定値 4")
    parser.add_argument("--visualize", action="store_true", help="一致計算中の処理画像を表示する（デバッグ用）")
    # 後処理設定
    parser.add_argument("--smooth_sec", type=float, default=0.5, help="一致度の平滑化ウィンドウ時間（秒）。大きくするとノイズに強くなるが応答が遅くなる。規定値 0.5")
    parser.add_argument("--preroll_sec", type=float, default=0.2, help="抽出区間の前に追加する余裕時間（秒）。規定値 0.2")
    parser.add_argument("--postroll_sec", type=float, default=0.2, help="抽出区間の後に追加する余裕時間（秒）。規定値 0.2")
    parser.add_argument("--min_segment_sec", type=float, default=0.3, help="抽出区間の最小長さ（秒）。これ未満の区間は破棄される。規定値 0.3")
    parser.add_argument("--max_gap_sec", type=float, default=0.15, help="抽出区間の結合最大ギャップ時間（秒）。これ以下のギャップは結合される。規定値 0.15")
    # 出力設定
    parser.add_argument("--bitrate", default="2000k", help="出力動画のビットレート設定（libx264）。例: 5000k, 8000k など。値が大きいほど高品質・大容量。一般的には3000k〜10000k程度が無難です。規定値 2000k")
    parser.add_argument("--preset", default="medium", help="出力動画のエンコードプリセット（libx264）。品質には影響しないが、速度と圧縮率に影響する。ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow から選択。規定値 medium")
    parser.add_argument("--audio_bitrate", default="192k", help="出力音声ビットレート（AAC）。規定値 192k")
    parser.add_argument("--no_deinterlace", action="store_true", help="デインターレース処理を無効化（デフォルトでは有効）")
    parser.add_argument("--yadif_args", default="mode=send_frame:parity=auto:deint=all", help="yadif フィルタの引数（デインターレース有効時）規定値 mode=send_frame:parity=auto:deint=all")
    args = parser.parse_args()

    # 入力/出力パス設定
    in_path = Path(args.input)
    ref_path = Path(args.ref_file) if args.ref_file else replace_suffix(in_path, ".jpg")
    if not ref_path.exists():
        raise FileNotFoundError(f"参照画像が見つかりません: {ref_path}（--ref_file で指定可能）")
    if args.output:
        out_path = Path(args.output)
    else:
        if in_path.suffix.lower() == ".mp4":
            out_path = Path(str(in_path.with_suffix("")) + "_out.mp4")
        else:
            out_path = replace_suffix(in_path, ".mp4")
    for exec_name in ["ffmpeg", "ffprobe"]:
        if not find_executable(exec_name):
            raise RuntimeError(f"{exec_name} が見つかりません。PATH を確認してください。")

    # CSV paths = remove suffix then add _seg.csv / _frm.csv
    stem_no_suffix = out_path.with_suffix("")
    seg_csv_path = (Path(str(stem_no_suffix) + "_seg.csv")) if args.segment_csv else None
    frm_csv_path = (Path(str(stem_no_suffix) + "_frm.csv")) if args.frame_csv else None

    # 処理条件の表示
    print(f"Input : {in_path}")
    print(f"Ref   : {ref_path}")
    print(f"Output: {out_path}")
    if seg_csv_path:
        print(f"SegCSV: {seg_csv_path}")
    if frm_csv_path:
        print(f"FrmCSV: {frm_csv_path}")
    input_duration = probe_duration_ffprobe(str(in_path))  # for ffmpeg progress denominator
    if input_duration:
        print(f"Duration (ffprobe): {input_duration:.2f}s")
    print(f"Matcher: {args.matcher.upper()}")

    segs, fps = analyze_segments(
        input_path=str(in_path),
        ref_image_path=str(ref_path),
        frame_csv_path=frm_csv_path,
        frame_skip=args.frame_skip,
        matcher_type=args.matcher,
        start_sec=args.start,
        end_sec=args.end,
        phash_ubound=args.phash_ubound,
        phash_threshold=args.phash_threshold,
        match_size=args.match_size,
        match_enter=args.match_enter,
        match_leave=args.match_leave,
        smooth_sec=args.smooth_sec,
        preroll_sec=args.preroll_sec,
        postroll_sec=args.postroll_sec,
        min_segment_sec=args.min_segment_sec,
        max_gap_sec=args.max_gap_sec,
        feature_threshold=args.feature_threshold,
        visualize=args.visualize,
        min_good_matches=args.min_good_matches,
    )

    print(f"Detected segments: {len(segs)} (fps={fps:.3f})")

    if seg_csv_path is not None:
        params = {
            "matcher": args.matcher,
            "feature_threshold": args.feature_threshold if args.matcher == "orb" else "",
            "phash_maxdist": args.phash_maxdist if args.matcher != "orb" else "",
            "match_enter": args.match_enter,
            "match_leave": args.match_leave,
            "smooth_sec": args.smooth_sec,
            "preroll_sec": args.preroll_sec,
            "postroll_sec": args.postroll_sec,
            "min_segment_sec": args.min_segment_sec,
            "max_gap_sec": args.max_gap_sec,
            "deinterlace": not args.no_deinterlace,
            "yadif_args": args.yadif_args if not args.no_deinterlace else "",
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
        deinterlace=not args.no_deinterlace,
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
