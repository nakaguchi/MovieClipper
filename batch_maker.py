"""
MovieClipper GUI - TkEasyGUI版
動画ファイルの読み込み、再生、シークバー操作が可能なGUI
"""
import os
import io
import json
import subprocess
import TkEasyGUI as eg
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
from pathlib import Path


class MovieClipperGUI:
    """動画編集GUI"""
    
    def __init__(self):
        """初期化"""
        
        # 変数初期化
        self.preview_size = (640, 360)
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.frame_width = 0
        self.frame_height = 0
        self.current_frame = 0
        self.start_frame = None  # 選択範囲の開始フレーム
        self.end_frame = None    # 選択範囲の終了フレーム
        
        # ウィンドウレイアウト定義
        self.layout = [
            [
                eg.Button("ファイルを開く", size=(12, 1)),
                eg.Text("未読み込み", key="-FILE_STATUS-", expand_x=True),
            ],
            [
                eg.Text("フレーム情報:", font=("Arial", 10, "bold")),
                eg.Text("", key="-FRAME_INFO-", expand_x=True),
            ],
            [
                eg.Image(key="-IMAGE-", size=self.preview_size),
            ],
            [
                eg.Slider(
                    range=(0, 100),
                    default_value=0,
                    orientation="h",
                    expand_x=True,
                    # size=(60, 2),
                    key="-SLIDER-",
                    enable_events=True,
                ),
            ],
            [
                eg.Button("開始点を設定", size=(15, 1)),
                eg.Button("終了点を設定", size=(15, 1)),
                eg.Button("選択をクリア", size=(15, 1)),
                eg.Text("", key="-TIME_INFO-", size=(30, 1)),
            ],
            [
                eg.Text("選択範囲:", font=("Arial", 10, "bold")),
                eg.Text("未設定", key="-SELECTION_INFO-", expand_x=True),
            ],
            [
                eg.Text("出力ファイル名:", size=(12, 1)),
                eg.Input("AO2026_", key="-SAVE_NAME-", size=(40, 1), expand_x=True),
                eg.Button("バッチ保存", size=(15, 1), background_color="lightblue"),
            ],
        ]
        
        self.window = eg.Window("MovieClipper GUI", self.layout, finalize=True)
        self.image_element = self.window["-IMAGE-"]
    
    def get_video_info_ffprobe(self, filepath):
        """ffprobeを使用して正確な動画情報を取得"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,nb_read_packets,duration',
                '-of', 'json',
                filepath
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if data.get('streams'):
                    stream = data['streams'][0]
                    width = stream.get('width', 0)
                    height = stream.get('height', 0)
                    
                    # フレームレートを取得
                    fps = 30  # デフォルト値
                    if 'r_frame_rate' in stream:
                        parts = stream['r_frame_rate'].split('/')
                        if len(parts) == 2:
                            fps = float(parts[0]) / float(parts[1])
                    
                    # 総フレーム数を計算
                    duration = stream.get('duration', 0)
                    if duration:
                        duration = float(duration)
                        total_frames = int(duration * fps)
                    else:
                        total_frames = 0
                    
                    return width, height, fps, total_frames
        except Exception as e:
            print(f"ffprobe error: {e}")
        
        return None
        
    def open_video(self, filepath):
        """動画ファイルを開く"""
        if self.cap:
            self.cap.release()
        
        self.video_path = filepath
        
        # ffprobeで動画情報を取得
        video_info = self.get_video_info_ffprobe(filepath)
        
        if video_info is None:
            eg.popup_error("エラー", f"動画ファイルをffprobeで読み込みできません: {filepath}")
            self.video_path = None
            return False
        
        self.frame_width, self.frame_height, self.fps, self.total_frames = video_info
        
        # OpenCVで動画をオープン
        self.cap = cv2.VideoCapture(filepath)
        
        if not self.cap.isOpened():
            eg.popup_error("エラー", f"動画ファイルを開けません: {filepath}")
            self.video_path = None
            return False
        
        # 動画情報を設定
        self.current_frame = 0
        self.start_frame = 0
        self.end_frame = self.total_frames - 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        
        # ファイル名を表示
        filename = Path(filepath).name
        self.window["-FILE_STATUS-"].update(f"読み込み完了: {filename}")
        
        # スライダーの最大値を設定
        self.window["-SLIDER-"].update(range=(0, self.total_frames - 1), value=0)
        
        # 最初のフレームを表示
        self.display_frame()
        self.update_info()
        
        return True
    
    def display_frame(self):
        """現在フレームを表示"""
        if not self.cap:
            return
        
        self.current_frame = max(0, min(self.current_frame, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        
        if not ret:
            return
                
        # フレームをBGRからRGBに変換してPILImage化
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.preview_size)
        
        # 選択範囲を視覚化（開始点と終了点が設定されている場合）
        if self.start_frame is not None and self.end_frame is not None:
            frame_array = np.array(frame_resized)
            
            # 選択範囲内のフレームか判定
            if not (self.start_frame <= self.current_frame <= self.end_frame):
                # 選択範囲外は緑色の透明度付き矩形で表示
                overlay = frame_array.copy()
                overlay[:, :] = [50, 200, 50]  # 緑色
                frame_array = cv2.addWeighted(overlay, 0.3, frame_array, 0.7, 0)
            
            # フレームの下部に選択範囲のタイムラインバーを描画
            bar_height = 10
            bar_width = self.preview_size[0]
            
            # 全体のバーを灰色で描画
            frame_array[-bar_height:, :] = [100, 100, 100]
            
            # 選択範囲部分を色で表示
            start_pixel = int((self.start_frame / max(1, self.total_frames)) * bar_width)
            end_pixel = int((self.end_frame / max(1, self.total_frames)) * bar_width)
            start_pixel = max(0, min(start_pixel, bar_width - 1))
            end_pixel = max(0, min(end_pixel, bar_width - 1))
            if start_pixel <= end_pixel:
                frame_array[-bar_height:, start_pixel:end_pixel+1] = [255, 255, 0]  # 黄色
            
            # 現在フレーム位置を赤色で表示
            current_pixel = int((self.current_frame / max(1, self.total_frames)) * bar_width)
            current_pixel = max(0, min(current_pixel, bar_width - 1))
            frame_array[-bar_height:, current_pixel] = [255, 0, 0]  # 赤色
            
            frame_resized = frame_array
        
        pil_image = Image.fromarray(frame_resized)
        
        # PIL ImageをPPM形式のバイト列に変換
        with io.BytesIO() as output:
            pil_image.save(output, format="PPM")
            image_data = output.getvalue()
        
        self.image_element.update(data=image_data)
        
        # 時間情報を更新
        current_time = self.current_frame / self.fps if self.fps > 0 else 0
        total_time = self.total_frames / self.fps if self.fps > 0 else 0
        time_str = self.format_time(current_time) + " / " + self.format_time(total_time)
        self.window["-TIME_INFO-"].update(time_str)
    
    
    def update_info(self):
        """フレーム情報を更新"""
        if not self.cap:
            return

        total_time = self.total_frames / self.fps if self.fps > 0 else 0        
        info = f"解像度: {self.frame_width} × {self.frame_height}" \
                + f"　フレームレート: {self.fps:.2f} fps" \
                + f"　総フレーム数: {self.total_frames}" \
                + f"　総再生時間: {self.format_time(total_time)}"
        self.window["-FRAME_INFO-"].update(info)

        """選択範囲情報を更新"""
        if self.start_frame is None or self.end_frame is None:
            self.window["-SELECTION_INFO-"].update("未設定")
        else:
            start_time = self.format_time(self.start_frame / self.fps) if self.fps > 0 else "00:00"
            end_time = self.format_time(self.end_frame / self.fps) if self.fps > 0 else "00:00"
            duration_frames = self.end_frame - self.start_frame + 1
            duration_time = self.format_time(duration_frames / self.fps) if self.fps > 0 else "00:00"
            selection_info = f"開始: フレーム {self.start_frame} ({start_time})" \
                + f"　終了: フレーム {self.end_frame} ({end_time}), " \
                + f"　長さ: {duration_frames}フレーム ({duration_time})"
            self.window["-SELECTION_INFO-"].update(selection_info)

    def format_time(self, seconds):
        """秒を MM:SS 形式に変換"""
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes:02d}:{secs:02d}"
    
    def run(self):
        """GUIメインループ"""
        frame_delay = 30  # ミリ秒
        
        while True:
            event, values = self.window.read(timeout=None)
            
            if event == eg.WINDOW_CLOSED or event == "終了":
                break
            
            elif event == "ファイルを開く":
                filepath = eg.popup_get_file(
                    "動画ファイルを選択",
                    file_types=(("動画ファイル", "*.mp4 *.ts *.mkv"), ("All Files", "*.*")),
                    initial_folder='D:\\usr\\DL\\video\\AO2026\\',
                )
                if filepath:
                    self.open_video(filepath)
            
            elif event == "-SLIDER-":
                self.current_frame = int(values["-SLIDER-"])
                self.display_frame()
            
            elif event == "開始点を設定":
                self.start_frame = self.current_frame
                self.display_frame()
                self.update_info()
            
            elif event == "終了点を設定":
                self.end_frame = self.current_frame
                self.display_frame()
                self.update_info()
            
            elif event == "選択をクリア":
                self.start_frame = 0
                self.end_frame = self.total_frames - 1
                self.display_frame()
                self.update_info()

            elif event == "バッチ保存":
                # 入力ファイル名と出力名、選択範囲の開始/終了時間を batch.txt に追記
                if not self.video_path:
                    eg.popup_error("エラー", "動画が読み込まれていません")
                    continue

                output_name = values.get("-SAVE_NAME-", "").strip()
                if not output_name:
                    eg.popup_error("エラー", "保存ファイル名を入力してください")
                    continue
                if not output_name.endswith(".mp4"):
                    output_name += ".mp4"
                output_name = Path(self.video_path).parent / output_name

                if self.start_frame is None or self.end_frame is None:
                    eg.popup_error("エラー", "選択範囲が設定されていません")
                    continue

                try:
                    start_time = self.start_frame / self.fps if self.fps > 0 else 0.0
                    end_time = self.end_frame / self.fps if self.fps > 0 else 0.0
                    ref_name = "D:\\usr\\DL\\video\\AO2026\\AO2026_court.jpg"
                    batch_path = Path.cwd() / "batch.ps1"
                    with open(batch_path, "a", encoding="utf-8-sig") as bf:
                        bf.write(f'python.exe movie_clipper.py "{self.video_path}" --output "{output_name}" ' \
                                    + f'--ref "{ref_name}" --frame_skip 1 ' \
                                    # + f'--matcher phash --match_enter 0.6 --match_leave 0.5 ' \
                                    + f'--matcher orb --feature_threshold 0.8 --min_good_matches 15 ' \
                                    + f'--start {start_time:.3f} --end {end_time:.3f}\n')
                    eg.popup("保存完了", f"{batch_path} に追記しました")
                except Exception as e:
                    eg.popup_error("エラー", f"バッチ保存に失敗しました: {e}")
        
        # クリーンアップ
        if self.cap:
            self.cap.release()
        self.window.close()


def main():
    """メイン関数"""
    gui = MovieClipperGUI()
    gui.run()


if __name__ == "__main__":
    main()
