"""
MovieClipper GUI - TkEasyGUI版
動画ファイルの読み込み、再生、シークバー操作が可能なGUI
"""

import os
import io
import TkEasyGUI as eg
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
from pathlib import Path


class MovieClipperGUI:
    """動画編集GUI"""
    
    def __init__(self):
        
        # 変数初期化
        self.preview_size = (640, 360)
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.frame_width = 0
        self.frame_height = 0
        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 1.0
        
        # ウィンドウレイアウト定義
        self.layout = [
            [
                eg.Button("ファイルを開く", size=(12, 1)),
                eg.Text("未読み込み", key="-FILE_STATUS-", size=(40, 1)),
            ],
            [
                eg.Image(key="-IMAGE-", size=self.preview_size),
            ],
            [
                eg.Slider(
                    range=(0, 100),
                    default_value=0,
                    orientation="h",
                    size=(60, 2),
                    key="-SLIDER-",
                    enable_events=True,
                ),
            ],
            [
                eg.Button("▶ 再生", size=(8, 1), key="-PLAY-"),
                eg.Button("⏸ 一時停止", size=(8, 1), key="-PAUSE-"),
                eg.Button("⏹ 停止", size=(8, 1), key="-STOP-"),
                eg.Text("", key="-TIME_INFO-", size=(30, 1)),
            ],
            [
                eg.Text("フレーム情報:", font=("Arial", 10, "bold")),
                eg.Text("", key="-FRAME_INFO-", size=(80, 1)),
            ],
            [
                eg.Text("再生速度:"),
                eg.Slider(
                    range=(0.25, 2.0),
                    default_value=1.0,
                    resolution=0.25,
                    orientation="h",
                    size=(30, 1),
                    key="-SPEED-",
                    enable_events=True,
                ),
                eg.Text("", key="-SPEED_TEXT-", size=(10, 1)),
            ],
        ]
        
        self.window = eg.Window("MovieClipper GUI", self.layout, finalize=True)
        self.image_element = self.window["-IMAGE-"]
        
    def open_video(self, filepath):
        """動画ファイルを開く"""
        if self.cap:
            self.cap.release()
        
        self.video_path = filepath
        self.cap = cv2.VideoCapture(filepath)
        
        if not self.cap.isOpened():
            eg.popup_error("エラー", f"動画ファイルを開けません: {filepath}")
            self.video_path = None
            return False
        
        # 動画情報を取得
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0
        self.is_playing = False
        
        # ファイル名を表示
        filename = Path(filepath).name
        self.window["-FILE_STATUS-"].update(f"読み込み完了: {filename}")
        
        # スライダーの最大値を設定
        self.window["-SLIDER-"].update(range=(0, self.total_frames - 1), value=0)
        
        # 最初のフレームを表示
        self.display_frame()
        self.update_frame_info()
        
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
    
    def update_frame_info(self):
        """フレーム情報を更新"""
        if not self.cap:
            return
        
        info = f"解像度: {self.frame_width} × {self.frame_height}" + f"　フレームレート: {self.fps:.2f} fps" + \
                f"　総フレーム数: {self.total_frames}" + f"　現在のフレーム: {self.current_frame}" + \
                f"　総再生時間: {self.format_time(self.total_frames / self.fps if self.fps > 0 else 0)}"
        self.window["-FRAME_INFO-"].update(info)
    
    def format_time(self, seconds):
        """秒を MM:SS 形式に変換"""
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes:02d}:{secs:02d}"
    
    def play(self):
        """再生"""
        if not self.cap:
            print("動画ファイルが読み込まれていません")
            eg.popup_warning("先に動画ファイルを読み込んでください", "警告")
            return
        
        # self.window["-SLIDER-"].update(disabled=True)
        
        if self.current_frame >= self.total_frames - 1:
            self.current_frame = 0
            self.display_frame()
        
        self.is_playing = True
        self.playback_loop()
    
    def playback_loop(self):
        """再生ループ（スレッド内で実行）"""
        frame_delay = int(1000 / (self.fps * self.playback_speed)) if self.fps > 0 else 0
        print(f"frame_delay: {frame_delay} ms")
        
        while self.is_playing and self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.display_frame()
            self.window["-SLIDER-"].update(value=self.current_frame)
            self.update_frame_info()
            
            # イベント処理
            event, values = self.window.read(timeout=1000)
            
            if event == eg.WINDOW_CLOSED or event == "終了":
                self.is_playing = False
                return
            elif event == "-PAUSE-":
                self.is_playing = False
                return
            elif event == "-STOP-":
                self.is_playing = False
                self.current_frame = 0
                self.display_frame()
                self.window["-SLIDER-"].update(value=self.current_frame)
    
    def run(self):
        """GUIメインループ"""
        frame_delay = 30  # ミリ秒
        
        while True:
            event, values = self.window.read(timeout=frame_delay)
            
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
            
            elif event == "-PLAY-":
                self.play()
            
            elif event == "-PAUSE-":
                self.is_playing = False
            
            
            elif event == "-SLIDER-":
                if not self.is_playing:
                    self.current_frame = int(values["-SLIDER-"])
                    self.display_frame()
            
            elif event == "-SPEED-":
                self.playback_speed = values["-SPEED-"]
                speed_text = f"{self.playback_speed:.2f}x"
                self.window["-SPEED_TEXT-"].update(speed_text)
        
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
