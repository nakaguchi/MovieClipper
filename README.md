# MovieClipper

MovieClipper は、参照画像に基づいて動画ファイルから特定のシーンを自動検出・抽出する Python ツールです。pHash と SSIM を使用してフレームを比較し、類似したシーンをクリップします。

## 特徴

- 参照画像との類似度に基づくシーン検出
- pHash (64-bit) と SSIM を組み合わせた高速比較
- インターレース除去 (デフォルト有効、--no_deinterlace で無効化)
- 音声ストリームの自動検出と処理
- プログレスバー付きの処理
- CSV 出力による詳細ログ

## 必要条件

- Python 3.7+
- ffmpeg と ffprobe (PATH に存在するか、Python 実行ファイルと同じディレクトリに配置)
- 必要な Python パッケージ: opencv-python, numpy, scikit-image

## インストール

1. リポジトリをクローンまたはダウンロードします。
2. 依存関係をインストールします：

   ```bash
   pip install -r requirements.txt
   ```

3. ffmpeg と ffprobe をインストールし、PATH に追加するか、Python 実行ファイルと同じディレクトリに配置します。

## 使用方法

基本的な使用法：

```bash
python movie_clipper.py input_video.mp4 --ref reference.jpg --output output.mp4
```

### オプション

- `input`: 入力動画ファイル (必須)
- `--ref`: 参照フレーム画像 (デフォルト: 入力ファイルの拡張子を .jpg に置換)
- `--output`: 出力 MP4 ファイル (デフォルト: 入力に応じて自動決定)
- `--segment_csv`: 抽出区間 CSV を出力
- `--frame_csv`: フレーム単位ログ CSV を出力
- `--phash_maxdist`: pHash 最大ハミング距離 (デフォルト: 12)
- `--ssim_size`: SSIM 計算時のリサイズサイズ (デフォルト: 256)
- `--ssim_enter`: SSIM 進入閾値 (デフォルト: 0.78)
- `--ssim_exit`: SSIM 退出閾値 (デフォルト: 0.72)
- `--smooth_sec`: スムージング時間 (秒) (デフォルト: 0.5)
- `--preroll_sec`: プリロール時間 (秒) (デフォルト: 0.2)
- `--postroll_sec`: ポストロール時間 (秒) (デフォルト: 0.2)
- `--min_segment_sec`: 最小セグメント長 (秒) (デフォルト: 0.3)
- `--max_gap_sec`: 最大ギャップ時間 (秒) (デフォルト: 0.15)
- `--fps`: FPS 上書き (デフォルト: 0.0, 自動検出)
- `--crf`: H.264 CRF 品質 (デフォルト: 20)
- `--preset`: H.264 プリセット (デフォルト: medium)
- `--audio_bitrate`: 音声ビットレート (デフォルト: 192k)
- `--no_deinterlace`: インターレース除去を無効化 (デフォルト: 有効)
- `--yadif_args`: yadif フィルタ引数 (デフォルト: mode=send_frame:parity=auto:deint=all)
- `--progress_interval`: プログレス更新間隔 (秒) (デフォルト: 0.2)

### 例

参照画像を使用してシーンを抽出：

```bash
python movie_clipper.py movie.ts --ref scene.jpg --output clipped.mp4 --segment_csv
```

インターレース除去を無効化：

```bash
python movie_clipper.py movie.ts --no_deinterlace
```

## アルゴリズム

1. 参照画像の pHash と SSIM 特徴を計算
2. 動画の各フレームを処理：
   - pHash を計算し、ハミング距離をチェック
   - pHash が一致する場合、SSIM を計算
   - EMA スムージングを適用
   - ヒステリシスを使用してセグメントを検出
3. 検出されたセグメントをマージ
4. ffmpeg を使用してセグメントを抽出・連結

## CSV 出力

- セグメント CSV: 検出された各セグメントの開始/終了時間とパラメータ
- フレーム CSV: 各フレームの pHash 距離、SSIM スコア、スムージング値

## ライセンス

[ライセンス情報をここに記載]
