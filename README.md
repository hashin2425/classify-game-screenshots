# classify-game-screenshots

大量のゲームのスクショを自動分類するための画像分類モデル

## 画像分類モデル

SwinTransformerを用いた画像分類モデル。ゲーム画面に特有の要素（UI、テキスト、キャラクターなど）を効率的に認識し、高精度な分類を実現するらしい。

## 実行環境

- Python 3.11.3
- GPU
  - RTX3060 / 12GB VRAM
  - CUDA Version 12.7

## 使い方

1. 環境構築する

   ```sh
   python -m venv .venv
   .\.venv\Scripts\activate
   python.exe -m pip install --upgrade pip
   python.exe -m pip install -r requirements.txt
    ```

    CUDAを使うときは、<https://pytorch.org/>から適切なバージョンに対応するインストールコマンドを実行する必要がある。CUDAバージョンは多少ずれていても問題なさそう。
    一応、`nvidia-smi`でGPUとCUDAのバージョンを確認できる。

    ```sh
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```

2. 学習用のデータを用意する
3. 分類を行わせる

## ディレクトリ構成

`input`・`output`・`training`ディレクトリは、そのサブディレクトリの名前が画像の分類タグであり、学習時と分類時にはそのディレクトリ名を用いる。

```sh
./
├── input/
│   ├── Screenshot_2024.12.01_00.00.00.000.png
│   ├── Screenshot_2024.12.01_01.00.00.000.png
│   ├── Screenshot_2024.12.01_02.00.00.000.png
│   └── Screenshot_2024.12.01_03.00.00.000.png
├── output/
├── training/ # 訓練用データのディレクトリ
│   ├── tag1/
│   │   ├── Screenshot_2024.12.01_04.00.00.000.png
│   │   └── Screenshot_2024.12.01_05.00.00.000.png
│   └── tag2/
│       ├── Screenshot_2024.12.01_06.00.00.000.png
│       └── Screenshot_2024.12.01_07.00.00.000.png
├── model/
│   └── trained_model_1.pickle # 訓練済みモデル
├── utils/ # ユーティリティを定義する
│   └── image_utils.py
├── .gitignore
├── run_training.ipynb # trainingディレクトリのデータからモデルの学習を行う
├── run_classify.py # 訓練済みモデルを用いてinputディレクトリの画像を分類する
└── README.md
```
