import os
import argparse
import torch
from timm import create_model
from tqdm import tqdm
from PIL import Image
import shutil
from utils.image_utils import GameScreenshotDataset, load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Classify game screenshots")
    parser.add_argument("--model", type=str, default="model/trained_model_1.pickle", help="Path to trained model")
    parser.add_argument("--input", type=str, default="input", help="Input directory containing images to classify")
    parser.add_argument("--output", type=str, default="output", help="Output directory for classified images")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    return parser.parse_args()


def prepare_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model("swin_base_patch4_window7_224", num_classes=num_classes)
    model = load_model(model, model_path)
    model = model.to(device)
    model.eval()
    return model, device


def classify_images(model, device, input_dir, output_dir, transform):
    # 出力ディレクトリの準備
    os.makedirs(output_dir, exist_ok=True)

    # 入力画像の取得
    image_files = [f for f in os.listdir(input_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

    for image_file in tqdm(image_files, desc="Classifying images"):
        # 画像の読み込みと前処理
        image_path = os.path.join(input_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # 予測
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = outputs.max(1)
            predicted_class = model.class_names[predicted.item()]

        # 分類結果に基づいて画像を移動
        class_dir = os.path.join(output_dir, predicted_class)
        os.makedirs(class_dir, exist_ok=True)
        shutil.copy2(image_path, os.path.join(class_dir, image_file))


def main():
    args = parse_args()

    # 訓練データからクラス情報を取得
    dataset = GameScreenshotDataset("training")
    num_classes = len(dataset.classes)

    # モデルの準備
    model, device = prepare_model(args.model, num_classes)
    model.class_names = dataset.classes

    # 画像の分類実行
    classify_images(model, device, args.input, args.output, dataset.get_default_transform())

    print(f"Classification completed. Results saved in {args.output}")


if __name__ == "__main__":
    main()
