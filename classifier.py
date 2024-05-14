import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.datasets import ImageFolder
import os
import yaml


class UnlabeledDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): データセットが保存されているディレクトリのパス。
            transform (callable, optional): サンプルに適用されるオプショナルな変換（前処理やデータ拡張）。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.file_names = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_names[idx])
        image = Image.open(img_name).convert("RGB")  # 画像をRGBで読み込む

        if self.transform:
            image = self.transform(image)

        return image, os.path.basename(img_name)


# モデル構造の定義
def initialize_model(num_classes):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def load_model(filepath, model, device):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def prepare_dataloader(data_dir, batch_size=1):
    data_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # dataset = ImageFolder(os.path.join(data_dir), data_transforms)
    dataset = UnlabeledDataset(os.path.join(data_dir, "segment_fullimages"), data_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def inference(model, dataloader, device, idx_to_class):
    model.eval()
    segment_classes = []
    with torch.no_grad():
        for inputs, file_name in dataloader:
            inputs = inputs.to(device)
            # labels = labels.to(device)
            outputs = model(inputs)
            predicted_score, predicted = torch.max(outputs.data, 1)
            # print(predicted_score)
            # if predicted_score < 3.0:
            #     continue
            # print(outputs)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            # print(f'Accuracy: {100 * correct / total}%')
            # print(f'fileName, class', file_name, idx_to_class[predicted.item()])

            # print(predicted,)
            if idx_to_class[predicted.item()] is not None:
                # print(idx_to_class)
                segment_classes.append(idx_to_class[predicted.item()])
    # print(segment_classes)
    return segment_classes


def run_classifier():
    # デバイスの設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # モデルとオプティマイザの初期化
    # num_classes = 15
    # num_classes = 18
    # num_classes = 36
    num_classes = 10
    # クラス数を指定
    model = initialize_model(num_classes).to(device)
    # モデルのロード
    # model = load_model("classifier/results/veg_dataset/resnet18-fine/model_epoch_best.pth", model, device)
    # model = load_model("classifier/results/veg_and_fruite/resnet50-fine/model_epoch_15.pth", model, device)
    # model = load_model("classifier/results /resnet18-fine/model_epoch_best.pth", model, device)

    # best model!
    model = load_model("./models/resnet50-fine/model_epoch_best.pth", model, device)

    data_dir = "./data/"
    dataloader = prepare_dataloader(data_dir)
    # with open('./classifier/results/veg_dataset/resnet18-fine/train_class_ids.yaml') as f:
    # with open('./classifier/results/veg_and_fruite/resnet50-fine/train_class_ids.yaml') as f:
    # best model!
    with open("./models/resnet50-fine/train_class_ids.yaml") as f:
        idx_to_class = yaml.load(f, Loader=yaml.FullLoader)
    # print(idx_to_class)
    segment_classes = inference(model, dataloader, device, idx_to_class)

    return segment_classes


def get_name(class_name):
    if class_name == "apple":
        return "りんご"
    elif class_name == "banana":
        return "バナナ"
    elif class_name == "Carrot":
        return "にんじん"
    elif class_name == "Cabbage":
        return "キャベツ"
    elif class_name == "Cauliflower":
        return "カリフラワー"
    elif class_name == "eggplant":
        return "ナス"
    elif class_name == "onion":
        return "玉ねぎ"
    elif class_name == "Potato":
        return "じゃがいも"
    elif class_name == "Tomato":
        return "トマト"
    elif class_name == "None":
        return ""
    else:
        raise ValueError("Unknown class name")

if __name__ == "__main__":
    run_classifier()
