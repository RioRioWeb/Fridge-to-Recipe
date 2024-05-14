from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {class_name: i for i, class_name in enumerate(sorted(os.listdir(root_dir)))}
        # ディレクトリを探索して画像とラベルのリストを作成
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_file)
                    self.images.append(img_path)
                    # self.labels.append(label)
                    self.labels.append(int(self.class_to_idx[label]))  # 文字列ラベルを数値インデックスに変換

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        label = torch.tensor(int(label))
        if self.transform:
            image = self.transform(image)

        return {"x": image, "label": label}


from torch.utils.data.dataloader import default_collate


def custom_data_collator(batch):
    # バッチ内の各データポイントがディクショナリ形式であることを確認し、テンソルに変換
    batch = {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    return batch
