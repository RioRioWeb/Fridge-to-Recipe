import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import yaml
import sys
import argparse


def get_model(model_name, num_classes=30):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
    elif model_name == "resnet18-fine":
        model = models.resnet18(pretrained=True)
        # すべてのパラメータを凍結
        for param in model.parameters():
            param.requires_grad = False

        # 最後の畳み込みブロックと全結合層の凍結を解除
        for param in model.layer4.parameters():
            param.requires_grad = True
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)  # 例えば、10クラス分類問題の場合
        for param in model.fc.parameters():
            param.requires_grad = True
        
        return model
    elif model_name == "resnet50-fine":
        model = models.resnet50(pretrained=True)
        # すべてのパラメータを凍結
        for param in model.parameters():
            param.requires_grad = False

        # 最後の畳み込みブロックと全結合層の凍結を解除
        for param in model.layer4.parameters():
            param.requires_grad = True
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)  # 例えば、10クラス分類問題の場合
        for param in model.fc.parameters():
            param.requires_grad = True
        return model
    else:
        raise ValueError("Invalid model name")


def save_model(model, optimizer, epoch, loss, filepath="model_checkpoint.pth"):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        filepath,
    )


def train_model(
    model,
    criterion,
    optimizer,
    num_epochs=25,
    dataloaders=None,
    dataset_sizes=None,
    device="cuda:0",
    results_dir="results",
):
    best_loss = float("inf")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # モデルの保存
            if phase == "valid" and epoch_loss < best_loss:
                best_loss = epoch_loss
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                file_path = os.path.join(results_dir, f"model_epoch_best.pth")
                save_model(model, optimizer, epoch, epoch_loss, filepath=file_path)
                print(f"Model saved: model_epoch_{epoch}.pth")

    return model


def evaluate_model(model, dataloader, criterion, device="cuda:0"):
    model.eval()  # モデルを評価モードに設定
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():  # 勾配計算を無効化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

    total_loss = running_loss / total_samples
    total_accuracy = running_corrects.double() / total_samples

    print(f"Loss: {total_loss:.4f} Accuracy: {total_accuracy:.4f}")


def main():
    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(description="Train a model on the specified dataset.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory where the dataset is located.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory where the results should be saved.")
    parser.add_argument("--model", type=str, required=True, help="Model to use for training.")

    # 引数を解析
    args = parser.parse_args()

    # 引数の値を使用
    data_dir = args.data_dir
    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    model_name = args.model
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "valid", "test"]
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4) for x in ["train", "valid", "test"]
    }
    class_to_idx = image_datasets["train"].class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    train_class_ids = os.path.join(results_dir, model_name, "train_class_ids.yaml")
    with open(train_class_ids, "w") as f:
        yaml.dump(idx_to_class, f)
    class_to_idx = image_datasets["test"].class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    test_class_ids = os.path.join(results_dir, model_name, "test_class_ids.yaml")
    with open(test_class_ids, "w") as f:
        yaml.dump(idx_to_class, f)
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "valid", "test"]}
    class_names = image_datasets["train"].classes
    print(f"Number of classes: {len(class_names)}")
    print(f"Dataset sizes: {dataset_sizes}")
    model = get_model(model_name, num_classes=len(class_names))
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    results_dir = os.path.join(results_dir, model_name)
    model = train_model(
        model,
        criterion,
        optimizer,
        num_epochs=20,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=device,
        results_dir=results_dir,
    )
    evaluate_model(model, dataloaders["test"], criterion, device=device)


if __name__ == "__main__":
    main()
