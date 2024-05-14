from transformers import ViTFeatureExtractor, Trainer, TrainingArguments, ViTForImageClassification
from Dataset import CustomImageDataset, custom_data_collator
import torch
from time import time
from torchvision import transforms
from torchvision import models


def main():
    # デバイス設定（GPUが利用可能な場合はGPUを使用）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    # model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
    model = models.resnet50(pretrained=True)
    num_labels = 30
    model.fc = torch.nn.Linear(model.fc.in_features, num_labels)
    model.to(device)

    # # モデルの分類器部分をカスタムクラス数に対応するように置き換え
    # model.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)

    # データセットの読み込み
    train_dataset = CustomImageDataset('cook-ai-dataset/train', transform=data_transforms['train'])
    test_dataset = CustomImageDataset('cook-ai-dataset/test', transform=data_transforms['test'])

    # トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=960,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    # トレーナーの設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=custom_data_collator,
    )

    # トレーニングの実行
    s = time()
    train_result = trainer.train()
    print(f"training time: [{time()-s:.1f} sec]")
    trainer.save_model()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)

    dataset_metrics = trainer.evaluate(metric_key_prefix="test")
    metrics.update(dataset_metrics)
    trainer.log_metrics("eval", metrics)

    trainer.save_metrics("all", metrics)


if __name__ == "__main__":
    main()
