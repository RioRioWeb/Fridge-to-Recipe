import cv2
from datetime import datetime
from classifier import run_classifier, get_name
from sam import run_sam


# 画像処理関数
def ML():
    # 物体検出モデルSAMで入力画像中の物体画像をcrop
    run_sam() 
    # cropされた物体画像を食材分類モデルで分類
    segement_classes = run_classifier()
    # 検出された食材名のリスト
    food_names = [get_name(segement_class) for segement_class in segement_classes]
    return food_names
