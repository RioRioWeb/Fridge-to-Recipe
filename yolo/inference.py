from ultralytics import YOLO


def main():
    # Load a model
    # model = YOLO('yolov8n.pt')  # load an official model
    model = YOLO('runs/detect/train3/weights/best.pt')  # load a custom model

    # Predict with the model
    # results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
    # results = model('../data/refrigerator_test/IMG_1817.JPG')
    # print(results[0])
    model.predict('../data/refrigerator_test/IMG_1554.jpg', save=True, conf=0.1)
    
if __name__ == "__main__":
    main()