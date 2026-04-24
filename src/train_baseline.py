from ultralytics import YOLO

def train_baseline():
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="./dataset/data.yaml",
        epochs=15,           
        imgsz=320,           
        batch=8,           
        patience=10,         
        project="./runs",
        name="baseline",
        exist_ok=True
    )
    print("Бейзлайн завершён")
    return results

if __name__ == "__main__":
    train_baseline()
