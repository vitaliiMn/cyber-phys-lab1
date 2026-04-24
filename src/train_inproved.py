from ultralytics import YOLO

def train_improved():
    model = YOLO("yolo11n.pt"
    results = model.train(
        data="./dataset/data.yaml",
        epochs=15,
        imgsz=320,
        batch=8,
        
        # Гипотезы
        mosaic=0.5,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.5,
        fliplr=0.5,
        
        # Гиперпараметры
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.0005,
        patience=10, 
        
        project="./runs",
        name="improved",
        exist_ok=True
    )
    
    print("Обучение улучшенной модели завершено")
    return results

if __name__ == "__main__":
    train_improved()