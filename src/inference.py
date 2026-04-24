
import json
from pathlib import Path
from ultralytics import YOLO

def run_inference_and_export():
    weights_path = Path("./runs/detect/runs/improved/weights/best.pt")
    if not weights_path.exists():
        raise FileNotFoundError(f"Файл весов не найден: {weights_path}.")

    model = YOLO(weights_path)

    print("Запуск валидации на тестовом наборе")
    metrics = model.val(data="./dataset/data.yaml", split="test", imgsz=320, project="./runs", name="final_eval")

    print("Генерация предсказаний на тестовых изображениях")
    pred_results = model.predict(
        source="./dataset/test/images", 
        save=True, 
        conf=0.25, 
        imgsz=320, 
        project="./runs", 
        name="predictions"
    )

    final_report = {
        "model_version": "YOLOv11n (Improved, imgsz=320)",
        "metrics": {
            "mAP@0.5": float(metrics.box.map50),
            "mAP@0.5:0.95": float(metrics.box.map),
            "Precision": float(metrics.box.mp),
            "Recall": float(metrics.box.mr)
        },
        "test_images_count": len(pred_results)
    }


    report_path = Path("./results/final_metrics.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    print("\nФинальные метрики (Improved):")
    for k, v in final_report["metrics"].items():
        print(f"   {k}: {v:.4f}")
    print(f"\nОтчёт сохранён в {report_path}")
    print(f"Визуализации предсказаний: ./runs/predictions/")

if __name__ == "__main__":
    run_inference_and_export()