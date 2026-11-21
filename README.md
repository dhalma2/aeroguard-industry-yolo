# AeroGuard Industry: Fire and Smoke Detection (YOLOv8n)

AI vision module of **AeroGuard Industry**, our project for the Tunisian Aeroday 2027 "Challenge Aero Entrepreneur" (partners: Protection Civile and INJAZ Tunisia).

**Idea:** an autonomous drone patrols industrial sites (electrical rooms, storage piles, chemical depots). The pipeline is Sense, Predict, Detect, Act. This repository contains the **Detect** step: a model that finds fire and smoke in camera images.

## Results

Evaluated on the 435 test images of the dataset (never seen during training).

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| All   | 0.765     | 0.700  | 0.756  | 0.448     |
| Fire  | 0.809     | 0.779  | 0.828  | 0.486     |
| Smoke | 0.721     | 0.622  | 0.684  | 0.410     |

Inference speed: about 3 ms per image on a Colab T4 GPU. This is not a measurement on a drone's on-board computer. We consider the model small enough (3.1 million parameters, 6 MB) to be a candidate for on-board use, but it still has to be tested on real drone hardware.

## Model

- Architecture: YOLOv8n (Ultralytics), single-stage object detector
- Classes: 0 = Fire, 1 = Smoke
- Image size: 416, 30 epochs with early stopping, trained on a free Colab T4 GPU
- Why YOLOv8n: small and fast, so it can run at the edge (on the drone) without Wi-Fi, and easy to train in a hackathon

## Files

| File | Content |
|------|---------|
| `best.pt` | Trained weights (PyTorch) |
| `best.onnx` | Same model exported to ONNX |
| `1_vision_yolo111111.py` | Training script (run in Google Colab) |
| `MODEL_REPORT.md` | Full model report |
| `results.png`, `results.csv` | Training curves and values |
| `confusion_matrix.png`, `confusion_matrix_normalized.png` | Confusion matrices |
| `BoxPR_curve.png`, `BoxF1_curve.png` | Precision-recall and F1 curves |
| `demo_predictions/` | Example detections |
| `CLASSES.txt` | Class names |

## How to use

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model("image.jpg")
results[0].show()
```

Install first with `pip install ultralytics`.

## Dataset and credit

Roboflow Universe, "Fire-Smoke-Detection-Yolov11" by **sayed-gamall**, version 2, licence **CC BY 4.0**.
https://universe.roboflow.com/sayed-gamall/fire-smoke-detection-yolov11
Split: 9,156 train / 872 validation / 435 test images. The dataset is not included in this repository.

## Honest limits

- The training data are public images, mostly not from Tunisian factories.
- The training set contains augmented copies of the same photos, so the scores may be a little optimistic.
- Smoke is missed more often than fire (recall 0.62 against 0.78). This is why AeroGuard also uses thermal and gas sensors.
- Some images give no detection at all (see `demo_predictions/`).
- No test has been done on a real drone yet.
- Indoor chemical zones would need ATEX-certified equipment.
