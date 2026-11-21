# Fire & Smoke detection model (Unit A - YOLO)

## 1. Goal
Detect **fire** and **smoke** in RGB images or video frames from a drone or a camera. The detections feed the
risk score of the AeroGuard Industry solution (early warning before a fire spreads).

## 2. Architecture
- Model: **YOLOv8n** (Ultralytics), single-stage object detector, about 3.0 million parameters, 8.2 GFLOPs.
- Starting point: weights pretrained on COCO (transfer learning), then fine-tuned on fire and smoke.
- Input size: 416 x 416. Output: bounding boxes, class (Fire / Smoke) and a confidence score.

## 3. Why this model
- Small and fast: it can run **on board a drone** (edge computer) with no Wi-Fi, which answers the "no network" problem.
- One pass over the image gives real-time detection for video.
- Easy to export (ONNX) and to integrate in the demo application.
- Open-source tools, so the cost of the solution stays low.

## 4. Dataset
- Name: Fire & Smoke detection (YOLO format), Roboflow Universe, `sayed-gamall/fire-smoke-detection-yolov11`, version 2.
- Licence: **CC BY 4.0** (credit the authors).
- Classes: 0 = Fire, 1 = Smoke.
- Split: 9,156 train / 872 validation / 435 test images (about 1,516 labelled objects in validation).
- The training folder contains several augmented copies of each original photo (Roboflow augmentation).

## 5. Training
- Epochs: up to 30 (early stopping patience 8), epochs actually run: 30.
- Image size 416, batch 32, optimizer chosen automatically (AdamW), Google Colab T4 GPU.
- Standard YOLO augmentations (mosaic, flips, colour changes).

## 6. Results (on the 435 TEST images, never used for training)
| Metric | Value |
|---|---|
| Precision | 0.765 |
| Recall | 0.700 |
| mAP@50 | 0.756 |
| mAP@50-95 | 0.448 |

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| Fire | 0.809 | 0.779 | 0.828 | 0.486 |
| Smoke | 0.721 | 0.622 | 0.684 | 0.410 |

Figures (in this folder): `results.png` (training curves), `test_confusion_matrix.png`,
`test_BoxPR_curve.png`, `demo_predictions/` (example detections).

## 7. Limits (to say honestly in the pitch)
- General images (many outdoor and wildfire scenes), **not Tunisian factories** and not thermal images.
- Possible near-duplicates between training and test images (augmentation), so the numbers can be a bit optimistic.
- False alarms are possible with orange lights, welding, sun glare and steam. Smoke is harder than flames.
- Not a certified fire alarm: it is an early visual warning that supplements certified detectors.
- Next step: test on industrial images and thermal images, and fine-tune on them.

## 8. How to use
```python
from ultralytics import YOLO
m = YOLO("best.pt")
r = m("image.jpg", conf=0.25)
r[0].show()
```
Class ids: 0 = Fire, 1 = Smoke.
