"""
AeroGuard Industry - Step 1: fire & smoke detection with YOLOv8 on D-Fire.

Run in Google Colab (Runtime > Change runtime type > T4 GPU):
    !pip install -q ultralytics
    !python 1_vision_yolo.py /content/dfire

/content/dfire = folder where you copied the D-Fire "Training, validation and test sets".
The script finds the images/ folders by itself and prints what it found.

IMPORTANT: check the class ids on 3-4 images before training (see CHECK below).
D-Fire is usually: 0 = smoke, 1 = fire. If your check shows the opposite, change NAMES.
"""
import sys
import pathlib
import yaml
from ultralytics import YOLO

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/content/dfire")
EPOCHS = 30          # lower to 15-20 if you are short on time
IMGSZ = 416          # D-Fire images are small, 416 is fast and enough
NAMES = {0: "smoke", 1: "fire"}   # VERIFY with the CHECK below
FIRE_ID = 1
SMOKE_ID = 0

# ---------------------------------------------------------------- find folders
img_dirs = sorted(p for p in ROOT.rglob("images") if p.is_dir())
print("images folders found:")
for p in img_dirs:
    print("  ", p, "->", len(list(p.glob("*"))), "files")
if not img_dirs:
    sys.exit("No 'images' folder found under " + str(ROOT) + ". Check the path / unzip.")

# If the dataset ships its own data.yaml (Roboflow / Kaggle YOLO datasets do), read the class
# names and order from it. This is safer than guessing which id is fire and which is smoke.
yml = sorted(ROOT.rglob("data.yaml"))
if yml:
    cfg = yaml.safe_load(yml[0].read_text())
    nm = cfg.get("names")
    if isinstance(nm, list):
        nm = {i: n for i, n in enumerate(nm)}
    if isinstance(nm, dict):
        NAMES = {int(k): str(v) for k, v in nm.items()}
        for k, v in NAMES.items():
            if v.lower() == "fire":
                FIRE_ID = k
            if v.lower() == "smoke":
                SMOKE_ID = k
        print("Class names read from", yml[0], "->", NAMES, "| FIRE_ID =", FIRE_ID, "| SMOKE_ID =", SMOKE_ID)


def pick(*keywords):
    for kw in keywords:
        for p in img_dirs:
            rel = [x.lower() for x in p.relative_to(ROOT).parts]
            if kw in rel:
                return p
    return None


train_dir = pick("train")
val_dir = pick("val", "valid", "validation", "test")
if train_dir is not None and val_dir is not None:
    train_src, val_src = str(train_dir), str(val_dir)
else:
    # One flat images/ folder: make our own 80/20 split with list files (no copying needed)
    import random
    all_imgs_dir = max(img_dirs, key=lambda p: len(list(p.glob("*"))))
    files = sorted(f for f in all_imgs_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    random.Random(42).shuffle(files)
    cut = int(0.8 * len(files))
    pathlib.Path("train.txt").write_text("\n".join(str(f) for f in files[:cut]))
    pathlib.Path("val.txt").write_text("\n".join(str(f) for f in files[cut:]))
    train_dir = val_dir = all_imgs_dir
    train_src = str(pathlib.Path("train.txt").resolve())
    val_src = str(pathlib.Path("val.txt").resolve())
    print("No train/val folders found: made our own split from", all_imgs_dir)
    print("  train images:", cut, "| val images:", len(files) - cut)
print("TRAIN:", train_src)
print("VAL  :", val_src)

# ---------------------------------------------------------------- CHECK (class ids)
# Draw boxes on 4 training images so you can SEE which id is fire and which is smoke.
try:
    import cv2
    out = pathlib.Path("check_labels")
    out.mkdir(exist_ok=True)
    shown = 0
    for img_path in sorted(train_dir.glob("*")):
        lbl = pathlib.Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        if not lbl.exists() or lbl.stat().st_size == 0:
            continue
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        for line in lbl.read_text().splitlines():
            c, x, y, bw, bh = line.split()[:5]
            x, y, bw, bh = float(x) * w, float(y) * h, float(bw) * w, float(bh) * h
            p1, p2 = (int(x - bw / 2), int(y - bh / 2)), (int(x + bw / 2), int(y + bh / 2))
            cv2.rectangle(img, p1, p2, (0, 0, 255), 2)
            cv2.putText(img, "id=" + c, (p1[0], max(p1[1] - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(out / img_path.name), img)
        shown += 1
        if shown == 4:
            break
    print("CHECK: open the folder 'check_labels' and confirm which id is fire and which is smoke.")
except Exception as e:  # not fatal
    print("label check skipped:", e)

# ---------------------------------------------------------------- data.yaml
data = {"train": train_src, "val": val_src, "names": NAMES}
yaml_path = pathlib.Path("dfire.yaml")
yaml_path.write_text(yaml.safe_dump(data))

# ---------------------------------------------------------------- train
model = YOLO("yolov8n.pt")   # nano: small and fast, fits a drone's on-board computer
model.train(
    data=str(yaml_path),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=32,
    patience=8,
    project="runs",
    name="dfire_v8n",
    workers=2,
)

# ---------------------------------------------------------------- evaluate
best = YOLO("runs/dfire_v8n/weights/best.pt")
metrics = best.val(data=str(yaml_path), imgsz=IMGSZ)
print("mAP50    :", metrics.box.map50)
print("mAP50-95 :", metrics.box.map)
print("(confusion matrix + curves are saved in runs/dfire_v8n and runs/detect/val*)")


# ---------------------------------------------------------------- risk level 0..3 (visual part)
def risk_level(result):
    """Visual risk from one YOLO result. Thresholds are yours to tune and explain."""
    h, w = result.orig_shape
    fire = smoke = 0.0
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = ((x2 - x1) * (y2 - y1)) / (h * w)
        if int(box.cls) == FIRE_ID:
            fire += area
        elif int(box.cls) == SMOKE_ID:
            smoke += area
    if fire > 0.10 or (fire > 0.02 and smoke > 0.15):
        return 3, fire, smoke
    if fire > 0.02 or (fire > 0 and smoke > 0):
        return 2, fire, smoke
    if fire > 0 or smoke > 0:
        return 1, fire, smoke
    return 0, fire, smoke


# ---------------------------------------------------------------- demo on a few validation images
sample = sorted(val_dir.glob("*"))[:12]
results = best.predict(source=[str(p) for p in sample], conf=0.25, imgsz=IMGSZ, save=True, project="runs", name="demo")
for p, r in zip(sample, results):
    lvl, f, s = risk_level(r)
    print(p.name, "-> risk level", lvl, "| fire area %.3f | smoke area %.3f" % (f, s))

# For your demo video, run on a video file:
#   best.predict(source="my_video.mp4", conf=0.25, save=True)
print("DONE. Weights: runs/dfire_v8n/weights/best.pt  (copy it to your Drive and to your GitHub repo)")
