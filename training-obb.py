from roboflow import Roboflow
from ultralytics import YOLO

# 1. Download your dataset via Roboflow API
rf = Roboflow(api_key="")
project = rf.workspace("").project("")
version = project.version(1)
dataset = version.download("yolo26")

# 2. Load the YOLO26 Small model
model = YOLO("yolo26s-obb.pt")

# 3. Train using the dynamically pulled data.yaml path
model.train(
    data=f"{dataset.location}/data.yaml",  # <-- This points right to the downloaded file
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,              
    workers=0,
    patience=20
)