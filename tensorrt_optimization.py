from ultralytics import YOLO
import tensorrt as trt
import torch

# 1. Quick check to ensure the new installation is visible
print(f"Using TensorRT Version: {trt.__version__}")
print(f"CUDA Available for Torch: {torch.cuda.is_available()}")

# 2. Load your custom YOLO26s weights
model = YOLO("yolo26_s_best.onnx") # or model = YOLO("yolo26_s.pt")

# 3. Export to TensorRT Engine
# imgsz: Matches your training size (640 is standard)
# half: Enables FP16 for massive speed gains on your RTX 3060
# device: 0 ensures it uses your GPU
# workspace: 4 assigns 4GB of VRAM for the optimization process
try:
    print("Starting TensorRT optimization... this may take a few minutes.")
    model.export(
        format="engine", 
        device=0, 
        half=True, 
        imgsz=640, 
        workspace=4
    )
    print("Optimization Complete! Generated: best_26_s.engine")
except Exception as e:
    print(f"An error occurred during export: {e}")