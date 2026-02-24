import torch
from ultralytics.nn.tasks import RTDETRDetectionModel

# === Paths ===
ckpt_path = "/home/eeel126/Desktop/ENEL645Project/overhead_vehicle_detection/student_best.pt"  # old checkpoint
fixed_path = "/home/eeel126/Desktop/ENEL645Project/car_depth_reduced.pt"  # YOLO-ready output
yaml_cfg = "rtdetr-s.yaml"  # RTDETR config

#  Load raw Pytorch Modrl
raw_model = torch.load(ckpt_path, map_location="cpu")

# Rebuild Ultralytics RT-DETR wrapper from YAML
ultra_model = RTDETRDetectionModel(cfg=yaml_cfg)
ultra_model.nc = 1  # must match training

# Copy weights from raw model into Ultralytics model
ultra_model.model.load_state_dict(raw_model.state_dict(), strict=True)

# ave in TRUE YOLO CLI format
head = ultra_model.model[-1]
head.loss_coeff = torch.tensor([2.0, 2.5, 2.5])  # Match your training values

# Save with train_args to preserve settings
torch.save(
    {
        "model": ultra_model,
        "epoch": 0,
        "best_fitness": 0.0,
        "ema": None,
        "optimizer": None,
        "updates": 0,
        "train_args": {
            "task": "detect",
            "mode": "train",
            "model": yaml_cfg,
            "data": "data.yaml",
            "imgsz": 640,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
        },
    },
    fixed_path,
)


print("File saved to:", fixed_path)

