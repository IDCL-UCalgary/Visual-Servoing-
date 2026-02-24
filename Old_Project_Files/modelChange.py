import torch
from ultralytics.nn.tasks import RTDETRDetectionModel

# === Paths ===
ckpt_path = "/home/eeel126/Desktop/ENEL645Project/student_best_yolo_full.pt"  # your old checkpoint
fixed_path = "/home/eeel126/Desktop/ENEL645Project/student_best_ultralytics.pt"  # YOLO-ready output
yaml_cfg = "rtdetr-s.yaml"  # RTDETR config

#  Load raw Pytorch Modrl
raw_model = torch.load(ckpt_path, map_location="cpu")

# Rebuild Ultralytics RT-DETR wrapper from YAML
ultra_model = RTDETRDetectionModel(cfg=yaml_cfg)
ultra_model.nc = 10  # must match training

# Copy weights from raw model into Ultralytics model
ultra_model.model.load_state_dict(raw_model.state_dict(), strict=True)

# ave in TRUE YOLO CLI format
torch.save(
    {
        "model": ultra_model,    
        "ema": None,
        "optimizer": None,
        "updates": 0,
        "args": {"task": "detect"},
    },
    fixed_path,
)

print("File saved to:", fixed_path)

