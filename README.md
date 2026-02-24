# A Compact and Efficient Model for Aerial Object Detection

<p align="center">
  <img src="Media/kinova-orange.gif" width="70%" />
</p>

<p align="center">
  <em>Fruit Detection with Visual Servoing Pick and Place</em>
</p>

---

## Overview

This repository presents a compact and computationally efficient aerial object detection framework based on RT-DETR and knowledge distillation. The objective is to achieve strong detection performance while reducing model size and computational cost for UAV deployment.

This project was completed by MSc students in the Intelligent Dynamics and Control Lab (IDCL), University of Calgary.

---

## Authors

- Daniel Yang — MSc Student, IDCL  
- Hiranya — MSc Student, IDCL  
- Fisayo — MSc Student, IDCL  

**Lab:** Intelligent Dynamics and Control Lab (IDCL)  
Department of Mechanical and Manufacturing Engineering  
Schulich School of Engineering  
University of Calgary  

Lab Director: Dr. Mahdis Bisheban  
Supervisors:  
- Dr. Samira Ebrahimi Kahou  
- Dr. Mahdis Bisheban  

More research from IDCL:  
https://github.com/IDCL-UCalgary

---

## Repository Structure

```
.
├── Media/                      # Demo GIFs for README
├── Results/                    # Evaluation outputs
├── Student_results/            # Student model metrics
├── teacher_results/            # Teacher model metrics
├── knowledge_distillation_trainer.py
├── modelChange.py
├── rtdetr-s.yaml
├── rtdetr-l.yaml
├── data.yaml
└── README.md
```

---

## Environment Setup

### System Configuration

- PyTorch 2.5.1+cu121  
- CUDA 12.1  
- NVIDIA GeForce RTX 3060  

### Install Dependencies

```bash
pip install ultralytics
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Activate Environment

```bash
source /home/.../ENEL645Project/vs_tran/bin/activate
```

Adjust the path according to your system.

---

## Training

### Train Teacher Model (YOLO CLI)

```bash
yolo train \
  model=/path/to/rtdetr-l.yaml \
  data=/path/to/data.yaml \
  epochs=125 \
  imgsz=640 \
  optimizer=AdamW \
  batch=4 \
  device=0 \
  workers=2 \
  warmup_epochs=25 \
  lr0=0.00001 \
  lrf=0.02
```

---

### Train Student Model with Knowledge Distillation

1. Open `knowledge_distillation_trainer.py`
2. Adjust:
   - Teacher `.pt` file paths
   - YAML configuration paths
   - Dataset paths
3. Run:

```bash
python knowledge_distillation_trainer.py
```

---

## Convert Student Model for YOLO CLI

After training:

```bash
python modelChange.py
```

This converts the distilled model into YOLO-compatible format.

---

## Inference

```bash
yolo predict \
  model=/path/to/runs/detect/student/weights/best.pt \
  source=/path/to/overhead_drone.mp4 \
  iou=0.5 \
  conf=0.45
```

---

## Model Weights

Model weights (`.pt` files) are not included due to size constraints.

For access, contact:

daniel.yang2@ucalgary.ca

---

## Results

Performance metrics for teacher and student models are available in:

- `Results/`
- `Student_results/`
- `teacher_results/`

Metrics include:
- Precision
- Recall
- mAP
- F1-score
- Confusion matrices

---

## Citation

```
@misc{IDCL_AerialDetection_2026,
  title={PDVS-RT-DETR: A Compact and Efficient Model for Aerial Vehicle Detection},
  author={Daniel Yang, Hiranya Udagedara, Fisayo Olofin, Mahdis Bisheban, Samira Ebrahimi Kahou},
  year={2026},
  institution={University of Calgary}
}
```

---

