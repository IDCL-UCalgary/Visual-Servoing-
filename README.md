# A Compact and Efficient Model for Aerial Object Detection


This repository was initiated and completed by [Daniel](your github profile address)'s, [Hiranya](your github profile address)'s, [Fisayo](your github profile address)'s MSc students in the [Intelligent Dynamics and Control Lab (IDCL)](https://ucalgary.ca/labs/intelligent-dynamics-control-lab), [Department of Mechanical and Manufacturing Engineering](https://schulich.ucalgary.ca/mechanical-manufacturing), Schulich School of Engineering, University of Calgary. The lab director is [Dr. Mahdis Bisheban](https://profiles.ucalgary.ca/mahdis-bisheban). The project was supervised by [Dr. Samira Ebrahimi Kahou](https://saebrahimi.github.io) and  [Dr. Mahdis Bisheban](https://profiles.ucalgary.ca/mahdis-bisheban).

For more research and open-source contributions, please visit [IDCL Lab GitHub page](https://github.com/IDCL-UCalgary)

To use this training script, download ultralytics and following pytorch and cuda versions:

System info: 
  Pytorch 2.5.1+cu121
  cuda 12.1
  NVIDIA GeForce RTX 3060

  Activate the vstran environment as your first step:
  source /home/.../ENEL645Project/vs_tran/bin/activate

  Email daniel.yang2@ucalgary.ca for .pt files. They are too large to be uploaded.

  For training using yolo CLI (ie. teacher model) use yolo train, for example:
  yolo train \
    model=/home/.../ENEL645Project/rtdetr-l.yaml \
    data=/home/.../ENEL645Project/overhead_vehicle_detection/data.yaml \
    epochs=125 \
    imgsz=640 \
    optimizer=AdamW \
    batch=4 \
    device=0 \
    workers=2 \
    warmup_epochs=25 \
    lr0=0.00001 \
    lrf=0.02

  To train the student model or rtdetr with KD, open the knowledge_distillation_trainer.py script in vscode. Activate the vstran environment in the terminal. 
  Adjust filepaths to match with teacher model .pt file locations and .yaml locations. Run the script via python knowledge_distillation_train.py

  Once it is finished, change the model so you can use it with yolo CLI using modelChange.py script. Run the script via python modelChange.py

  To run a prediction:
  yolo predict \
  model=/home/.../ENEL645Project/runs/detect/student/weights/best.pt \
  source=/home/.../overhead_drone.mp4 \
  iou=0.5 \
  conf=0.45
  
    
