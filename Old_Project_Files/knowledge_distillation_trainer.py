import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.metrics import ap_per_class
from ultralytics.utils.metrics import ap_per_class
from ultralytics.utils.ops import scale_boxes, xyxy2xywh
from ultralytics.utils.metrics import box_iou
from PIL import Image
import os   
import numpy as np
import glob
import torch.optim.lr_scheduler as lr_scheduler
import gc

import torchvision.transforms as T

# clear any leftover GPU memory
torch.cuda.empty_cache()

def clear_cuda():
        torch.cuda.empty_cache()   
        torch.cuda.synchronize()   
        gc.collect() 

class OverheadVehicleDataset(Dataset):
    def __init__(self, data_yaml, split="train", imgsz=640, transforms=None):
        import yaml

        #Get the images through my data.yaml file
        with open(data_yaml) as f:
            data = yaml.safe_load(f)

        self.img_dir = data[split]
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.*")))
        self.label_dir = self.img_dir.replace("images", "labels")
        self.label_files = [
            os.path.join(self.label_dir, os.path.basename(f).replace(".jpg", ".txt").replace(".png", ".txt"))
            for f in self.img_files
        ]

        self.imgsz = imgsz

        #yaml file does not specify how to process image, just use pytorch to tensor
        if transforms is None:
            self.transforms = T.ToTensor()
        else:
            self.transforms = transforms

        if len(self.img_files) == 0:
            raise ValueError(f"No images found in {self.img_dir}")
        
    
    #how many images
    def __len__(self):
        return len(self.img_files)

    #get one image and turn it into input form: tensors from labels file and processed image
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.imgsz, self.imgsz))

        labels = []
        label_path = self.label_files[idx]
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls, x_center, y_center, w, h = map(float, line.strip().split())
                    labels.append([cls, x_center, y_center, w, h])
        labels = torch.tensor(labels) if labels else torch.zeros((0, 5))

        img = self.transforms(img)

        return img, labels

#.yaml, .pt paths
student_cfg = "/home/eeel126/Desktop/ENEL645Project/rtdetr-s.yaml"

teacher_weights = [
    "/home/eeel126/Desktop/ENEL645Project/runs/detect/fold1/weights/best.pt",
    "/home/eeel126/Desktop/ENEL645Project/runs/detect/fold5/weights/best.pt",
]

data_yaml = "/home/eeel126/Desktop/ENEL645Project/overhead_vehicle_detection/data.yaml"

#We want to prioritize object detection, not model mapping.
#First train student model to detect
kd_warmup_epochs = 25  
kd_feat_weight_initial = 0.0  # No knowledge distillation for first 25 epochs
kd_feat_weight_final = 0.05    # Apply 0.05 weighting on KD after 25 epochs
       
epochs = 125 #total number of training epochs
batch_size = 4  #process 4 images at a time
imgsz = 640
#slowly increase learning rate from 1e-5 to 2e-4 to prevent instability
initial_lr = 1e-5
target_lr = 2e-4
weight_decay = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

#Configure the student model
student = RTDETRDetectionModel(cfg=student_cfg)
student.nc = 10
head = student.model[-1]                 # RTDETRDetectionHead
head.loss_coeff = torch.tensor([2.0,     #cross entropy classification weight
                                2.0,     #L1 bounding box weight
                                3.0])    #GIOU union of bounding boxes weight - ensures that overlap exists - model can cheat with this though

import yaml
print("="*60)
print("MODEL LOADING DIAGNOSTICS")
print("="*60)

# Check what's in the YAML
with open(student_cfg) as f:
    yaml_content = yaml.safe_load(f)
print(f"YAML scales['l']: {yaml_content['scales']['l']}")

# Check what the model actually has
total_params = sum(p.numel() for p in student.parameters())
print(f"Model parameters: {total_params:,}")

# Check model's internal config
if hasattr(student, 'yaml'):
    print(f"Model's yaml attribute: {student.yaml}")
if hasattr(student, 'args'):
    print(f"Model's args: {student.args}")

# Check a specific layer to see actual dimensions
for name, module in student.named_modules():
    if isinstance(module, torch.nn.Conv2d) and 'model.10' in name:  # The Conv layer after backbone
        print(f"Layer {name}: out_channels={module.out_channels}")
        print(f"Expected for 0.7x: ~180 channels")
        print(f"Expected for 1.0x: ~256 channels")
        break

print("="*60)

if total_params > 20_000_000:
    print("Model is >20M parameters - scale NOT applied!")
    print("Expected: ~16M for 0.7x scaling")
elif total_params < 18_000_000:
    print("Model size looks correct")


student.train()
student.to(device)

print(f"Student model loaded. Parameters: {sum(p.numel() for p in student.parameters()):,}")

#Configure the teachers
teachers = []
for w_path in teacher_weights:
    
    ckpt = torch.load(w_path, map_location="cpu")   # store the teachers on CPU to save GPU memory
    teacher = RTDETRDetectionModel(cfg='rtdetr-l.yaml').load(ckpt)
    teacher.nc = 10
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teachers.append(teacher)
print(f"Loaded {len(teachers)} teacher models")

#KNOWLEDGE DISILLATION: CREATE HOOKS BETWEEN TEACHER HIDDEN LAYERS AND STUDENT HIDDEN CONVOLUTIONAL LAYERS - after warmup
def register_hooks(model, n_layers=3):
    """Register hooks on the last n convolutional layers, as student depth and teacher depth are different - ie. student hooks onto the last few layers where output is well represented"""
    #get conv layers
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    convs = convs[-n_layers:]
    feats = []

    def hook(module, input, output):
        feats.append(output)

    #create the hook
    handles = [c.register_forward_hook(hook) for c in convs]
    return feats, handles

student_feats = []
student_handles = []
teacher_feats_list = []
teacher_handles_list = []

#KNOWLEDGE DISTILLATION LOSS
def kd_feature_loss(student_feats, teacher_feats_list):

    loss = 0.0
    n_layers = len(student_feats)

    #here we want to compare the student layer with the last few teacher layers, and get the loss as the student model vs teacher model it best aligns with:
    for layer_idx in range(n_layers):
        s = student_feats[layer_idx]          #student data stored as (B, C_s, H, W) in tensor notation - batch, channels, height, width of image or intermediate representation

        #teacher data  (list of (B, C_t, H, W) )
        t_candidates = [t_feats[layer_idx] for t_feats in teacher_feats_list
                        if len(t_feats) > layer_idx]
        if not t_candidates:
            continue

        # ---- spatial alignment ----
        t_candidates = [
            F.interpolate(t, size=s.shape[2:], mode="bilinear", align_corners=False)
            for t in t_candidates
        ]

        # ---- channel alignment ----
        c_min = min(s.shape[1], min(t.shape[1] for t in t_candidates))
        s = s[:, :c_min]
        t_candidates = [t[:, :c_min] for t in t_candidates]

        #Find which teacher is the most similar/lowest loss
        # flatten spatial → vector per sample
        B = s.shape[0]
        s_vec = s.view(B, -1)                      # (B, C·H·W), multiply to get total number of pixels

        # compute squared L2 distance to every teacher
        dist = torch.stack(
            #.view() flattens hidden layers to a 1D representation, We use the difference of 1D representations to calculate KD loss
            [(t.view(B, -1) - s_vec).pow(2).mean(dim=1)
             for t in t_candidates],
            dim=1
        )                                           

        # index of closest teacher for each sample
        #select lowest L2 loss
        closest_idx = dist.argmin(dim=1)           

        # build the chosen teacher tensor (gradient flows only through selected)
        t_closest = torch.stack(
            [t_candidates[idx][b] for b, idx in enumerate(closest_idx)],
            dim=0
        )                                           # same shape as s

        #MSE to most similar teacher only
        loss += F.mse_loss(s, t_closest) #mse loss from torch.nn.functional

    return loss / max(n_layers, 1)

# Dataset & Dataloader - used in importing and processing iamges and labels
def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)

    cls_list = []
    bboxes_list = []
    batch_idx_list = []

    for i, lbl in enumerate(labels):
        if lbl.numel() == 0:
            continue
        cls_list.append(lbl[:, 0].long())
        bboxes_list.append(lbl[:, 1:5])
        batch_idx_list.append(torch.full((lbl.shape[0],), i, dtype=torch.long))

    if cls_list:
        cls_tensor = torch.cat(cls_list)
        bboxes_tensor = torch.cat(bboxes_list)
        batch_idx_tensor = torch.cat(batch_idx_list)
    else:
        cls_tensor = torch.tensor([], dtype=torch.long)
        bboxes_tensor = torch.tensor([], dtype=torch.float32)
        batch_idx_tensor = torch.tensor([], dtype=torch.long)

    return {
        "img": imgs,
        "label": list(labels),
        "batch_idx": batch_idx_tensor,
        "cls": cls_tensor,
        "bboxes": bboxes_tensor
    }

train_dataset = OverheadVehicleDataset(data_yaml, split="train", imgsz=imgsz)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)

print(f"Dataset size: {len(train_dataset)} images")
print(f"Batches per epoch: {len(train_loader)}")

#AdamW optimization
optimizer = torch.optim.AdamW(student.parameters(), lr=initial_lr, weight_decay=weight_decay)

warmup_epochs = 20

# Learning rate scheduler with warmup - defines rate in which the learning rate grows towards its targeted value
def warmup_lr_scheduler(optimizer, epoch, initial_lr, target_lr, warmup_epochs):
    if epoch < warmup_epochs:
        lr = initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = target_lr
    return lr

# Learning rate scheduler - Using ReduceLROnPlateau for stability
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=8, verbose=True, min_lr=1e-6
)

#Training Loop
best_loss = float('inf')
patience = 25
patience_counter = 0

#Print losses for debugging
loss_history = {
    'total': [],
    'detection': [],
    'kd_feature': []
}

#step up knowledge distillation weights from 0 to desired
def get_kd_weight(epoch, warmup_epochs, initial_weight, final_weight):
    """Gradually increase KD weight after warmup"""
    if epoch < warmup_epochs:
        return initial_weight
    else:
        # Linear ramp from initial to final
        progress = min(1.0, (epoch - warmup_epochs) / (epochs - warmup_epochs))
        return initial_weight + progress * (final_weight - initial_weight)
    
try:
    clear_cuda()

    for epoch in range(epochs):
        #call fxn for learning rate (low at start)
        lr = warmup_lr_scheduler(optimizer, epoch, initial_lr, target_lr, warmup_epochs)

        print(f"Epoch {epoch+1}/{epochs}, Learning Rate: {lr:.6f}")

        #put student model in training mode
        student.train()
        epoch_loss = 0.0
        epoch_det_loss = 0.0
        epoch_kd_loss = 0.0
        
        #call fxn for knowledge dist weight (0 at start)
        current_kd_weight = get_kd_weight(
            epoch, kd_warmup_epochs, kd_feat_weight_initial, kd_feat_weight_final
        )
        
        #after warmup epochs, enable knowledge distillation by setting up hooks
        if epoch == kd_warmup_epochs and len(student_handles) == 0:
            print("\n" + "="*60)
            print("ENABLING KNOWLEDGE DISTILLATION")
            print("="*60 + "\n")
            student_feats, student_handles = register_hooks(student, n_layers=3)
            for t in teachers:
                feats, handles = register_hooks(t, n_layers=3)
                teacher_feats_list.append(feats)
                teacher_handles_list.append(handles)

        #loop through all image batches in the dataset, using device as GPU
        for batch_idx, batch in enumerate(train_loader):
            imgs = batch["img"].to(device)
            
            # Move all tensors to device
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) 
                    for k, v in batch.items()}

            # Clear feature lists
            if len(student_feats) > 0:
                student_feats.clear()
                for feats in teacher_feats_list:
                    feats.clear()

            # Forward pass - student
            student_out = student(imgs)

            #get cross ent loss, l1, gious loss
            loss_student, loss_items = student.loss(batch, student_out)

            # Print out loss diagnostics every epoch.
            if batch_idx == 0:                   
                

                print('\n----- LOSS COMPONENTS (epoch {}) -----'.format(epoch+1))
                print('cls (focal) :', loss_items[0].item())
                print('box L1      :', loss_items[1].item())
                print('box GIoU    :', loss_items[2].item())
                print('-------------------------------------\n')

            #Calculate and implement KD Loss
            loss_kd = torch.tensor(0.0, device=device)
            if epoch >= kd_warmup_epochs and current_kd_weight > 0:
                
                for t in teachers:
                    t = t.to(device)          #transfer one teacher on GPU
                    with torch.no_grad():
                        _ = t(imgs)
                    t = t.cpu()               #immediately transfer back to CPU to save memory
                
                #KD Loss = kd weight * function output for KD loss
                if len(student_feats) > 0 and any(len(f) > 0 for f in teacher_feats_list):
                    loss_kd = current_kd_weight * kd_feature_loss(student_feats, teacher_feats_list)

            #Define total_loss (used in stepping weights) as sum of detection and kd loss
            total_loss = loss_student + loss_kd

            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0) #prevent exploding gradients
            optimizer.step()

            # Accumulate losses
            epoch_loss += total_loss.item()
            epoch_det_loss += loss_student.item()
            epoch_kd_loss += loss_kd.item()

            #Show loss diagnostics every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}: "
                      f"Total={total_loss.item():.4f}, "
                      f"Det={loss_student.item():.4f}, "
                      f"KD={loss_kd.item():.4f} (weight={current_kd_weight:.4f})")

        # Epoch averages
        avg_loss = epoch_loss / len(train_loader)
        avg_det = epoch_det_loss / len(train_loader)
        avg_kd = epoch_kd_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Loss_history for plotting
        loss_history['total'].append(avg_loss)
        loss_history['detection'].append(avg_det)
        loss_history['kd_feature'].append(avg_kd)
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")
        print(f"  Total Loss:     {avg_loss:.4f}")
        print(f"  Detection Loss: {avg_det:.4f}")
        print(f"  KD Loss:        {avg_kd:.4f}")
        print(f"  KD Weight:      {current_kd_weight:.4f}")
        print(f"  Learning Rate:  {current_lr:.6f}")
        
        #Step weights based on total loss
        scheduler.step(avg_loss)

        #Save best model based on DETECTION loss
        if avg_det < best_loss:
            best_loss = avg_det
            patience_counter = 0
            for h in student_handles:
                h.remove()
            for handles in teacher_handles_list:
                for h in handles:
                    h.remove()
            torch.save(student.model, "student_best_yolo_full.pt")

            
            print(f"  ✓ Saved new best model (detection loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")

        #Early stopping checkpoint for convergence
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            for h in student_handles:
                h.remove()
            for handles in teacher_handles_list:
                for h in handles:
                    h.remove()
            torch.save(student.model, f"student_checkpoint_epoch{epoch+1}.pt")
            print(f"  Checkpoint saved: student_checkpoint_epoch{epoch+1}.pt")

        # Warning if detection loss is too high
        if epoch > kd_warmup_epochs and avg_det > 80:
            print(f"\n  ⚠ WARNING: Detection loss is high ({avg_det:.2f})")
            print(f"  The model may not be learning to detect properly!")

except KeyboardInterrupt:
    print("\nTraining interrupted. Saving model...")
    for h in student_handles:
        h.remove()
    for handles in teacher_handles_list:
        for h in handles:
            h.remove()
    torch.save(student.model, "student_final_yolo_full.pt")

#Remove hooks before saving, otherwise it will crash training and corrupt file
for h in student_handles:
    h.remove()
for handles in teacher_handles_list:
    for h in handles:
        h.remove()

print("\n" + "="*60)
print("Training completed!")
print("="*60)
print(f"Best detection loss achieved: {best_loss:.4f}")
print(f"Final total loss: {avg_loss:.4f}")

#Save as student_final_yolo_full.pt
for h in student_handles:
    h.remove()
for handles in teacher_handles_list:
    for h in handles:
        h.remove()
torch.save(student.model, "student_final_yolo_full.pt")

print("\nSaved models:")
print("  - student_best.pt (best detection loss)")
print("  - student_final.pt (final epoch)")