import os
import glob
import yaml
import shutil
from pathlib import Path

#Remove label files where the images are deleted
def remove_orphaned_labels(dataset_root):
    removed_count = 0
    
    for split in ['train', 'valid', 'test']:
        label_dir = os.path.join(dataset_root, split, 'labels')
        image_dir = os.path.join(dataset_root, split, 'images')
        
        if not os.path.exists(label_dir):
            continue
        
        print(f"\n🔍 Checking for orphaned labels in {split}/labels...")
        
        label_files = glob.glob(os.path.join(label_dir, '*.txt'))
        
        for label_path in label_files:
            label_name = os.path.basename(label_path)
            image_found = False
            
            # Try common image extensions
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                img_path = os.path.join(image_dir, label_name.replace('.txt', ext))
                if os.path.exists(img_path):
                    image_found = True
                    break
            
            # Remove orphaned label file
            if not image_found:
                os.remove(label_path)
                removed_count += 1
                print(f"  Removed orphaned label: {label_name}")
    
    return removed_count

#Convert multi class dataset to single class detection
def convert_labels_to_binary(dataset_root, backup=True):
    
    # Find all label directories
    label_dirs = [
        os.path.join(dataset_root, 'train', 'labels'),
        os.path.join(dataset_root, 'valid', 'labels'),
        os.path.join(dataset_root, 'test', 'labels'),
    ]
    
    stats = {
        'total_files': 0,
        'files_modified': 0,
        'oranges_kept': 0,
        'apples_kept': 0,
        'other_removed': 0,
        'empty_files': 0,
    }
    
    for label_dir in label_dirs:
        if not os.path.exists(label_dir):
            print(f"Directory not found: {label_dir}")
            continue
            
        print(f"\nProcessing: {label_dir}")
        
        # Backup original labels
        if backup:
            backup_dir = label_dir + '_backup'
            if not os.path.exists(backup_dir):
                shutil.copytree(label_dir, backup_dir)
                print(f"  Backup created: {backup_dir}")
        
        # Process each label file
        label_files = glob.glob(os.path.join(label_dir, '*.txt'))
        
        for label_path in label_files:
            stats['total_files'] += 1
            new_lines = []
            file_modified = False
            
            # Read and filter labels
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    class_id = int(parts[0])
                    
                    # Keep Apple (class 0)
                    if class_id == 0:
                        new_lines.append(line)
                        stats['apples_kept'] += 1
                    
                    # Keep orange (class 4) but change to class 1
                    elif class_id == 1:
                        parts[0] = '0'
                        new_lines.append(' '.join(parts) + '\n')
                        stats['oranges_kept'] += 1
                        file_modified = True
                    
                    # Remove all other classes
                    else:
                        stats['other_removed'] += 1
                        file_modified = True
            
            # Write back to file
            if file_modified or len(new_lines) == 0:
                stats['files_modified'] += 1
                
            with open(label_path, 'w') as f:
                f.writelines(new_lines)
            
            # Track empty files
            if len(new_lines) == 0:
                stats['empty_files'] += 1
    
    # Update data.yaml
    data_yaml_path = os.path.join(dataset_root, 'data.yaml')
    
    if os.path.exists(data_yaml_path):
        # Backup original
        if backup:
            shutil.copy(data_yaml_path, data_yaml_path + '.backup')
        
        # Update with new class info
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        data['nc'] = 2
        data['names'] = ['apple', 'orange']
        
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        print(f"\n Updated data.yaml:")
        print(f"  nc: 2")
        print(f"  names: ['apple', 'orange']")
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Total files processed: {stats['total_files']}")
    print(f"Files modified: {stats['files_modified']}")
    print(f"Oranges kept: {stats['oranges_kept']}")
    print(f"Apples kept: {stats['apples_kept']}")
    print(f"Other classes removed: {stats['other_removed']}")
    print(f"Empty files created: {stats['empty_files']}")
    
    if stats['empty_files'] > 0:
        print(f"\n WARNING: {stats['empty_files']} files have no labels!")
        print(f"   These images contain only non-apple/orange fruits.")
        if stats['empty_files'] > 250:
            print(f"   Will keep maximum 300 empty files and remove the rest.")
        else:
            print(f"   All empty files will be kept (under 300 limit).")
    
    return stats

def remove_excess_empty_files(dataset_root, max_empty=250):
    """
    Remove empty label files and their images, keeping at most max_empty files.
    """
    # Collect all empty files across splits
    empty_files = []
    
    for split in ['train', 'valid', 'test']:
        label_dir = os.path.join(dataset_root, split, 'labels')
        image_dir = os.path.join(dataset_root, split, 'images')
        
        if not os.path.exists(label_dir):
            continue
        
        label_files = glob.glob(os.path.join(label_dir, '*.txt'))
        
        for label_path in label_files:
            # Check if file is empty
            if os.path.getsize(label_path) == 0:
                # Find corresponding image
                label_name = os.path.basename(label_path)
                image_path = None
                
                # Try common image extensions
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    img_path = os.path.join(image_dir, label_name.replace('.txt', ext))
                    if os.path.exists(img_path):
                        image_path = img_path
                        break
                
                empty_files.append({
                    'label': label_path,
                    'image': image_path,
                    'split': split
                })
    
    total_empty = len(empty_files)
    
    if total_empty <= max_empty:
        print(f"\n Found {total_empty} empty files - all will be kept (under {max_empty} limit)")
        return 0
    
    # Remove excess files (keep first max_empty, remove the rest)
    files_to_remove = empty_files[max_empty:]
    removed_count = 0
    
    print(f"\nRemoving {len(files_to_remove)} excess empty files (keeping {max_empty})...")
    
    for file_info in files_to_remove:
        # Remove image if it exists
        if file_info['image'] and os.path.exists(file_info['image']):
            os.remove(file_info['image'])
            print(f"  Removed image: {os.path.basename(file_info['image'])} ({file_info['split']})")
        
        # Remove label file
        if os.path.exists(file_info['label']):
            os.remove(file_info['label'])
            removed_count += 1
    
    print(f"\n Removed {removed_count} empty label files and their images")
    print(f" Kept {max_empty} empty files in the dataset")
    return removed_count


if __name__ == "__main__":
    # UPDATE THIS PATH to your dataset root directory
    dataset_root = "/home/eeel126/Desktop/ENEL645Project/datasets/fruit2/appleorange"  # e.g., "/home/user/datasets/fruits"
    
    print("="*60)
    print("FRUIT DATASET CONVERTER: 7-class → Binary (Apple/Orange)")
    print("="*60)
    
    # Step 0: Remove orphaned labels (labels without images)
    print("\n" + "="*60)
    print("STEP 1: REMOVING ORPHANED LABELS")
    print("="*60)
    orphaned_count = remove_orphaned_labels(dataset_root)
    if orphaned_count > 0:
        print(f"\n Removed {orphaned_count} orphaned label files")
    else:
        print(f"\n No orphaned labels found")
    
    # Step 1: Convert labels
    print("\n" + "="*60)
    print("STEP 2: CONVERTING LABELS TO BINARY")
    print("="*60)
    stats = convert_labels_to_binary(dataset_root, backup=True)
    
    # Step 2: Remove excess empty files (keeping max 300)
    if stats['empty_files'] > 250:
        print("\n" + "="*60)
        response = input(f"Remove {stats['empty_files'] - 250} excess empty files (keep 300)? (y/n): ")
        if response.lower() == 'y':
            remove_excess_empty_files(dataset_root, max_empty=250)
    elif stats['empty_files'] > 0:
        print(f"\n {stats['empty_files']} empty files found - all kept (under 300 limit)")

    print(f"Backups saved with '_backup' suffix")
    print(f"You can now train with: data.yaml")