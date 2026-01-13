from PIL import Image
from pillow_heif import register_heif_opener
import os
import shutil

# Register HEIF opener
register_heif_opener()

def convert_all_heic_to_new_folder(source_dataset, target_dataset):
    """
    Convert all HEIC files to JPG and save in new dataset folder
    Preserves folder structure
    Also copies existing JPG/PNG files to new folder
    """
    
    # Your gesture folders
    folders = ['none', 'paper', 'scissor', 'rock']
    
    total_converted = 0
    total_copied = 0
    total_errors = 0
    
    print("="*60)
    print("HEIC to JPG Batch Converter")
    print("="*60)
    print(f"Source: {source_dataset}")
    print(f"Target: {target_dataset}")
    print("="*60)
    
    # Create target dataset folder if it doesn't exist
    os.makedirs(target_dataset, exist_ok=True)
    
    for folder_name in folders:
        source_folder = os.path.join(source_dataset, folder_name)
        target_folder = os.path.join(target_dataset, folder_name)
        
        # Check if source folder exists
        if not os.path.exists(source_folder):
            print(f"\n⚠ Warning: Folder '{folder_name}' not found in source, skipping...")
            continue
        
        # Create target folder
        os.makedirs(target_folder, exist_ok=True)
        
        print(f"\n📁 Processing folder: {folder_name}")
        print("-" * 60)
        
        # Get all files in source folder
        files = os.listdir(source_folder)
        
        for filename in files:
            source_path = os.path.join(source_folder, filename)
            
            # Skip if not a file
            if not os.path.isfile(source_path):
                continue
            
            # Handle HEIC files - convert to JPG
            if filename.lower().endswith('.heic'):
                output_filename = os.path.splitext(filename)[0] + '.jpg'
                target_path = os.path.join(target_folder, output_filename)
                
                try:
                    # Open and convert HEIC
                    image = Image.open(source_path)
                    
                    # Convert to RGB if necessary
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Save as JPG
                    image.save(target_path, 'JPEG', quality=95)
                    
                    print(f"   ✓ Converted: {filename} → {output_filename}")
                    total_converted += 1
                    
                except Exception as e:
                    print(f"   ✗ Error converting {filename}: {e}")
                    total_errors += 1
            
            # Handle existing JPG/JPEG/PNG files - just copy them
            elif filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                target_path = os.path.join(target_folder, filename)
                
                try:
                    shutil.copy2(source_path, target_path)
                    print(f"   → Copied: {filename}")
                    total_copied += 1
                    
                except Exception as e:
                    print(f"   ✗ Error copying {filename}: {e}")
                    total_errors += 1
            
            else:
                # Skip other file types
                print(f"   ⊗ Skipped: {filename} (unsupported format)")
    
    # Summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"✓ HEIC converted to JPG: {total_converted} files")
    print(f"→ Existing images copied: {total_copied} files")
    print(f"✗ Errors: {total_errors} files")
    print(f"📊 Total images in new dataset: {total_converted + total_copied}")
    print("="*60)
    
    if total_converted + total_copied > 0:
        print(f"\n✓ All done! Your new dataset is ready at: {target_dataset}")
    else:
        print("\n⚠ No files were processed.")

# ============================================
# USAGE
# ============================================
if __name__ == "__main__":
    # Source dataset (original with HEIC files)
    source_dataset = "dataset"
    
    # Target dataset (new folder with JPG files)
    target_dataset = "dataset_new"
    
    # Run conversion
    convert_all_heic_to_new_folder(source_dataset, target_dataset)
    
    print("\n💡 Next steps:")
    print("1. Check the dataset_new folder to verify conversions")
    print("2. Update your config file to use 'dataset_new' as RAW_DATA_DIR")
    print("3. Run your training script!")