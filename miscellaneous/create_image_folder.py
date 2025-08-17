import os
import shutil

def flatten_images(source_dir, target_dir):
    """
    Walks through all subfolders of source_dir, finds images,
    and copies them into target_dir.
    Skips files that already exist in target_dir.
    """
    os.makedirs(target_dir, exist_ok=True)

    count = 0
    skipped = 0
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                src_path = os.path.join(root, file)

                # destination path (keep same filename to detect duplicates)
                dst_path = os.path.join(target_dir, file)

                if os.path.exists(dst_path):
                    skipped += 1
                    print("skipping...")
                    continue  # don’t copy if already exists

                shutil.copy2(src_path, dst_path)
                count += 1
            else:
                print("⚠️ Skipped non-image:", file)

    print(f"✅ Copied {count} new images into {target_dir} (skipped {skipped} duplicates)")

# Example usage:
source = "raw image data"   # path to your nested folders
target = "data\images"           # path to where you want flattened images
flatten_images(source, target)
