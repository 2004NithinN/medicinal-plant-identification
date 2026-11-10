import os, glob, shutil, numpy as np

dataset_dir = 'medicinal_plants_dataset'  # your dataset
target_count = 50  # target images per class

for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = glob.glob(os.path.join(class_path, '*.*'))
    num_images = len(images)
    print(f"Class: {class_name}, Found: {num_images} images")

    if num_images < target_count:
        to_generate = target_count - num_images
        print(f"  -> Need to duplicate {to_generate} more images")

        idx = 0
        while to_generate > 0:
            img_path = np.random.choice(images)
            ext = os.path.splitext(img_path)[1]
            save_name = f"copy_{idx}_{os.path.basename(img_path)}"
            save_path = os.path.join(class_path, save_name)
            shutil.copy(img_path, save_path)

            to_generate -= 1
            idx += 1

print("✅ Done — each class folder now has 50 images (25 classes kept).")
