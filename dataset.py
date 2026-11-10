from icrawler.builtin import GoogleImageCrawler
import os
import glob

# ---- 30 medicinal plants ----
plants = [
    "Aloe vera", "Neem", "Tulsi", "Ashwagandha", "Ginger", "Turmeric",
    "Peppermint", "Chamomile", "Eucalyptus", "Moringa", "Lavender",
    "Rosemary", "Sage", "Thyme", "Basil", "Cinnamon", "Garlic",
    "Hibiscus", "Fennel", "Gotu Kola", "Valerian", "Licorice",
    "Shankhpushpi", "Arjuna", "Ginkgo biloba", "Goldenseal", "Calendula",
    "St Johns Wort", "Elderberry", "Bay leaf"
]

# ---- Main dataset folder ----
dataset_dir = "medicinal_plants_dataset"
os.makedirs(dataset_dir, exist_ok=True)

# ---- Target number of images per plant ----
target_images = 50

# ---- Loop for each plant ----
for plant in plants:
    plant_name_clean = plant.replace(" ", "_")
    save_dir = os.path.join(dataset_dir, plant_name_clean)
    os.makedirs(save_dir, exist_ok=True)

    # count existing images in the folder
    existing_images = len(glob.glob(os.path.join(save_dir, "*")))
    missing_images = target_images - existing_images

    if missing_images <= 0:
        print(f"✅ {plant}: already has {existing_images} images, skipping.")
        continue

    print(f"🔄 Downloading {missing_images} more images for: {plant}")

    google_crawler = GoogleImageCrawler(storage={'root_dir': save_dir})

    google_crawler.crawl(
        keyword=f"{plant} plant leaves",  # More specific keyword
        max_num=missing_images,
        filters={
            'size': 'medium',  # good quality
            'type': 'photo'    # avoid clipart
        }
    )

    print(f"✅ Completed downloading for: {plant} (now has {target_images} images or more)")

print("\n🎉 All missing images downloaded!")
