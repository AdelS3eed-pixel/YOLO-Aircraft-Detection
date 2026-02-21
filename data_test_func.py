# data_test_func.py
from data import download_dataset, preprocess_image, preprocess_batch

# ============================================================
# Test 1: Download Dataset
# ============================================================
print("=" * 50)
print("📥 Test 1: Download Dataset")
print("=" * 50)

try:
    dataset = download_dataset()
    print(f"✅ Dataset downloaded successfully: {dataset.location}")
except Exception as e:
    print(f"❌ Download failed: {e}")

# ============================================================
# Test 2: Preprocess Single Image
# ============================================================
print("\n" + "=" * 50)
print("🖼️ Test 2: Preprocess Single Image")
print("=" * 50)

test_image_path = "fighter-jets-rxc4w-dek4n-1/train/images/your_image.jpg"  # ← غير المسار

try:
    image = preprocess_image(test_image_path)
    print(f"✅ Image preprocessed successfully!")
    print(f"   Shape  : {image.shape}")
    print(f"   Dtype  : {image.dtype}")
    print(f"   Min val: {image.min():.4f}")
    print(f"   Max val: {image.max():.4f}")
except Exception as e:
    print(f"❌ Single image preprocessing failed: {e}")

# ============================================================
# Test 3: Preprocess Batch
# ============================================================
print("\n" + "=" * 50)
print("📂 Test 3: Preprocess Batch")
print("=" * 50)

test_images_dir = "fighter-jets-rxc4w-dek4n-1/train/images"  # ← غير المسار لو محتاج

try:
    results = preprocess_batch(test_images_dir)
    print(f"\n✅ Batch done! عدد الصور المعالجة: {len(results)}")
    print(f"   Sample shape: {results[0]['image'].shape}")
    print(f"   Sample path : {results[0]['path']}")
except Exception as e:
    print(f"❌ Batch preprocessing failed: {e}")