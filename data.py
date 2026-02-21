def download_dataset():
    from roboflow import Roboflow
    
    rf = Roboflow(api_key="YKpidd2GHBAOlaqS2tq0")
    project = rf.workspace("my-images-data").project("fighter-jets-rxc4w-dek4n")
    version = project.version(1)
    dataset = version.download("yolo26")
    
    return dataset


import cv2
import numpy as np
import os


def load_image(image_path: str) -> np.ndarray:
    """تحميل الصورة من المسار"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"مش قادر يحمل الصورة من: {image_path}")
    return image


def resize_image(image: np.ndarray, target_size: tuple = (640, 640)) -> np.ndarray:
    """تغيير حجم الصورة"""
    return cv2.resize(image, target_size)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """تحويل قيم البيكسل من [0, 255] إلى [0, 1]"""
    return image.astype(np.float32) / 255.0


def preprocess_image(image_path: str, target_size: tuple = (640, 640)) -> np.ndarray:
    """
    Pipeline كامل للـ preprocessing على صورة واحدة
    1. Load
    2. Resize
    3. Normalize
    """
    image = load_image(image_path)
    image = resize_image(image, target_size)
    image = normalize_image(image)
    return image


def preprocess_batch(image_dir: str, target_size: tuple = (640, 640)) -> list:
    """
    تشغيل الـ preprocessing على فولدر كامل من الصور
    بيرجع list من الـ arrays
    """
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp")
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(supported_formats)
    ]

    if not image_paths:
        raise ValueError(f"مفيش صور في الفولدر: {image_dir}")

    processed_images = []
    for path in image_paths:
        try:
            processed = preprocess_image(path, target_size)
            processed_images.append({"path": path, "image": processed})
            print(f"✅ تم معالجة: {os.path.basename(path)}")
        except Exception as e:
            print(f"❌ خطأ في: {os.path.basename(path)} - {e}")

    return processed_images




                