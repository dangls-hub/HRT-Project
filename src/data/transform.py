import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import src.config as config

class MorphologicalTransform(ImageOnlyTransform):
    """
    Biến đổi hình thái học tự định nghĩa cho Albumentations (Co nét / Giãn nét chữ).
    """
    def __init__(self, operation='erosion', kernel_size=2, p=0.5):
        super().__init__(p=p)
        self.operation = operation
        self.kernel_size = kernel_size

    def apply(self, img, **params):
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        if self.operation == 'erosion':
            return cv2.erode(img, kernel, iterations=1)
        elif self.operation == 'dilation':
            return cv2.dilate(img, kernel, iterations=1)
        return img

def get_train_transform() -> A.Compose:
    """
    Trả về pipeline augmentation cho training:
    - Xoay ngẫu nhiên ±15 độ (để học cách nhận diện chữ nghiêng)
    - Xiên nét (shear) ngẫu nhiên ±15 độ (xác suất 0.5)
    - Biến dạng đàn hồi ElasticTransform (để mô phỏng nét viết tay tự nhiên)
    - Blur nhẹ
    - Thêm Gaussian noise
    - Tăng giảm độ sáng/độ tương phản
    """
    return A.Compose([
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=255),
        A.Affine(shear=(-15, 15), p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=255),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3, border_mode=cv2.BORDER_CONSTANT, fill_value=255),
        A.GaussianBlur(blur_limit=3, sigma_limit=(0.1, 1.5), p=0.3),
        A.GaussNoise(std_range=(0.04, 0.12), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.OneOf([
            MorphologicalTransform(operation='erosion', kernel_size=2, p=1.0),
            MorphologicalTransform(operation='dilation', kernel_size=2, p=1.0),
        ], p=0.2),
    ])

def get_val_transform() -> A.Compose:
    """
    Trả về transform cho validation/test (không có augmentation).
    """
    return A.Compose([])

def resize_and_pad(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    1. Resize ảnh giữ nguyên tỉ lệ aspect ratio (scale theo height).
    2. Nếu chiều rộng mới nhỏ hơn target_w, pad bên phải bằng pixel trắng (255).
    3. Nếu chiều rộng mới lớn hơn target_w, resize ép chiều rộng về target_w để không mất thông tin.
    """
    h, w = image.shape[:2]
    
    # Tính toán scale factor theo height
    scale = target_h / h
    new_w = int(w * scale)
    
    if new_w == target_w:
        # Nếu vừa khớp, chỉ resize bình thường
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    elif new_w < target_w:
        # Resize giữ nguyên tỷ lệ
        resized = cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_AREA)
        # Pad thêm màu trắng về bên phải
        if len(image.shape) == 3:
            padded = np.full((target_h, target_w, image.shape[2]), 255, dtype=np.uint8)
            padded[:, :new_w, :] = resized
        else:
            padded = np.full((target_h, target_w), 255, dtype=np.uint8)
            padded[:, :new_w] = resized
        return padded
        
    else:
        # Nếu rộng hơn target_w, co ảnh lại theo target_w để giữ thông tin chữ viết
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

def to_tensor_and_normalize(image: np.ndarray) -> torch.Tensor:
    """
    1. Convert ảnh grayscale (1 channel) -> 3 channels (bằng cách nhân bản).
    2. Chuẩn hóa pixel [0, 255] -> [0, 1].
    3. Chuẩn hóa theo ImageNet mean/std.
    4. Trả về PyTorch tensor (C, H, W).
    """
    # Đảm bảo ảnh có 3 kênh
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = np.concatenate([image, image, image], axis=-1)
        
    # Scale [0, 255] -> [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Chuẩn hóa ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    
    # Chuyển đổi HWC -> CHW cho PyTorch
    image = image.transpose(2, 0, 1)
    
    return torch.tensor(image, dtype=torch.float32)