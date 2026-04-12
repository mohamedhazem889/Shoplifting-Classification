"""
ml_model.py  –  R3D-18 shoplifting classifier inference
Loaded once at startup via Django AppConfig.ready()
"""
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models.video as video_models
from torchvision.models.video import R3D_18_Weights

# ── Kinetics-400 normalisation stats used during training ──────────────────
R3D_MEAN = np.array([0.43216, 0.394666, 0.37645],  dtype=np.float32)
R3D_STD  = np.array([0.22803, 0.22145,  0.216989], dtype=np.float32)

CLASS_NAMES = ["non shop lifter", "shop lifter"]

_model  = None
_device = None


def build_r3d_model(num_classes: int = 1) -> nn.Module:
    """Recreate the same architecture used in training (head only)."""
    weights = R3D_18_Weights.KINETICS400_V1
    model   = video_models.r3d_18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_model(model_path: str) -> None:
    """Load weights from disk into the global singleton."""
    global _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model  = build_r3d_model()
    state   = torch.load(model_path, map_location=_device)
    _model.load_state_dict(state)
    _model.to(_device)
    _model.eval()
    print(f"[R3D] Model loaded on {_device}")


# ── Video pre-processing (identical to training) ───────────────────────────

def _uniform_frame_sampling(video_path: str, seq_len: int, img_size: int) -> np.ndarray:
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return np.zeros((seq_len, img_size, img_size, 3), dtype=np.float32)
    indices = np.linspace(0, total - 1, seq_len).astype(int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.float32)


def preprocess_video(video_path: str, seq_len: int = 16, img_size: int = 112) -> np.ndarray:
    frames  = _uniform_frame_sampling(video_path, seq_len, img_size)
    frames  = frames / 255.0
    frames  = (frames - R3D_MEAN) / R3D_STD   # normalise
    frames  = frames.transpose(3, 0, 1, 2)    # (T,H,W,C) → (C,T,H,W)
    return frames.astype(np.float32)


# ── Public inference function ───────────────────────────────────────────────

def predict(video_path: str, seq_len: int = 16, img_size: int = 112) -> dict:
    """
    Run inference on a single video file.

    Returns:
        {
            "probability":   float,   # P(shoplifter)
            "label":         int,     # 0 or 1
            "class_name":    str,
            "confidence_pct": float,  # percentage of winning class
        }
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    video  = preprocess_video(video_path, seq_len=seq_len, img_size=img_size)
    tensor = torch.tensor(video).unsqueeze(0).to(_device)   # (1,C,T,H,W)

    with torch.no_grad():
        prob = torch.sigmoid(_model(tensor)).item()

    label      = int(prob > 0.5)
    class_name = CLASS_NAMES[label]
    confidence = prob if label == 1 else 1.0 - prob

    return {
        "probability":    round(prob, 4),
        "label":          label,
        "class_name":     class_name,
        "confidence_pct": round(confidence * 100, 1),
    }
