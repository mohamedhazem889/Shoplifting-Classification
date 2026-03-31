"""
Inference module for the R3D-18 shoplifting classifier.

Loads the pretrained model once at startup and exposes a single
`predict_video(video_path)` function that returns a dict with:
  - label  : "Shoplifting" or "No Shoplifting"
  - confidence : float in [0, 1]
"""

import os
import threading
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

logger = logging.getLogger(__name__)

# ── Hyper-parameters that must match the training notebook ──────────────────
NUM_FRAMES = 16
FRAME_SIZE = 112
MEAN = [0.43216, 0.394666, 0.37645]
STD = [0.22803, 0.22145, 0.216989]
# ─────────────────────────────────────────────────────────────────────────────

_model = None
_device = None
_lock = threading.Lock()


def _build_model(weights_path: str, device: torch.device) -> nn.Module:
    """Build R3D-18 with the same head used during fine-tuning."""
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 1),
    )
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def get_model():
    """Return the singleton model, loading it on first call (thread-safe)."""
    global _model, _device
    if _model is None:
        with _lock:
            if _model is None:
                from django.conf import settings

                weights_path = settings.MODEL_WEIGHTS_PATH
                if not os.path.exists(weights_path):
                    raise FileNotFoundError(
                        f"Model weights not found at '{weights_path}'. "
                        "Please copy Pretrained.pth into shoplifting_django/model_weights/."
                    )
                _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info("Loading model from %s on %s", weights_path, _device)
                _model = _build_model(weights_path, _device)
                logger.info("Model loaded successfully.")
    return _model, _device


def _extract_frames(video_path: str) -> np.ndarray:
    """
    Uniformly sample NUM_FRAMES frames from the video.

    Returns an ndarray of shape (NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3)
    with pixel values in [0, 1] (float32).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError("Video has no readable frames.")

    indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            if not frames:
                cap.release()
                raise ValueError("Cannot read the first frame of the video. The file may be corrupted.")
            # Duplicate the last valid frame for subsequent missing frames
            frame = frames[-1].copy()
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        frames.append(frame)
    cap.release()

    frames = np.stack(frames).astype(np.float32) / 255.0  # (T, H, W, C)
    return frames


def _preprocess(frames: np.ndarray) -> torch.Tensor:
    """
    Normalise and reshape frames into a model-ready tensor.

    Input  : (T, H, W, C) float32 in [0, 1]
    Output : (1, C, T, H, W) float32 tensor
    """
    mean = np.array(MEAN, dtype=np.float32)
    std = np.array(STD, dtype=np.float32)
    frames = (frames - mean) / std                  # (T, H, W, C)
    frames = frames.transpose(3, 0, 1, 2)           # (C, T, H, W)
    tensor = torch.from_numpy(frames).unsqueeze(0)  # (1, C, T, H, W)
    return tensor


def predict_video(video_path: str) -> dict:
    """
    Run inference on a video file.

    Args:
        video_path: Absolute path to the uploaded video.

    Returns:
        {
            "label":      "Shoplifting" | "No Shoplifting",
            "confidence": float,          # probability of shoplifting
        }
    """
    model, device = get_model()

    frames = _extract_frames(video_path)
    tensor = _preprocess(frames).to(device)

    with torch.no_grad():
        logit = model(tensor)                       # (1, 1)
        prob = torch.sigmoid(logit).item()

    label = "Shoplifting" if prob >= 0.5 else "No Shoplifting"
    return {"label": label, "confidence": round(prob, 4)}
