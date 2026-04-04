"""
ml_model.py – R3D-18 model loading and video inference utilities.

Preprocessing pipeline and model architecture mirror those used during
training in pretrained.ipynb so that inference is identical.

Heavy ML imports (torch, torchvision, cv2) are deferred until first use so
that the Django process starts quickly and ``manage.py check`` works even when
these libraries are not yet installed in the current environment.
"""

import logging
import threading

logger = logging.getLogger(__name__)

# ── Preprocessing constants (Kinetics-400 statistics) ─────────────────────────
# Defined as plain Python tuples to avoid importing numpy at module load time.
R3D_MEAN = (0.43216, 0.394666, 0.37645)
R3D_STD = (0.22803, 0.22145, 0.216989)

SEQ_LEN = 16   # number of frames sampled per clip
IMG_SIZE = 112  # spatial resolution expected by R3D-18

CLASS_NAMES = {0: "Non-Shoplifter", 1: "Shoplifter"}

# ── Model construction ─────────────────────────────────────────────────────────

def _build_model(num_classes: int = 1):
    """Return an R3D-18 model with a custom binary-classification head."""
    import torch.nn as nn
    from torchvision.models.video import R3D_18_Weights, r3d_18

    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )
    return model


# ── Singleton model loader ─────────────────────────────────────────────────────

_model = None
_model_lock = threading.Lock()


def get_model(weights_path: str):
    """Load the R3D model once and cache it for the lifetime of the process.

    Parameters
    ----------
    weights_path:
        Path to the saved ``state_dict`` (``.pth`` file produced by training).
    """
    import torch

    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:  # double-checked locking
            return _model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_model(num_classes=1)

        try:
            state = torch.load(weights_path, map_location=device, weights_only=True)
            # Support both raw state_dict and checkpoint dicts
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
            logger.info("Loaded R3D weights from %s", weights_path)
        except FileNotFoundError:
            logger.warning(
                "Weights file not found at '%s'. "
                "Running with random weights – predictions will be meaningless. "
                "Place your trained Pretrained.pth at that path.",
                weights_path,
            )

        model.to(device)
        model.eval()
        _model = model

    return _model


# ── Video preprocessing ────────────────────────────────────────────────────────

def _uniform_frame_sampling(
    video_path: str,
    seq_len: int = SEQ_LEN,
    img_size: int = IMG_SIZE,
) -> "np.ndarray":
    """Return an array of *seq_len* uniformly sampled frames (H×W×C, uint8)."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"Video has no readable frames: {video_path}")

    indices = np.linspace(0, total_frames - 1, seq_len, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            # Repeat the last valid frame if seeking fails near EOF
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)

    cap.release()
    return np.stack(frames, axis=0)  # (T, H, W, C)


def preprocess_video(
    video_path: str,
    seq_len: int = SEQ_LEN,
    img_size: int = IMG_SIZE,
) -> "np.ndarray":
    """Preprocess a video file into a float32 array ready for R3D-18 inference.

    Returns
    -------
    np.ndarray of shape (C, T, H, W) with dtype float32.
    """
    import numpy as np

    mean = np.array(R3D_MEAN, dtype=np.float32)
    std = np.array(R3D_STD, dtype=np.float32)
    frames = _uniform_frame_sampling(video_path, seq_len=seq_len, img_size=img_size)
    frames = frames.astype(np.float32) / 255.0
    frames = (frames - mean) / std                 # normalise
    frames = frames.transpose(3, 0, 1, 2)          # (T,H,W,C) → (C,T,H,W)
    return frames


# ── Inference ──────────────────────────────────────────────────────────────────

def predict(video_path: str, weights_path: str) -> dict:
    """Run R3D-18 inference on a single video and return a result dict.

    Parameters
    ----------
    video_path:
        Absolute path to the video file on disk.
    weights_path:
        Path to the trained model weights file.

    Returns
    -------
    dict with keys:
        ``probability`` (float, 0-1) – probability of being a shoplifter,
        ``label``       (int, 0 or 1),
        ``class_name``  (str).
    """
    model = get_model(weights_path)
    device = next(model.parameters()).device

    import torch

    frames = preprocess_video(video_path)
    tensor = torch.tensor(frames).unsqueeze(0).to(device)  # (1, C, T, H, W)

    with torch.no_grad():
        logit = model(tensor)
        probability = torch.sigmoid(logit).item()

    label = 1 if probability > 0.5 else 0
    return {
        "probability": round(probability, 4),
        "label": label,
        "class_name": CLASS_NAMES[label],
    }
