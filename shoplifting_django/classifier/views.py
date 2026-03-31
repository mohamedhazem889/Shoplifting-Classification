import os
import uuid
import logging

from django.shortcuts import render, redirect
from django.conf import settings

from .forms import VideoUploadForm
from .inference import predict_video

logger = logging.getLogger(__name__)


def index(request):
    """Render the upload form (GET) or process the uploaded video (POST)."""
    if request.method == "POST":
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_file = form.cleaned_data["video"]

            # Save the upload to MEDIA_ROOT/uploads/<uuid>.<ext>
            ext = video_file.name.rsplit(".", 1)[-1].lower() if "." in video_file.name else "mp4"
            filename = f"{uuid.uuid4().hex}.{ext}"
            upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            save_path = os.path.join(upload_dir, filename)

            with open(save_path, "wb") as f:
                for chunk in video_file.chunks():
                    f.write(chunk)

            try:
                result = predict_video(save_path)
            except Exception as exc:
                logger.exception("Inference failed for %s", save_path)
                form.add_error(None, f"Inference error: {exc}")
                return render(request, "classifier/index.html", {"form": form})
            finally:
                # Clean up the uploaded file after inference
                if os.path.exists(save_path):
                    os.remove(save_path)

            return render(request, "classifier/result.html", {
                "label": result["label"],
                "confidence": result["confidence"],
                "confidence_pct": round(result["confidence"] * 100, 1),
            })
    else:
        form = VideoUploadForm()

    return render(request, "classifier/index.html", {"form": form})
