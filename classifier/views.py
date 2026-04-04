import logging
import os
import tempfile

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .forms import ALLOWED_VIDEO_EXTENSIONS, VideoUploadForm
from .ml_model import predict

logger = logging.getLogger(__name__)

# Map of allowed extensions to safe suffixes used for temp files.
# Using a fixed allowlist prevents path-injection via user-supplied filenames.
_SAFE_SUFFIX = {ext: ext for ext in ALLOWED_VIDEO_EXTENSIONS}


def _safe_suffix(filename: str) -> str:
    """Return a safe temp-file suffix for *filename*, falling back to '.tmp'."""
    ext = os.path.splitext(filename)[1].lower()
    return _SAFE_SUFFIX.get(ext, ".tmp")


@require_http_methods(["GET", "POST"])
def upload_view(request):
    """Render the video-upload form (GET) or process an uploaded video (POST).

    POST accepts both multipart/form-data (browser form) and returns an HTML
    result page.  For programmatic access use the ``/api/predict/`` endpoint.
    """
    form = VideoUploadForm()
    context = {"form": form}

    if request.method == "POST":
        form = VideoUploadForm(request.POST, request.FILES)
        if not form.is_valid():
            context["form"] = form
            return render(request, "classifier/upload.html", context)

        video_file = request.FILES["video"]

        # Save the upload to a temporary file so OpenCV can read it by path.
        # Use a sanitized extension from the allowlist to avoid path injection.
        suffix = _safe_suffix(video_file.name)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            for chunk in video_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            result = predict(tmp_path, settings.R3D_MODEL_PATH)
        except Exception:
            logger.exception("Prediction failed for file '%s'", video_file.name)
            context["form"] = form
            context["error"] = "Prediction failed. Please check the video file and try again."
            return render(request, "classifier/upload.html", context)
        finally:
            os.unlink(tmp_path)

        context = {
            "filename": video_file.name,
            "class_name": result["class_name"],
            "probability": result["probability"],
            "label": result["label"],
        }
        return render(request, "classifier/result.html", context)

    return render(request, "classifier/upload.html", context)


@require_http_methods(["POST"])
def api_predict(request):
    """JSON API endpoint for video classification.

    POST a multipart/form-data request with a ``video`` field.

    Returns
    -------
    JSON::

        {
            "class_name": "Shoplifter" | "Non-Shoplifter",
            "label": 0 | 1,
            "probability": 0.9123
        }
    """
    form = VideoUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return JsonResponse({"error": form.errors}, status=400)

    video_file = request.FILES["video"]
    suffix = _safe_suffix(video_file.name)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        for chunk in video_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        result = predict(tmp_path, settings.R3D_MODEL_PATH)
    except Exception:
        logger.exception("API prediction failed for file '%s'", video_file.name)
        return JsonResponse({"error": "Prediction failed. Please check the video file and try again."}, status=500)
    finally:
        os.unlink(tmp_path)

    return JsonResponse(result)
