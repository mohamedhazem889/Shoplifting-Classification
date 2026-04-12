import os
import uuid
import json

from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest
from django.views import View
from django.views.generic import TemplateView
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from .ml_model import predict


ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
MAX_UPLOAD_MB = 200


# ── Helpers ────────────────────────────────────────────────────────────────

def _save_upload(uploaded_file) -> str:
    """Save an uploaded video to MEDIA_ROOT/uploads/ and return its path."""
    ext      = os.path.splitext(uploaded_file.name)[1].lower()
    filename = f"{uuid.uuid4().hex}{ext}"
    dest     = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'wb') as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)
    return dest


def _validate_video(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        return f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_UPLOAD_MB} MB."
    return None


# ── REST API endpoint  POST /api/predict/ ─────────────────────────────────

@method_decorator(csrf_exempt, name='dispatch')
class PredictAPIView(View):
    """
    POST /api/predict/
    Content-Type: multipart/form-data
    Body:  video=<file>

    Returns JSON:
    {
        "success": true,
        "result": {
            "probability":    0.9231,
            "label":          1,
            "class_name":     "shop lifter",
            "confidence_pct": 92.3
        }
    }
    """

    def post(self, request):
        if 'video' not in request.FILES:
            return JsonResponse({'success': False, 'error': 'No video file provided. Use key "video".'}, status=400)

        video_file = request.FILES['video']
        error = _validate_video(video_file)
        if error:
            return JsonResponse({'success': False, 'error': error}, status=400)

        video_path = _save_upload(video_file)
        try:
            result = predict(
                video_path,
                seq_len=settings.SEQ_LEN,
                img_size=settings.IMG_SIZE,
            )
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
        finally:
            # Clean up uploaded file after inference
            if os.path.exists(video_path):
                os.remove(video_path)

        return JsonResponse({'success': True, 'result': result})

    def get(self, request):
        return JsonResponse({
            'info': 'POST a video file with key "video" to get a prediction.',
            'allowed_extensions': list(ALLOWED_VIDEO_EXTENSIONS),
            'max_upload_mb': MAX_UPLOAD_MB,
        })


# ── Simple web UI  GET / ───────────────────────────────────────────────────

class IndexView(TemplateView):
    template_name = 'detector/index.html'


# ── Health check  GET /health/ ─────────────────────────────────────────────

def health(request):
    from .ml_model import _model
    return JsonResponse({
        'status': 'ok',
        'model_loaded': _model is not None,
    })
