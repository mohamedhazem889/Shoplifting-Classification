# Shoplifting Classification – R3D-18 Django Deployment

Binary video classification of shoplifting events using a fine-tuned
[R3D-18](https://pytorch.org/vision/stable/models/video_resnet.html) model,
served through a Django web application.

---

## Project Structure

```
Shoplifting-Classification/
├── manage.py                      # Django management script
├── requirements.txt               # Python dependencies
├── pretrained.ipynb               # Training notebook (R3D-18)
├── Model.ipynb                    # Training notebook (custom CNN)
├── Pretrained.pth                 # ← place your trained weights here
│
├── shoplifting_api/               # Django project package
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
└── classifier/                    # Django app
    ├── ml_model.py                # Model loading & inference
    ├── forms.py                   # Upload form with validation
    ├── views.py                   # Web + JSON API views
    ├── urls.py
    └── templates/classifier/
        ├── upload.html            # Upload form page
        └── result.html            # Prediction result page
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your trained model weights

Copy the saved weights file produced by `pretrained.ipynb`
(`Pretrained.pth`) to the **project root**:

```bash
cp /path/to/Pretrained.pth .
```

You can also point to a different path via the `MODEL_PATH` environment
variable (see *Configuration* below).

### 3. Run the development server

```bash
python manage.py runserver
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.

---

## Using the Web Interface

1. Navigate to `http://127.0.0.1:8000/`.
2. Upload a video file (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`).
3. The model will:
   - Sample **16 uniformly-spaced frames** at **112 × 112** px.
   - Normalise with Kinetics-400 mean/std.
   - Return **Shoplifter** or **Non-Shoplifter** with a confidence score.

---

## JSON API

### `POST /api/predict/`

Send a `multipart/form-data` request with a `video` field.

**Example with `curl`:**

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -F "video=@/path/to/clip.mp4"
```

**Response:**

```json
{
  "class_name": "Shoplifter",
  "label": 1,
  "probability": 0.8731
}
```

| Field | Type | Description |
|-------|------|-------------|
| `class_name` | string | `"Shoplifter"` or `"Non-Shoplifter"` |
| `label` | int | `1` = shoplifter, `0` = non-shoplifter |
| `probability` | float | Sigmoid output of the model (0 – 1) |

---

## Configuration

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `DJANGO_SECRET_KEY` | insecure default | Set a strong random key in production |
| `DJANGO_DEBUG` | `True` | Set to `False` in production |
| `DJANGO_ALLOWED_HOSTS` | `127.0.0.1 localhost` | Space-separated allowed hosts |
| `MODEL_PATH` | `Pretrained.pth` (project root) | Absolute path to `.pth` weights file |

---

## Production Deployment Notes

1. Set `DJANGO_DEBUG=False` and `DJANGO_SECRET_KEY` to a strong secret.
2. Collect static files: `python manage.py collectstatic`.
3. Serve with **Gunicorn** behind **Nginx**:
   ```bash
   gunicorn shoplifting_api.wsgi:application --workers 2
   ```
4. Configure Nginx to proxy requests and serve `staticfiles/` and `media/`.
5. GPU inference is used automatically when a CUDA device is available.
