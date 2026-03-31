# Shoplifting Classifier – Django Deployment

A Django web application that serves the pretrained **R3D-18** shoplifting
classifier. Upload a surveillance video and get an instant prediction.

---

## Project Structure

```
shoplifting_django/
├── manage.py
├── requirements.txt
├── model_weights/
│   └── Pretrained.pth          ← place your model weights here
├── shoplifting_django/          ← Django project package
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── classifier/                  ← Django app
    ├── inference.py             ← model loading & prediction
    ├── forms.py
    ├── views.py
    ├── urls.py
    └── templates/classifier/
        ├── index.html           ← upload page
        └── result.html          ← prediction result page
```

---

## Prerequisites

- Python 3.9 or later
- (Recommended) a GPU with CUDA for faster inference

---

## Setup

### 1. Clone / enter the project folder

```bash
cd shoplifting_django
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users** – install the CUDA-enabled PyTorch wheel first, then the rest:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> pip install -r requirements.txt
> ```

### 4. Add the model weights

Copy your trained weights file into the `model_weights/` directory:

```bash
cp /path/to/Pretrained.pth model_weights/Pretrained.pth
```

The path can be overridden with the `MODEL_WEIGHTS_PATH` environment variable.

### 5. Apply migrations and run

```bash
python manage.py migrate
python manage.py runserver
```

Open <http://127.0.0.1:8000> in your browser.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DJANGO_SECRET_KEY` | insecure default | Set to a long random string in production |
| `DJANGO_DEBUG` | `True` | Set to `False` in production |
| `DJANGO_ALLOWED_HOSTS` | `127.0.0.1 localhost` | Space-separated list of allowed hosts |
| `MODEL_WEIGHTS_PATH` | `model_weights/Pretrained.pth` | Absolute path to model weights |

---

## How It Works

1. The user uploads an MP4/AVI/MOV video through the web form.
2. The server uniformly samples **16 frames** from the video using OpenCV.
3. Each frame is resized to **112 × 112** pixels and normalised with the
   Kinetics-400 mean/std values.
4. The tensor `(1, 3, 16, 112, 112)` is fed through the pretrained **R3D-18**
   model (fine-tuned for shoplifting detection).
5. The sigmoid output gives the shoplifting probability.
6. The result page shows the label and a confidence bar.

---

## Production Notes

- Replace the SQLite database with PostgreSQL for multi-user deployments.
- Serve static files with `whitenoise` or a dedicated CDN/Nginx.
- Use `gunicorn shoplifting_django.wsgi` behind a reverse proxy (Nginx/Caddy).
- Set `DJANGO_DEBUG=False` and configure a proper `SECRET_KEY`.
