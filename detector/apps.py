from django.apps import AppConfig


class DetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'

    def ready(self):
        """Load the R3D model once when Django starts."""
        import os
        from django.conf import settings
        from .ml_model import load_model

        model_path = str(settings.MODEL_PATH)
        if os.path.exists(model_path):
            load_model(model_path)
        else:
            import warnings
            warnings.warn(
                f"[R3D] Model file not found at {model_path}. "
                "Place your trained weights there before running predictions."
            )
