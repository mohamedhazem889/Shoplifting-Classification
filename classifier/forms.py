from django import forms


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class VideoUploadForm(forms.Form):
    video = forms.FileField(
        label="Select a video file",
        help_text=(
            f"Accepted formats: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}. "
            "Maximum size: 200 MB."
        ),
    )

    def clean_video(self):
        video = self.cleaned_data["video"]
        import os

        ext = os.path.splitext(video.name)[1].lower()
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise forms.ValidationError(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}."
            )
        return video
