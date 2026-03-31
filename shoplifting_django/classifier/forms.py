from django import forms


ACCEPTED_TYPES = ["video/mp4", "video/avi", "video/mov", "video/quicktime", "video/x-msvideo"]
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100 MB


class VideoUploadForm(forms.Form):
    video = forms.FileField(
        label="Upload Video",
        help_text="Accepted formats: MP4, AVI, MOV (max 100 MB)",
    )

    def clean_video(self):
        video = self.cleaned_data["video"]
        if video.size > MAX_VIDEO_SIZE:
            raise forms.ValidationError(
                f"File size ({video.size // (1024 * 1024)} MB) exceeds the 100 MB limit."
            )
        content_type = getattr(video, "content_type", "")
        ext = video.name.rsplit(".", 1)[-1].lower() if "." in video.name else ""
        if content_type not in ACCEPTED_TYPES and ext not in ("mp4", "avi", "mov"):
            raise forms.ValidationError(
                "Unsupported file type. Please upload an MP4, AVI, or MOV video."
            )
        return video
