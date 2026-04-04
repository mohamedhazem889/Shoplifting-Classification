from django.urls import path

from . import views

app_name = "classifier"

urlpatterns = [
    path("", views.upload_view, name="upload"),
    path("api/predict/", views.api_predict, name="api_predict"),
]
