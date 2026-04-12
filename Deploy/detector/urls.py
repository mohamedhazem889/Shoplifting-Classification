from django.urls import path
from .views import PredictAPIView, IndexView, health

urlpatterns = [
    path('',               IndexView.as_view(),    name='index'),
    path('api/predict/',   PredictAPIView.as_view(), name='predict'),
    path('health/',        health,                  name='health'),
]
