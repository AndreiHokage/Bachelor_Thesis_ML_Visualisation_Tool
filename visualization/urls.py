from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('upload/', views.upload_file, name="upload"),
    path('visualization/', views.visualization, name="visualization"),
    path('coords/', views.coords, name="coords"),
    path('augmenting/', views.augmenting_traffic_signs, name="augmenting")
]

if settings.DEBUG:
     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)