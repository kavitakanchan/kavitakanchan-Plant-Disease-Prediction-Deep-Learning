# This is a NEW FILE you must create.

from django.urls import path
from . import views

urlpatterns = [
    # This will be our main page
    path('', views.index, name='index'),
    
    # This will be our prediction API endpoint
    path('predict/', views.predict_disease, name='predict_disease'),
]
