from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.inventory_list, name='inventory_list'),
    path('forecast/', include('forecast.urls')),
]