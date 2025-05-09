from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('forecast/', views.forecast, name='forecast'),
    path('inventory_list/', views.inventory_list, name='inventory_list'),
    path('inventory_list/add/', views.add_product, name='add_product'),
    path('inventory_list/edit/<int:pk>/', views.edit_product, name='edit_product'),
    path('inventory_list/delete/<int:pk>/', views.delete_product, name='delete_product'),
]