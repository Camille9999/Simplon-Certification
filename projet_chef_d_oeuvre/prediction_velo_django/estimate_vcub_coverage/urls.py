from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("index/", views.index, name="index"),
    path('view_estimation/', views.view_estimation, name='view_estimation'),
    path('compare/', views.compare, name='compare'),
    path('config_vcub/', views.config_vcub, name='config_vcub'),
    path('config_ep/', views.config_ep, name='config_ep'),
    path('copy_directory/', views.CopyDirectoryView.as_view(), name='copy_directory'),
    path('get_shapefile_data/', views.get_shapefile_data, name='get_shapefile_data'),
    path('save_shapefile/', views.save_shapefile, name='save_shapefile'),
    path('check_folder_exists/', views.check_folder_exists, name='check_folder_exists'),
    path('estimate_coverage/', views.EstimateCoverage.as_view(), name='estimate_coverage'),
    path('view_estimation/', views.ViewEstimation.as_view(), name='view_estimation_pred'),
    path('calculate_ep_config/', views.calculate_ep_config_view, name='calculate_ep_config'),
]
