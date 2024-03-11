from django.urls import path
from django.views.generic.base import RedirectView
from . import views

urlpatterns = [
    path("", RedirectView.as_view(url='/home/', permanent=True)),
    path("home/", views.home, name="home"),
    path("estimer/", views.estimate_vcub, name="estimate_vcub"),
    path('visualiser/', views.view_estimation, name='view_estimation'),
    path('comparer/', views.compare, name='compare'),
    path('configurer/vcub/', views.config_vcub, name='config_vcub'),
    path('configurer/ep/', views.config_ep, name='config_ep'),
    path('monitoring/', views.monitor, name='monitor'),
    path('get_shapefile_data/', views.get_shapefile_data, name='get_shapefile_data'),
    path('save_shapefile/', views.save_shapefile, name='save_shapefile'),
    path('check_folder_exists/', views.check_folder_exists, name='check_folder_exists'),
    path('estimate_coverage/', views.EstimateCoverage.as_view(), name='estimate_coverage'),
    path('visualiser/', views.ViewEstimation.as_view(), name='view_estimation_pred'),
    path('calculate_ep_config/', views.calculate_ep_config_view, name='calculate_ep_config'),
]
