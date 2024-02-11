from django.conf import settings
from django.shortcuts import render
from django.views.decorators.clickjacking import xframe_options_sameorigin
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.views import View
from django.views.generic.edit import FormView
from django.contrib import messages
from django.views.generic import TemplateView
from django.urls import reverse
from urllib.parse import urlencode
import traceback


from .estimate import estimate_coverage, calculate_ep_config
from .forms import ShapefileForm


import os
import shutil

import json
import geopandas as gpd
from shapely.geometry import Point


@xframe_options_sameorigin
def index(request):
    vcub_files = os.listdir(os.path.join(settings.MEDIA_ROOT, 'saved/vcub_config'))
    ep_files = os.listdir(os.path.join(settings.MEDIA_ROOT, 'saved/ep_config'))
    return render(request, 'index.html', {'title': 'Faire une estimation', 'vcub_files': vcub_files, 'ep_files': ep_files})

def view_estimation(request):
    vcub_files = os.listdir(os.path.join(settings.MEDIA_ROOT, 'saved/estimations'))
    return render(request, 'view_estimation.html', {'title': 'Visualiser une estimation', 'vcub_files': vcub_files})

def compare(request):
    return render(request, 'compare.html', {'title': 'Comparer des estimations'})

def config_vcub(request):
    vcub_files = os.listdir(os.path.join(settings.MEDIA_ROOT, 'saved/vcub_config'))
    return render(request, 'config_vcub.html', {'title': 'Créer une configuration VCUB', 'vcub_files': vcub_files})

def config_ep(request):
    return render(request, 'config_ep.html', {'title': 'Créer une configuration EP'})


class CopyDirectoryView(View):
    def post(self, request, *args, **kwargs):
        directory = request.POST.get('directory')
        destination = 'media/temp'
        destination_path = os.path.join(destination, os.path.basename(directory))
        try:
            print(f"Current working directory: {os.getcwd()}")
            print(f"Copying from {directory} to {destination_path}")
            if os.path.exists(destination_path):
                shutil.rmtree(destination_path)
            shutil.copytree(directory, destination_path)
            print(f"Successfully copied to {destination_path}")
            return JsonResponse({"status": "success"})
        except Exception as e:
            print(f"Error copying directory: {str(e)}")
            return JsonResponse({"status": "error", "message": str(e)})


def get_shapefile_data(request):
    directory = 'media/temp/' + request.GET.get('directory')
    searchTerm = request.GET.get('searchTerm')

    for file in os.listdir(directory):
        if file.endswith('.shp'):
            shapefile_path = os.path.join(directory, file)
            break
    else:
        return JsonResponse({'error': 'No shapefile found in the given directory'})

    gdf = gpd.read_file(shapefile_path)

    if searchTerm:
        gdf = gdf[gdf.apply(lambda row: row.astype(str).str.contains(searchTerm).any(), axis=1)]

    data = json.loads(gdf.to_json())

    return JsonResponse(data, safe=False)


@csrf_exempt
def check_folder_exists(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        filename = data.get('filename')
        directory = data.get('directory')
        if filename:
            file_path = os.path.join(directory, filename)
            folder_exists = os.path.exists(file_path)
            return JsonResponse({'folderExists': folder_exists})
        else:
            return JsonResponse({'status': 'failed', 'error': 'No filename provided'})
    else:
        return JsonResponse({'status': 'failed', 'error': 'Invalid request method'})


@csrf_exempt
def save_shapefile(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        features = []
        for feature in data['geojson']['features']:
            new_feature = {
                'id': feature['properties']['id'],
                'nom' : feature['properties']['nom'],
                'taille' : feature['properties']['taille'],
                'geometry': feature['geometry']
            }
            features.append(new_feature)
        gdf = gpd.GeoDataFrame(features)
        gdf['geometry'] = gdf.apply(lambda row: Point(row['geometry']['coordinates']), axis=1)
        gdf['id'] = gdf['id'].apply(int).copy()
        gdf['taille'] = gdf['taille'].apply(int).copy()
        gdf = gdf.set_crs(4326)

        directory = 'media/saved/vcub_config/' + data['filename']
        if not os.path.exists(directory):
            os.makedirs(directory)
            gdf.to_file(directory + '/' + data['filename'] + '.shp')
            return JsonResponse({'status': 'success'})
        else:
            return JsonResponse({'status': 'failed', 'error': 'Directory already exists'})
    else:
        return JsonResponse({'status': 'failed'})



class EstimateCoverage(FormView):
    template_name = 'index.html'
    form_class = ShapefileForm

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            self.vcub_config = form.cleaned_data['vcub_config']
            self.ep_config = form.cleaned_data['ep_config']
            return self.form_valid(form)
        else:
            return self.form_invalid(form)

    def form_valid(self, form):
        estimate_coverage(self.vcub_config, self.ep_config)
        return super().form_valid(form)

    def get_success_url(self):
        return reverse('view_estimation_pred') + '?vcub_config=' + self.vcub_config + '&ep_config=' + self.ep_config



class ViewEstimation(TemplateView):
    template_name = 'view_estimation.html'



@csrf_exempt
def calculate_ep_config_view(request):
    try:
        if request.method == 'POST':
            data = json.loads(request.body)
            filename = data['filename']
            ssthemes = data['ssthemes']
            calculate_ep_config(ssthemes, filename)
            return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()})
