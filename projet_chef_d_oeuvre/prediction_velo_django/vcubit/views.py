from django.conf import settings
from django.shortcuts import render
from django.views.decorators.clickjacking import xframe_options_sameorigin
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, FileResponse
from django.views import View
from django.views.generic.edit import FormView
from django.contrib import messages
from django.views.generic import TemplateView
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from urllib.parse import urlencode
import traceback

from .gdf_to_shp import gdf_to_shp_zip
from .estimate import estimate_coverage, calculate_ep_config
from .forms import ShapefileForm

import os
import shutil

import json
import geopandas as gpd
from shapely.geometry import Point




def home(request):
    return render(request, 'home.html', {'title': 'Accueil'})


def get_user_files(user, base_directory, estimation=False):
    user_directory = os.path.join(base_directory, user.username)
    files = []
    for directory in [base_directory + '/default' * estimation, user_directory]:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith('.zip') or file.endswith('.json') or estimation:
                    file_name = os.path.basename(file)
                    files.append({'path': '/' + os.path.join(directory, file_name), 'name': file_name})
    return files


@login_required
@xframe_options_sameorigin
def estimate_vcub(request):
    base_directory_vcub = 'media/saved/vcub_config'
    base_directory_ep = 'media/saved/ep_config'
    vcub_files = get_user_files(request.user, base_directory_vcub)
    ep_files = get_user_files(request.user, base_directory_ep)
    return render(request, 'estimate_vcub.html', {'title': 'Faire une estimation', 'vcub_files': vcub_files, 'ep_files': ep_files})

@login_required
def view_estimation(request):
    base_directory = 'media/saved/estimations'
    vcub_files = get_user_files(request.user, base_directory, estimation=True)
    return render(request, 'view_estimation.html', {'title': 'Visualiser une estimation', 'vcub_files': vcub_files})

@login_required
def compare(request):
    return render(request, 'compare.html', {'title': 'Comparer des estimations'})

@login_required
def config_vcub(request):
    base_directory = 'media/saved/vcub_config'
    vcub_files = get_user_files(request.user, base_directory)
    return render(request, 'config_vcub.html', {'title': 'Créer une configuration VCUB', 'vcub_files': vcub_files})

@login_required
def config_ep(request):
    return render(request, 'config_ep.html', {'title': 'Créer une configuration EP'})



@login_required
def get_shapefile_data(request):
    file_path = request.GET.get('file_path')
    if not os.path.exists(file_path):
        return JsonResponse({'error': 'File does not exist'})

    gdf = gpd.read_file(file_path)
    searchTerm = request.GET.get('searchTerm')
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
@login_required
def save_shapefile(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        features = []
        for feature in data['geojson']['features']:
            new_feature = {
                'nom' : feature['properties']['nom'],
                'taille' : feature['properties']['taille'],
                'geometry': feature['geometry']
            }
            features.append(new_feature)
        gdf = gpd.GeoDataFrame(features)
        gdf['geometry'] = gdf.apply(lambda row: Point(row['geometry']['coordinates']), axis=1)
        gdf['taille'] = gdf['taille'].apply(int).copy()
        gdf = gdf.set_crs(4326)

        directory = f'media/saved/vcub_config/{request.user.username}'
        name = data['filename']

        if not os.path.exists(f"{directory}/{name}.zip"):
            result = gdf_to_shp_zip(gdf, name, directory)
            if result != 0:
                return JsonResponse({'status': 'failed', 'error': 'An error occurred while saving the shapefile'})

        if data['download']:
            file_path = f"{directory}/{name}.zip"
            if os.path.exists(file_path):
                return FileResponse(open(file_path, 'rb'), as_attachment=True)
            else:
                return JsonResponse({'status': 'failed', 'error': 'File does not exist'})

        return JsonResponse({'status': 'success'})
    else:
        return JsonResponse({'status': 'failed'})



class EstimateCoverage(FormView):
    template_name = 'estimate_vcub.html'
    form_class = ShapefileForm

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        self.username = request.user.username
        if form.is_valid():
            self.vcub_config = form.cleaned_data['vcub_config']
            self.ep_config = form.cleaned_data['ep_config']
            return self.form_valid(form)
        else:
            return self.form_invalid(form)

    def form_valid(self, form):
        estimate_coverage(self.vcub_config[1:], self.ep_config[1:], self.username)
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
            os.makedirs(f'media/saved/ep_config/{request.user.username}', exist_ok=True)
            calculate_ep_config(ssthemes, filename, request.user.username)
            return JsonResponse({'status': 'success'})
    except Exception as e:
        print(e)
        return JsonResponse({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()})
