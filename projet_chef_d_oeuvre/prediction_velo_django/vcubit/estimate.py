import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler
from math import pi, acos, sqrt
import json
import folium
from pygad import GA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from .gdf_to_shp import gdf_to_shp_zip


# Ajoute la densité de population de la maille dans laquelle se situe les objets de gdf
def add_density(gdf : gpd.GeoDataFrame, gdf_density : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    print("Ajout de la densité de population...", end='')
    densities = gpd.sjoin(gdf, gdf_density, how='left', predicate='within')['density']
    gdf_with_density = gdf.copy().assign(density=densities)
    print(" Terminé")
    return gdf_with_density


# Calcule l'influence du réseau vcub sur les mailles de gdf_density
def add_influence(gdf_density : gpd.GeoDataFrame,
                  gdf_vcub : gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    print("Calcul de l'influence des stations vcub sur les mailles...", end='')
    coeffs = gdf_vcub['taille'] / gdf_vcub['density']

    def influence(distances : pd.Series) -> pd.Series:
        return 2 * coeffs * (1 - 1 / (1 + np.exp(-(distances.astype(float)/1000)**2)))

    distances = pd.DataFrame(index=gdf_vcub.index, columns=gdf_density.index)
    for i, row in gdf_vcub.iterrows():
        point = row.geometry
        distances.loc[i] = gdf_density.apply(lambda x: point.distance(x.geometry.centroid), axis=1)
    influences = distances.apply(lambda row: influence(row), axis=0).sum(axis=0)
    scaler = MinMaxScaler()
    influences_scaled = scaler.fit_transform(influences.values.reshape(-1, 1)).flatten()
    gdf_with_influence = gdf_density.copy().assign(inf_vcub=influences_scaled)
    print(" Terminé")
    return gdf_with_influence


# Scale une column d'un DataFrame
def scale_column(gdf : gpd.GeoDataFrame, column : str) -> gpd.GeoDataFrame:
    print(f"Scaling de la colonne {column}...", end='')
    scaler = MinMaxScaler()
    column_scaled = scaler.fit_transform(gdf[column].values.reshape(-1, 1))
    gdf_with_column_scaled = gdf.copy().assign(**{f'{column}_s' : column_scaled})
    print(" Terminé")
    return gdf_with_column_scaled


# Calcule le coefficient multiplicatif du besoin des mailles
def calculate_coeffs(gdf_density : gpd.GeoDataFrame,
                     gdf_vcub : gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    print("Calcul du coefficient multiplicatif du besoin des mailles...", end='')
    x = (gdf_vcub['geometry'].x * gdf_vcub['taille']).sum() / gdf_vcub['taille'].sum()
    y = (gdf_vcub['geometry'].y * gdf_vcub['taille']).sum() / gdf_vcub['taille'].sum()
    center_of_gravity = Point(x, y)
    max_distance = gdf_vcub['geometry'].distance(center_of_gravity).max()

    def circle_overlap_proportion(d : float, r=max_distance) -> float:
        if d >= 2 * r: return 0
        elif d <= 0: return 1
        return (2 * r**2 * acos(d / (2 * r)) - d / 2 * sqrt(4 * r**2 - d**2)) / (pi * r**2)

    gdf_temp = gdf_density.copy()
    gdf_temp['centroid'] = gdf_temp.geometry.centroid
    gdf_temp['distance_center'] = gdf_temp['centroid'].distance(center_of_gravity)
    coeffs = gdf_temp['distance_center'].apply(circle_overlap_proportion)
    gdf_density_with_coeffs = gdf_density.copy().assign(coeff=coeffs)
    print(" Terminé")
    return gdf_density_with_coeffs


# Calcule coeff * density_s - inf_vcub_s
def calculate_diff(gdf_density : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    print("Calcul de la différence besoin - influence des mailles...", end='')
    diff = gdf_density['density_s'] * gdf_density['coeff'] - gdf_density['inf_vcub_s']
    gdf_density_with_diff = gdf_density.copy().assign(difference=diff)
    print(" Terminé")
    return gdf_density_with_diff


# Associe à chaque maille le nombre d'eqpub du subset par split
def eqpub_in_polygon(gdf_density : gpd.GeoDataFrame,
                     gdf_eqpub : gpd.GeoDataFrame,
                     subset, split='sstheme') -> gpd.GeoDataFrame:

    print("Association des équipements publics remarquables à leur maille...", end='')
    joined = gpd.sjoin(gdf_eqpub[gdf_eqpub[split].isin(subset)], gdf_density, predicate='within')
    counts = joined.groupby(['index_right', split]).size().reset_index(name='count')
    counts_pivoted = counts.pivot(index='index_right', columns=split, values='count').fillna(0).astype(int)
    counts_pivoted.columns = [f'{col}' for col in counts_pivoted.columns]
    gdf_density_with_eqpub = gdf_density.copy().join(counts_pivoted).fillna(0)
    print(" Terminé")
    return gdf_density_with_eqpub


# Calcule les poids associés aux catégories de eqpub sélectionnés
def calculate_weights(gdf_density : gpd.GeoDataFrame,
                      subset, num_generations=50, sol_per_pop=200,
                      num_parents_mating=50, parent_selection_type='sus',
                      crossover_type='uniform', crossover_probability=0.9,
                      mutation_type='adaptive', mutation_probability=[0.25, 0.05]) -> list:

    gdf_special = gdf_density[(gdf_density['difference'] < 0) & (gdf_density[subset].sum(axis=1) != 0)]
    threshold = gdf_special['difference'].describe()['50%']
    gdf_special = gdf_special[gdf_special['difference'] <= threshold]

    def fitness_func(ga_instance, solution, solution_idx) -> float:
        bonus = np.dot(gdf_special[subset], solution)
        score = (gdf_special['density_s'] + bonus) * gdf_special['coeff'] - gdf_special['inf_vcub_s']
        fitness = 1 / abs(score).mean()
        return fitness

    def on_generation(ga_instance):
        gen = ga_instance.generations_completed
        print(f'Progress: {round(100 * gen / num_generations)}%\n{gen}/{num_generations} generations completed', end='')
        if gen != num_generations: print('\033[F\033[F')

    print("--- Algorithme génétique en cours d'exécution ---")

    ga_instance = GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=len(subset),
        gene_type=float,
        init_range_low=0,
        init_range_high=1,
        parent_selection_type=parent_selection_type,
        crossover_type=crossover_type,
        crossover_probability=crossover_probability,
        mutation_type=mutation_type,
        mutation_probability=mutation_probability,
        gene_space={'low' : 0, 'high' : 1},
        on_generation=on_generation
    )

    ga_instance.run()

    print("\n--- Exécution terminée ---")

    return list(ga_instance.best_solution()[0])


# Calcule le score de chaque maille
def calculate_score(gdf_density : gpd.GeoDataFrame, subset, weigths) -> gpd.GeoDataFrame:
    # weights = calculate_weights(gdf_density)
    print("Calcul du score des mailles...", end='')
    bonus = np.dot(gdf_density[subset], weigths)
    scores = (gdf_density['density_s'] + bonus) * gdf_density['coeff'] - gdf_density['inf_vcub_s']
    gdf_density_with_score = gdf_density.copy().assign(score=scores)
    print(" Terminé")
    return gdf_density_with_score


def preprocessing(ssthemes, vcub_config='vcubDefault.zip'):

    gdf_vcub = gpd.read_file(f'media/saved/vcub_config/{vcub_config}').to_crs(2154)
    gdf_eqpub = gpd.read_file('media/base/eqpub').to_crs(2154)
    gdf_density = gpd.read_file('media/base/density').to_crs(2154)

    gdf_vcub = add_density(gdf_vcub, gdf_density)
    gdf_density = add_influence(gdf_density, gdf_vcub)
    gdf_density = scale_column(gdf_density, 'density')
    gdf_density = scale_column(gdf_density, 'inf_vcub')
    gdf_density = calculate_coeffs(gdf_density, gdf_vcub)
    gdf_density = calculate_diff(gdf_density)
    gdf_density = eqpub_in_polygon(gdf_density, gdf_eqpub, ssthemes)

    return gdf_density


def calculate_ep_config(ep_selection, config_name):

    nb_gen = 20 + 5 * len(ep_selection)

    gdf_density = preprocessing(ep_selection)
    weights = calculate_weights(gdf_density, ep_selection, num_generations=nb_gen)

    data = {
        'ssthemes': ep_selection,
        'weights': weights
    }

    with open(f'media/saved/ep_config/{config_name}.json', 'w') as json_file:
        json.dump(data, json_file)



def estimate_coverage(vcub_config, ep_config):

    ep_path = f'media/saved/ep_config/{ep_config}'
    with open(ep_path, 'r') as f:
        data = json.load(f)

    ssthemes = data['ssthemes']
    weights = data['weights']

    gdf_density = preprocessing(ssthemes, vcub_config=vcub_config)
    gdf_density = calculate_score(gdf_density, ssthemes, weights)

    name = f'{vcub_config[:-4]}_{ep_config[:-5]}'
    shp_folder_path = f'media/saved/estimations/{name}'

    gdf_to_shp_zip(gdf_density.to_crs(4326), name, shp_folder_path)

    bench_points = gpd.GeoDataFrame({'geometry': [Point(428000, 6423000), Point(428000, 6423000)], 'score': [-1, 1]}, crs=2154)
    gdf_density = pd.concat([gdf_density, bench_points], ignore_index=True)
    gdf_density['surface'] = gdf_density.to_crs(2154).geometry.area
    gdf_density_filtered = gdf_density[gdf_density.coeff > 0.4]
    champ_action = gpd.GeoDataFrame(geometry=[gdf_density_filtered.geometry.unary_union], crs=2154)

    m = gdf_density.explore(
        column="score",
        scheme="NaturalBreaks",
        k=10,
        cmap="viridis",
        legend=True,
        legend_kwds=dict(colorbar=False),
        tooltip=False,
        popup=["nom_iris", "nom_com", "density", "score"],
        name="density",
    )

    m = champ_action.explore(m=m, name="champ_action", style_kwds={
            'fillOpacity': 0,
            'color': 'black',
            'weight': 3,
            'interactive' : False,
        })

    m = gpd.read_file(f'media/saved/vcub_config/{vcub_config}').to_crs(2154).explore(m=m, name="vcub", color='red')

    folium.TileLayer("Stamen Toner").add_to(m)
    folium.LayerControl().add_to(m)

    m.save(f'{shp_folder_path}/{name}.html')

    # plt.figure()
    # sns.histplot(gdf_density_filtered.score, fill=True, kde=True)
    # plt.axvline(gdf_density_filtered.score.median(), color='red', label=f'médiane : {round(gdf_density_filtered.score.median(), 4)}')
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(f'{shp_folder_path}/hist.png')
    # plt.close()

    def calc_stats(mini, maxi):
        pct = 100 * gdf_density_filtered[(gdf_density_filtered.score > mini) & (gdf_density_filtered.score < maxi)].score.count() / len(gdf_density_filtered)
        return round(pct, 2)

    stats = {
        "tbon" : calc_stats(gdf_density_filtered.score.min(), 0),
        "bon" : calc_stats(0, 0.1),
        "abon" : calc_stats(0.1, 0.2),
        "moyen" : calc_stats(0.2, 0.3),
        "insuf" : calc_stats(0.3, 0.5),
        "tinsuf" : calc_stats(0.5, gdf_density_filtered.score.max()),
        "pctSurface" : round(100 * gdf_density_filtered.surface.sum() / gdf_density.surface.sum(), 2),
        "pctPop" : round(100 * (gdf_density_filtered.density * gdf_density_filtered.surface / 1E06).sum() / (gdf_density.density * gdf_density.surface / 1E06).sum(), 2),
        "medianScore" : round(gdf_density_filtered.score.median(), 4)
    }

    labels = ['Très bonne', 'Bonne', 'Assez bonne', 'Moyenne', 'Insuffisante', 'Très insuffisante']
    colors = ['green', 'limegreen', 'yellow', 'orange', 'red', 'darkred']
    ranges = ['score ≤ 0', '0 < score ≤ 0.1', '0.1 < score ≤ 0.2', '0.2 < score ≤ 0.3', '0.3 < score ≤ 0.5', 'score > 0.5']
    patches = [mpatches.Patch(color=colors[i], label=ranges[i]) for i in range(len(labels))]
    values = [stats[key] for key in list(stats.keys())[:6]]

    plt.figure(figsize=(6, 3))
    plt.barh(labels, values, color=colors)
    plt.xlim(0, max(50, max(values)))
    plt.xlabel('%')
    plt.legend(handles=patches)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{shp_folder_path}/bars.png')
    plt.close()

    with open(f'{shp_folder_path}/stats.json', 'w') as f:
        json.dump(stats, f)
