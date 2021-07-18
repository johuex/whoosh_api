import networkx as nx
import osmnx as ox
ox.config(use_cache=True, log_console=True)
from config import ROOT_DIR
import os

import matplotlib
import matplotlib.pyplot as plt

import sklearn
from sklearn.neighbors import KDTree

import pandas as pd

import numpy as np

import geopandas as gpd

import overpass

import folium
from folium.plugins import MarkerCluster
from folium.plugins import HeatMap

from shapely.geometry import Point, LineString, MultiPoint, MultiLineString, MultiPolygon, Polygon
def best_parkings():

    """##Добываем московский граф"""

    #Записываем границы "Москвы"
    lat_max = 55.927854
    lat_min = 55.541984
    lon_max = 37.883441
    lon_min = 37.347858

    G = ox.graph_from_bbox(lat_max, lat_min, lon_max, lon_min, network_type='drive')


    ox.plot_raph(G, figsize=(20,20))
    plt.show()

    #список узлов всего графа - будем использовать в расчетах
    gdf_nodes = ox.graph_to_gdfs(G, edges=False)
    gdf_nodes.head()

    """## Расчет расстояния от выданной точки до велопарковок

    Загрузим датафрейм с парковками
    """

    parking_df = pd.read_csv('bikestops.csv')
    parking_df.head()

    """Сделаем из него геодатафрейм чтобы исползовать в расчете расстояний"""

    parking_gdf = gpd.GeoDataFrame(
        parking_df, geometry=gpd.points_from_xy(parking_df.Longitude_WGS84, parking_df.Latitude_WGS84)).set_crs('epsg:4326')
    parking_gdf.head()

    """Дальше пишем код для расчета расстояний"""

    #здесь мы ищем ближайшие узлы графа к какой-то выданной нам точке и просто получаем их индекс. индексы используются в расчете маршрутов и т.д.
    tree = KDTree(gdf_nodes[['y','x']], metric='euclidean')

    def find_nearest_node(tree, gdf, point):

      closest_idx = tree.query([(point.y, point.x)], k=1, return_distance=False)[0]
      nearest_node = gdf.iloc[closest_idx].index.values[0]

      return nearest_node

    #допустим у нас есть координаты актуального местоположения
    #подаем на вход координаты и превращаем их в геодатафрейм для использования в поиске узлов графа
    a_location_lon = 37.505559
    a_location_lat = 55.747303
    a_location_df = pd.DataFrame(
        {'act_lat': a_location_lat,
         'act_lon': a_location_lon}, index=[0])

    a_location_gdf = gpd.GeoDataFrame(
        a_location_df, geometry=gpd.points_from_xy(a_location_df.act_lon, a_location_df.act_lat)).set_crs('epsg:4326')

    #код расчета расстояния для каждой строки геодатафрейма относительно фиксированной точки расположения,которая подается в виде координат
    def nx_distance_actual(row):
      node1=find_nearest_node(tree,gdf_nodes,row['geometry']) #здесь например датасет всех парковок
      node2=find_nearest_node(tree,gdf_nodes,a_location_gdf['geometry'][0]) #здесь наш actual location
      try:
        row['distance_to_location'] = nx.shortest_path_length(G, node1, node2, weight='length')
      except nx.NetworkXNoPath:
        row['distance_to_location'] = None
      return row['distance_to_location']

    """Производим расчет расстояний в датафрейме с парковками"""

    # Commented out IPython magic to ensure Python compatibility.
    # %%time
    # parking_gdf['distance_to_location'] = parking_gdf.apply(nx_distance_actual, axis=1)

    parking_gdf.head()

    """##Поиграем с графом - поищем разные точки, ближайшие к ним узлы графа и пути между ними"""

    api = overpass.API(endpoint='https://overpass.kumi.systems/api/interpreter')
    response = api.get('node["name"="Эрмитаж"]')
    response

    hermitage=gpd.GeoDataFrame(
        geometry=[Point(response[0]['geometry']['coordinates'])],
        crs={'init':'epsg:4326'}
    )

    api = overpass.API(endpoint='https://overpass.kumi.systems/api/interpreter')
    response = api.get('node["name"="ВДНХ"]')
    response

    vdnh = gpd.GeoDataFrame(
        geometry=[Point(response[0]['geometry']['coordinates'])],
        crs={'init':'epsg:4326'}
    )

    api = overpass.API(endpoint='https://overpass.kumi.systems/api/interpreter')
    response = api.get('node["name"="Зарядье"]')
    response

    zaryadye = gpd.GeoDataFrame(
        geometry=[Point(response[0]['geometry']['coordinates'])],
        crs={'init':'epsg:4326'}
    )

    """Напишем функцию, которая будет получать расстояние между точкой, которую мы дадим и между точкой в Зарядье - примем это за центр города"""

    def nx_distance(point_a):
      node1=find_nearest_node(tree,gdf_nodes,point_a)
      node2=35884663
      #distance = ox.distance.euclidean_dist_vec(gdf_nodes[node1].y, gdf_nodes[node1].x, gdf_nodes[node2].y, gdf_nodes[node2].x)
      distance = nx.shortest_path_length(G, node1, node2, weight='length')
      return distance

    """Функция которая возвращает маршрут кратчайший между двумя точками"""

    def osmnx_route(x1,y1,x2,y2):
      node1=find_nearest_node(tree,gdf_nodes,Point(x1,y1))
      node2=find_nearest_node(tree,gdf_nodes,Point(x2,y2))
      route = nx.shortest_path(G, node1, node2)

    """Получаем здесь узлы рядом с нашими тестовыми точками"""

    nearest_vdnh = find_nearest_node(tree,gdf_nodes,vdnh.to_crs(epsg=4326)['geometry'][0])
    nearest_zaryadye = find_nearest_node(tree,gdf_nodes,zaryadye.to_crs(epsg=4326)['geometry'][0])
    nearest_hermitage = find_nearest_node(tree,gdf_nodes,hermitage.to_crs(epsg=4326)['geometry'][0])


    """## Найдем расстояния между всеми точками старта whoosh и центром города"""

    df = pd.read_csv('ks_aggregated.csv')

    """Имеет смысл смотреть на данные только в пределах "Москвы". Москвой можно считать вот эти координаты"""

    lat_max = 55.927854
    lat_min = 55.541984
    lon_max = 37.883441
    lon_min = 37.347858

    df.info()

    df.head(10)

    """Выкидываем лишнее через Query"""

    df = df.query('lat_x > @lat_min and lat_x < @lat_max and lon_x < @lon_max and lon_x > @lon_min')

    df.info()

    """Переводим в геодатафрейм"""

    gdf_start = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon_x, df.lat_x)).set_crs('epsg:4326')

    gdf_finish = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon_y, df.lat_y)).set_crs('epsg:4326')

    gdf_start.head()

    """Теперь находим для каждого датафрейма - стартов и финишей расстояние до центра. Сможем понять был ли пользователь ближе к центру на старте или в конце поездки"""

    #для простой пары точек
    def nx_distance(point_a):
      node1=find_nearest_node(tree,gdf_nodes,point_a)
      node2=nearest_zaryadye
      distance = nx.shortest_path_length(G, node1, node2, weight='length')
      return distance

    #для каждой строки геодатафрейма относительно фиксированной точки центра города - зарядье
    def nx_distance_row(row):
      node1=find_nearest_node(tree,gdf_nodes,row['geometry']) #здесь может быть датасет всех парковок
      node2=nearest_zaryadye #здесь может быть наш actual location
      try:
        row['distance_to_center'] = nx.shortest_path_length(G, node1, node2, weight='length')
      except nx.NetworkXNoPath:
        row['distance_to_center'] = None
      return row['distance_to_center']

    gdf_start['distance_to_center'] = gdf_start.apply(nx_distance_row, axis=1)

    gdf_finish['distance_to_center'] = gdf_finish.apply(nx_distance_row, axis=1)
    top_stations = parking_gdf.nsmallest(5, 'distance_to_location')
    return top_stations