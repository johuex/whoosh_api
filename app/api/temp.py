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
'''----------------------------------------'''
# ищем ближайшие узлы графа к какой-то выданной нам точке и просто получаем их индекс. индексы используются в расчете маршрутов и т.д.
def find_nearest_node(tree, gdf, point):
    closest_idx = tree.query([(point.y, point.x)], k=1, return_distance=False)[0]
    nearest_node = gdf.iloc[closest_idx].index.values[0]

    return nearest_node


#код расчета расстояния для каждой строки геодатафрейма относительно фиксированной точки расположения,которая подается в виде координат
def nx_distance_actual(row, a_location_gdf):
  node1=find_nearest_node(tree,gdf_nodes,row['geometry']) #здесь например датасет всех парковок
  node2=find_nearest_node(tree,gdf_nodes,a_location_gdf['geometry'][0]) #здесь наш actual location
  try:
    row['distance_to_location'] = nx.shortest_path_length(G, node1, node2, weight='length')
  except nx.NetworkXNoPath:
    row['distance_to_location'] = None
  return row['distance_to_location']


# функция, которая будет получать расстояние между точкой, которую мы дадим и между точкой центра города (Зарядье)
def nx_distance(point_a):
  node1=find_nearest_node(tree,gdf_nodes,point_a)
  node2=35884663
  distance = nx.shortest_path_length(G, node1, node2, weight='length')
  return distance

# Функция которая возвращает маршрут кратчайший между двумя точками
def osmnx_route(x1,y1,x2,y2):
  node1=find_nearest_node(tree,gdf_nodes,Point(x1,y1))
  node2=find_nearest_node(tree,gdf_nodes,Point(x2,y2))
  route = nx.shortest_path(G, node1, node2)


#для каждой строки геодатафрейма относительно фиксированной точки центра города - зарядье
def nx_distance_row(row):
  node1=find_nearest_node(tree,gdf_nodes,row['geometry']) #здесь может быть датасет всех парковок
  node2=35884663 #здесь может быть наш actual location
  try:
    row['distance_to_center'] = nx.shortest_path_length(G, node1, node2, weight='length')
  except nx.NetworkXNoPath:
    row['distance_to_center'] = None
  return row['distance_to_center']


'''--------------------------------------'''
#Записываем границы "Москвы"
lat_max = 55.927854
lat_min = 55.541984
lon_max = 37.883441
lon_min = 37.347858

G = ox.graph_from_bbox(lat_max, lat_min, lon_max, lon_min, network_type='drive')
#список узлов всего графа - будем использовать в расчетах
gdf_nodes = ox.graph_to_gdfs(G, edges=False)

'''РАСЧЕТ расстояние от выданной точки до велопарковок'''
bikestops_path = os.path.join(ROOT_DIR, 'other\\bikestops.csv')
parking_df = pd.read_csv(bikestops_path)
# делаем геодатафрейм чтобы использовать в расчете расстояний
parking_gdf = gpd.GeoDataFrame(
    parking_df, geometry=gpd.points_from_xy(parking_df.Longitude_WGS84, parking_df.Latitude_WGS84)).set_crs('epsg:4326')

# ищем ближайшие узлы графа к какой-то выданной нам точке и просто получаем их индекс. индексы используются в расчете маршрутов и т.д.
tree = KDTree(gdf_nodes[['y', 'x']], metric='euclidean')
# Производим расчет расстояний в датафрейме с парковками
parking_gdf['distance_to_location'] = parking_gdf.apply(nx_distance_actual, axis=1)

# допустим, что Зарядье - это за центр города
api = overpass.API(endpoint='https://overpass.kumi.systems/api/interpreter')
response = api.get('node["name"="Зарядье"]')
zaryadye = gpd.GeoDataFrame(
    geometry=[Point(response[0]['geometry']['coordinates'])],
    crs={'init':'epsg:4326'}
)

'''РАССТОЯНИЯ между всеми точками старта Whoosh и центром города (Зарядье)'''
ks_aggr_path = os.path.join(ROOT_DIR, 'other\\ks_aggregated.csv')
df = pd.read_csv(ks_aggr_path)
# убираем лишние остановки за пределами работы Whoosh
df = df.query('lat_x > @lat_min and lat_x < @lat_max and lon_x < @lon_max and lon_x > @lon_min')
# перевод в geodataframe
gdf_start = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.lon_x, df.lat_x)).set_crs('epsg:4326')
gdf_finish = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.lon_y, df.lat_y)).set_crs('epsg:4326')
'''находим для каждого датафрейма - стартов и финишей расстояние до центра. Сможем понять был ли пользователь ближе к центру на старте или в конце поездки'''
gdf_start['distance_to_center'] = gdf_start.apply(nx_distance_row, axis=1)
gdf_finish['distance_to_center'] = gdf_finish.apply(nx_distance_row, axis=1)
top_stations = parking_gdf.nsmallest(5, 'distance_to_location')
