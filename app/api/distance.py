import networkx as nx
import osmnx as ox
ox.config(use_cache=True, log_console=True)
import datetime as dt
from sklearn.neighbors import KDTree
import pandas as pd
import contextily as cx
import numpy as np
import geopandas as gpd
from libpysal.cg.alpha_shapes import alpha_shape_auto
import sys
from esda.adbscan import ADBSCAN, get_cluster_boundary, remap_lbls
import os
from config import ROOT_DIR, LINK, gdf_nodes, G, tree

'''---------------------------------------'''


def create_parking_dtf():
    '''Добываем московский граф'''
    # Границы "Москвы"
    lat_max = 55.927854
    lat_min = 55.541984
    lon_max = 37.883441
    lon_min = 37.347858
    G = ox.graph_from_bbox(lat_max, lat_min, lon_max, lon_min, network_type='drive')
    gdf_nodes = ox.graph_to_gdfs(G, edges=False)  # список узлов всего графа - будем использовать в расчетах

    '''добываем данные о погоде'''
    # открываем агрегированные данные по байкам
    ks_agg_path = os.path.join(ROOT_DIR, 'other/ks_aggregated.csv')
    df = pd.read_csv(ks_agg_path)
    df['datetime_x'] = df['datetime_x'].astype('datetime64')
    df['datetime_y'] = df['datetime_y'].astype('datetime64')

    weather_df = pd.read_csv(LINK,
                             compression='gzip', header=6, encoding='UTF-8',
                             error_bad_lines=False, delimiter=';', index_col=False)
    weather_df = weather_df.rename(columns={"Местное время в Москве (центр, Балчуг)": "datetime_w", "E'": "E_"})
    weather_df['datetime_w'] = weather_df['datetime_w'].astype('datetime64')
    # Корректировка данных о погоде
    weather_df['rain'] = weather_df.apply(WW_fixer, axis=1)
    weather_df['RRR'] = weather_df.apply(RRR_fixer, axis=1)
    weather_df['RRR'] = weather_df['RRR'].astype('float64')
    weather_df['Nh'] = weather_df.apply(Nh_fixer, axis=1)
    weather_df['Nh'] = weather_df['Nh'].astype('float64')
    weather_df['N'] = weather_df.apply(N_fixer, axis=1)
    weather_df['N'] = weather_df['N'].astype('float64')
    weather_df['H'] = weather_df.apply(H_fixer, axis=1)
    weather_df['H'] = weather_df['H'].astype('float64')
    weather_df_final = weather_df.drop(
        ['Pa', 'DD', 'ff10', 'ff3', 'Tn', 'Tx', 'Cl', 'Cm', 'Ch', 'VV', 'Td', 'E', 'Tg', 'E_', 'sss', 'WW', 'W1', 'W2',
         'tR'], axis=1)
    weather_df_final['datetime_w'] = weather_df_final['datetime_w'].astype('datetime64')

    '''Соединение обработанной погоды и агрегированных данных кикшеринга'''
    df['date_day'] = df['datetime_x'].dt.date
    weather_df_final['date_day'] = weather_df_final['datetime_w'].dt.date
    weather_df_final['start_hour'] = weather_df_final['datetime_w'].dt.hour
    df = df.merge(weather_df_final, how='left', on=['date_day', 'start_hour'])
    missing = ['T', 'Po', 'P', 'U', 'Ff', 'N', 'Nh', 'H', 'RRR', 'rain']
    df.sort_values(by='datetime_x', ascending=True)
    for i in missing:
        df[i] = df[i].fillna(method='ffill')
    df.sort_values(by='datetime_x', ascending=True).head(50)

    '''Пространственный кластеринг парковок на основе точек стартов самокатов - 
    получаемый новый датасет вместо парковок с data.mos.ru'''
    df = df.query('lat_x > @lat_min and lat_x < @lat_max and lon_x < @lon_max and lon_x > @lon_min')
    # делаем геодатафрейм из датафрейма стартов самокатов
    starts_gdf_ll = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon_x, df.lat_x)).set_crs('epsg:4326')
    # перепроецируем в более подходящую для карт и измерения расстояний проекцию https://epsg.io/20007
    starts_gdf = starts_gdf_ll.to_crs(epsg=20007)
    # записываем чтобы не потерять спроецированные метровые координаты
    starts_gdf["X"] = starts_gdf.geometry.x
    starts_gdf["Y"] = starts_gdf.geometry.y

    # создаем кластеры с помощью алгоритма ADBSCAN
    # основные параметры - 100м радиус поиска
    # 3 самоката - это уже будет кластер
    # 0.6 датасета используется при каждом перерасчете
    # 200 перерасчетов выполняется
    adbs = ADBSCAN(eps=100,
                   min_samples=3,
                   pct_exact=0.6,
                   algorithm='kd_tree',
                   reps=200,
                   keep_solus=True,
                   n_jobs=-1)
    np.random.seed(42)
    adbs.fit(starts_gdf)
    # переносим в геодатафрейм получившиеся классы кластеров
    starts_gdf = starts_gdf.merge(adbs.votes, left_index=True, right_index=True)
    parking = get_cluster_boundary(adbs.votes["lbls"], starts_gdf, crs=starts_gdf.crs)
    # сделаем геодатафрейм
    parking = gpd.GeoDataFrame(parking, geometry=parking, crs=parking.crs.to_string())
    # переводим в датафрейм с географическими координатами в виде градусов для маршрутизации
    parking_gdf = gpd.GeoDataFrame(parking, geometry=parking['geometry'].centroid.to_crs(epsg=4326), crs=4326).drop(0,
                                                                                                                    axis=1).reset_index(
        drop=True)
    # защита от "пропущенных" геометрий, которые могли возникнуть при поиске центроида
    parking_gdf['missing_geo'] = parking_gdf['geometry'].is_empty
    parking_gdf = parking_gdf.query('missing_geo == False').reset_index(drop=True)
    return G, gdf_nodes, parking_gdf
    # !!! далее именно этот датафрейм мы подставляем в качестве парковок в наш код, который рассчитывает расстояние от
    # заданной точки до парковок!!!


def WW_fixer(row):
    """функция, превращающая текст в числовые параметры"""
# используем две колонки - текущая погода и погода за прошлые три часа. 1 = дождь, 2 = ливень, 3 = гроза, 0 = дождя нет
    if row['WW'] == 'Гроза слабая или умеренная без града, но с дождем и/или снегом в срок наблюдения.' or row['WW'] == 'Гроза (с осадками или без них).':
        return 3
    if row['WW'] == 'Дождь незамерзающий непрерывный слабый в срок наблюдения.' or row['WW'] == 'Дождь незамерзающий непрерывный умеренный в срок наблюдения.' or ['WW'] == 'Дождь (незамерзающий) неливневый. ':
        return 1
    if row['WW'] == 'Ливневый(ые) дождь(и) слабый(ые) в срок наблюдения или за последний час. ' or row['WW'] == 'Ливневый(ые) дождь(и) очень сильный(ые) в срок наблюдения или за последний час. ':
        return 2
    if row['WW'] == 'Состояние неба в общем не изменилось. ' and row['W1'] == 'Ливень (ливни).':
        return 2
    if row['WW'] == 'Состояние неба в общем не изменилось. ' and row['W1'] == 'Дождь.':
        return 1
    if row['WW'] == 'Состояние неба в общем не изменилось. ' and row['W1'] == 'Гроза (грозы) с осадками или без них.':
        return 3
    else:
        return 0


def RRR_fixer(row):
    """Корректировка данных 'Кол-во осадков' """
    if row['RRR'] == 'Осадков нет':
        return 0.0
    return row['RRR']


def Nh_fixer(row):
    """Корректировка данных '% наблюдаемых кучевых или слоистых облаков' """
    if row['Nh'] == 'Облаков нет.':
        return 0.0
    if row['Nh'] == '20–30%.':
        return 0.3
    if row['Nh'] == '40%.':
        return 0.4
    if row['Nh'] == '50%.':
        return 0.5
    if row['Nh'] == '60%.':
        return 0.6
    if row['Nh'] == '70 – 80%.':
        return 0.8
    if row['Nh'] == '90  или более, но не 100%':
        return 0.9
    if row['Nh'] == '100%.':
        return 1
    return row['Nh']


def N_fixer(row):
    """Корректировка данных 'общая облачность в %' """
    if row['N'] == 'Облаков нет.':
        return 0.0
    if row['N'] == '10%  или менее, но не 0':
        return 0.1
    if row['N'] == '20–30%.':
        return 0.3
    if row['N'] == '40%.':
        return 0.4
    if row['N'] == '50%.':
        return 0.5
    if row['N'] == '60%.':
        return 0.6
    if row['N'] == '70 – 80%.':
        return 0.8
    if row['N'] == '90  или более, но не 100%':
        return 0.9
    if row['N'] == '100%.':
        return 1
    return row['N']


def H_fixer(row):
    """Корректировка данных 'высота основания самых низких облаков' """
    if row['H'] == '2500 или более, или облаков нет.':
        return 2500
    if row['H'] == '1000-1500':
        return 1000
    if row['H'] == '600-1000':
        return 600
    if row['H'] == '300-600':
        return 300
    return row['H']


def find_nearest_node(tree, gdf, point):
    """Поиск ближайших узлов графа к какой-то выданной нам точке и просто получаем их индекс"""
    # индексы используются в расчете маршрутов и т.д.
    closest_idx = tree.query([(point.y, point.x)], k=1, return_distance=False)[0]
    nearest_node = gdf.iloc[closest_idx].index.values[0]
    return nearest_node

'''
def nx_distance_actual(row, tree, gdf_nodes, a_location_gdf, G):
    node1 = find_nearest_node(tree, gdf_nodes, row['geometry'])  # здесь например датасет всех парковок
    node2 = find_nearest_node(tree, gdf_nodes, a_location_gdf['geometry'][0])  # здесь наш actual location
    try:
        row['distance_to_location'] = nx.shortest_path_length(G, node1, node2, weight='length')
    except nx.NetworkXNoPath:
        row['distance_to_location'] = None
    return row['distance_to_location']
    '''
'''---------------------------------------'''


def best_parking(data, parking_gdf):
    """
    data: It is dict with latitude and longitude
    return: List of 5 the best and nearest parks
    """

    '''Расчет расстояния от выданной точки до велопарковок созданных на основе кластеризации'''
    global G
    global gdf_nodes
    global tree
    # подаем на вход актуальные координаты и превращаем их в геодатафрейм для использования в поиске узлов графа
    if parking_gdf is None:
        G, gdf_nodes, parking_gdf = create_parking_dtf()
    a_location_lon = data['lon']
    a_location_lat = data['lat']
    a_location_df = pd.DataFrame(
        {'act_lat': a_location_lat,
         'act_lon': a_location_lon}, index=[0])
    a_location_gdf = gpd.GeoDataFrame(
        a_location_df, geometry=gpd.points_from_xy(a_location_df.act_lon, a_location_df.act_lat)).set_crs('epsg:4326')

    if tree is None:
    # здесь мы ищем ближайшие узлы графа к какой-то выданной нам точке и просто получаем их индекс. индексы используются в расчете маршрутов и т.д.
        tree = KDTree(gdf_nodes[['y', 'x']], metric='euclidean')

    # это аналитический стиль программирования не бейте
    def nx_distance_actual(row):
        node1 = find_nearest_node(tree, gdf_nodes, row['geometry'])  # здесь например датасет всех парковок
        node2 = find_nearest_node(tree, gdf_nodes, a_location_gdf['geometry'][0])  # здесь наш actual location
        try:
            row['distance_to_location'] = nx.shortest_path_length(G, node1, node2, weight='length')
        except nx.NetworkXNoPath:
            row['distance_to_location'] = None
        return row['distance_to_location']

    parking_gdf['distance_to_location'] = parking_gdf.apply(nx_distance_actual, axis=1)
    top_stations = parking_gdf.sort_values(by='distance_to_location', ascending=True)[['geometry', 'distance_to_location']].head(3)
    return parking_gdf, top_stations
