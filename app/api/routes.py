from app.api import bp
from app.api.errors import bad_request
from flask import jsonify, request, url_for
import pickle
import numpy as np
from config import ROOT_DIR, parking_gdf, gdf_nodes, G, api_key
import os
import app.api.distance as ds
import requests
import ast


@bp.route('/check')
def check():
    return 'Works!'


@bp.route('/parking', methods=['GET', 'POST'])
def get_park():
    global parking_gdf
    global G
    global gdf_nodes
    # receive data from request
    data = request.get_json() or {}
    # checking data and converting to float
    if 'lat' in data:
        data['lat'] = float(data['lat'])
    else:
        return bad_request("No latitude in request")
    if 'lon' in data:
        data['lon'] = float(data['lon'])
    else:
        return bad_request("No longitude in request")

    parking_gdf, best_parks = ds.best_parking(data, parking_gdf)  # give lan and lot and receive best parkings

    
    # opening models
    dict_path = os.path.join(ROOT_DIR, 'other/dict')
    with open(dict_path, 'rb') as fp:
        itemlist = pickle.load(fp)
    checkMl_z = {}
    pickle_path = os.path.join(ROOT_DIR, 'other/models.pckl')
    file = open(pickle_path, 'rb')
    for i in range(362):
        checkMl_z[itemlist[i]] = pickle.load(file)
    file.close()

    # prediction
    top_parks_dist = best_parks.distance_to_location
    top_parks_dist = top_parks_dist.tolist()
    top_parks_dist = np.array(top_parks_dist)
    top_parks_point = best_parks.geometry
    top_parks_point = top_parks_point.tolist()
    top_parks_point = np.array(top_parks_point)
    top_parks_id = best_parks.index
    top_parks_id = top_parks_id.tolist()
    top_parks_id = np.array(top_parks_id)
    assign = np.empty(3)
    assign = top_parks_dist/1000 * checkMl_z.get(282199231).predict(np.reshape([19,2], (1, -1)))**2
 
    if 'flag' in data:
        if data["flag"] == True:
            body = {"coordinates":[[data["lat"],data["lon"]],[float(top_parks_point[assign.argmax()].y), float(top_parks_point[assign.argmax()].x)]]}

            headers = {
                'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
                'Authorization': str(api_key),
                'Content-Type': 'application/json; charset=utf-8'
            }
            full_responce = requests.post('https://api.openrouteservice.org/v2/directions/foot-walking', json=body, headers=headers)
            content_responce = full_responce.content
            dict_content_responce = content_responce.decode()
            route_responce = ast.literal_eval(dict_content_responce)
            return jsonify(route_responce)

    park_responce = {"ID": int(top_parks_id[assign.argmax()]),
                    "lat": top_parks_point[assign.argmax()].y,
                    "lon": top_parks_point[assign.argmax()].x}
    return jsonify(park_responce)


