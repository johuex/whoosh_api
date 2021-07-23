from app.api import bp
from app.api.errors import bad_request
from flask import jsonify, request, url_for
import pickle
import numpy as np
from config import ROOT_DIR
import os
import app.api.serialize_sk as ssk
import app.api.distance as ds

parking_gdf = None

@bp.route('/check')
def check():
    return 'Works!'


@bp.route('/parking', methods=['POST'])
def get_park():
    global parking_gdf
    # receive data from request
    data = request.get_json() or {}
    # checking data and converting to float
    if 'lat' in data:
        data['lat'] = float(data['lat'])
    else:
        bad_request("No latitude in request")
    if 'lon' in data:
        data['lon'] = float(data['lon'])
    else:
        bad_request("No longitude in request")

    parking_gdf, best_parks = ds.best_parking(data, parking_gdf)  # give lan and lot and receive best parkings

    # opening models
    dict_path = os.path.join(ROOT_DIR, 'other\dict')
    with open(dict_path, 'rb') as fp:
        itemlist = pickle.load(fp)
    checkMl_z = {}
    pickle_path = os.path.join(ROOT_DIR, 'other\models.pckl')
    file = open(pickle_path, 'rb')
    for i in range(362):
        checkMl_z[itemlist[i]] = pickle.load(file)
    file.close()

    # prediction
    assign = np.sqrt(checkMl_z.get(282199231).predict(best_parks))
    '''
    где 282199231 - это id парковки из того файла
    0 - час который мы предсказываем
    4 - день который мы предсказываем (0 - понедельник и т.д.)
    '''
    response = ssk.encode(assign.nbiggest(5))

    return jsonify(response)


