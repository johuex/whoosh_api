from app.api import bp
from app.api.errors import bad_request
from flask import jsonify, request, url_for


@bp.route('/parking/{lat}/{lon}', methods=['GET', 'POST'])
def get_park(lat, lon):
    lat = float(lat)
    lon = float(lon)
    response = {'': ''}
    return jsonify(response)


