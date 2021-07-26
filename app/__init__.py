from flask import Flask
from flask_cors import CORS, cross_origin


def create_app():
    app = Flask(__name__)
    cors = CORS(app)
    app.config['CORS_HEADERS'] = 'Content-Type'
    from app.api import bp as api_bp
    app.register_blueprint(api_bp)

    return app