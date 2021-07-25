import os, sys

# инпуты чтобы составить ссылку по которой мы будем качать архив:
WEB = 'http://37.9.3.253/download/files.synop/'
REGION = '27/'
STATION = '27605.'
START = '17.05.2021.'
FINISH = '31.05.2021.'
PIECE = '1.0.0.ru.'
CODE = 'utf8.'
FORMAT = '00000000.csv.gz'

ROOT_DIR = os.path.dirname(os.path.abspath('config.py'))
LINK = WEB + REGION + STATION + START + FINISH + PIECE + CODE + FORMAT

parking_gdf = None
G = None
gdf_nodes = None
tree = None
