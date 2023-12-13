
from pathlib import Path
from os import environ

DIR_SRC = Path(__file__).resolve().absolute().parents[1]
DIR_OUTPUTS = Path(environ.get('DIR_OUTPUTS', DIR_SRC / 'outputs'))
DIR_DATASETS = Path(environ.get('DIR_DATASETS', DIR_SRC / 'datasets'))
