
from pathlib import Path
import logging, logging.handlers, sys

def init_log():
	log_root = logging.getLogger(__name__)
	log_root.setLevel(logging.DEBUG)

	# Path('./logs').mkdir(exist_ok=True)
	
	handlers = [
		logging.StreamHandler(sys.stdout),
		# logging.handlers.RotatingFileHandler('logs/watchdog.log', maxBytes=1024*1024, backupCount=7),
	]

	handlers[0].setLevel(logging.DEBUG)
	# handlers[1].setLevel(logging.INFO)

	formatter = logging.Formatter('{asctime} | {name} {levelname} | {message}', style='{')

	for handler in handlers:
		handler.setFormatter(formatter)
		log_root.addHandler(handler)

init_log()

