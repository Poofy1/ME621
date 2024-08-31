
import base64

# config.py
global_config = {
    'config': None,
    'headers': None
}

def initialize_global_config(config):
    global_config['config'] = config
    if config:
        key = f"{config['USERNAME']}:{config['API_KEY']}"
        key = base64.b64encode(key.encode()).decode()
        global_config['headers'] = {
            'User-Agent': "User Annotation (https://github.com/Poofy1/ME621)",
            'Authorization': f"Basic {key}"
        }
    else:
        global_config['headers'] = None