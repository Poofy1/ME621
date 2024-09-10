
import base64, os, json
project_root = os.path.abspath(os.path.dirname(__file__))


# config.py
global_config = {
    'config': None,
    'headers': None,
}

CONFIG_FILE = f'{project_root}/config.json'


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            initialize_global_config(json.load(f))
            return True
    return False

def save_config(username, api_key, chat_id, bot_api):
    config = {
        "USERNAME": username,
        "E621_API": api_key,
        "TELEGRAM_GROUP_ID": chat_id,
        "TELEGRAM_API": bot_api
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Update global configuration
    initialize_global_config(config)

def initialize_global_config(config):
    global_config['config'] = config
    if config:
        key = f"{config['USERNAME']}:{config['E621_API']}"
        key = base64.b64encode(key.encode()).decode()
        global_config['headers'] = {
            'User-Agent': "User Annotation (https://github.com/Poofy1/ME621)",
            'Authorization': f"Basic {key}"
        }
    else:
        global_config['headers'] = None