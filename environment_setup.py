import os
from dotenv import load_dotenv
import yaml

load_dotenv()

with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Set environment variables
os.environ['DOMAIN'] = config['target_domain']
os.environ['USE_CASE'] = config['use_case']
os.environ['BASE_MODEL'] = config['base_model']
os.environ['API_AUTH_KEY'] = config['deployment']['api_auth_key']
