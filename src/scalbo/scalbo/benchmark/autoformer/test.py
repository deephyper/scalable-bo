import yaml
from yaml.loader import SafeLoader

with open('configs.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)
    print(data['data']['batch_size'])