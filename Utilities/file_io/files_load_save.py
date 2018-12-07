import yaml

def load_yaml(file):
        if not isinstance(file, str): raise ValueError('Invalid input argument')
        with open(file, 'r') as f:
                loaded = yaml.load(f)
        return loaded
