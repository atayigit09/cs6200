import yaml

class ConfigLoader:
    @staticmethod
    def load(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def get_model_config(config, model_type):
        return config['models'][model_type]
