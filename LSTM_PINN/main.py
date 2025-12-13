from config import Config
from train import run_comparison

if __name__ == "__main__":
    # NOTE: You must have a config.py file with the necessary Config class
    app_config = Config() 
    run_comparison(app_config)
