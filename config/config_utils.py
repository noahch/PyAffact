import json
from dotmap import DotMap


def get_config(file_name):
    with open("config/" + file_name + ".json", "r") as f:
        my_dict = json.load(f)
    config = DotMap(my_dict)
    return config
