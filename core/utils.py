"""Module containing utility functions for plots and other auxiliary tasks

"""
import json


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)
