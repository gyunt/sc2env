import json
import os


def get_settings(game):
    data_filename = os.path.join(os.path.dirname(__file__), game, 'data.json')

    try:
        with open(data_filename) as f:
            settings = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError('Game not found: %s' % game)

    return settings
