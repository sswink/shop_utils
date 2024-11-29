import os
import json
import codecs
from configparser import ConfigParser

config: json = None



def load_config():
    global config

    config = json.load(codecs.open('../config.json', encoding='utf-8'))

def save_config(config_data):
    global config
    config = config_data
    file = codecs.open('../config.json', mode='w', encoding='utf-8')
    file.write(json.dumps(config, sort_keys=True, indent=4, separators=(',', ': ')))
    file.close()
    # for line in json.dumps(config, sort_keys=True, indent=4, separators=(',', ': ')).split("\n"):
    #     print(line)
