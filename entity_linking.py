"""Example of Python client calling Knowledge Graph Search API."""
import json
import urllib.parse
import time
import urllib.request
#import urllib.parse.urlencode
from fuzzywuzzy import fuzz
import argparse
# api_key = open('.AIzaSyCicUQZLC2qCiQ7WpoG2nLmkz2qTcqOgcE').read()

import json
def load_json(fname):
    with open(fname) as f:
        return json.load(f)

def dump_json(obj, fname, indent=None):
    with open(fname, 'w') as f:
        return json.dump(obj, f, indent=indent)

service_url = 'https://kgsearch.googleapis.com/v1/entities:search'

entities = []

file = open("./wq_detection_result.txt", 'r', encoding='utf-8').readlines()

for line in file:
    entities.append(line.strip())

def entity_linking(entities):
    count = 0
    entity2id = {}
    for entity in entities:
        params = {
            'query': entity,
            'key': 'AIzaSyCicUQZLC2qCiQ7WpoG2nLmkz2qTcqOgcE',
        }

        url = service_url + '?' + urllib.parse.urlencode(params)

        try:
            response = json.loads(urllib.request.urlopen(url).read())
        except:
            print("lll")
            # output_file.write(entity +"|"+ "NoneLinkedResult" +"\n")
        str_result = []
        items = []
        for element in response['itemListElement']:
            if "g/" in element['result']['@id']:
                continue
            score = fuzz.ratio(entity, element['result']['name']) / 100.0
            items.append((element['result']['name'], element['result']['@id'], element['resultScore'], score))

        # items = sorted(items, key = lambda x: (x[2], x[3]), reverse = True)
        # print(items)
        if not items:
            continue
        iii = items[0]
        ids = iii[1]
        ids = ids.strip().replace("kg:/",'')
        ids = ids.replace("/", '.')
        if entity not in entity2id.keys():
            entity2id[entity]=ids
        count += 1
        print("process lines...:", count)
    dump_json(entity2id, "./results/webqsp_entity2label.json")

if __name__ == "__main__":
    # entities = ['wnyc-fm']
    entity_linking(entities)
