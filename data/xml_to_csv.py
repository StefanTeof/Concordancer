import csv
import xml.etree.ElementTree as ET
import json


with open('../annotations.json', 'r', encoding='utf-8') as json_file:
    annotations = json.load(json_file)

tree = ET.parse('../oana-mk.ana15.02.xml')
root = tree.getroot()

namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}

csv_file = open('../data.csv', 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)


csv_writer.writerow(['sentence_id', 'word', 'lemma', 'category', 'type', 'gender', 'number', 'case', 'definite', 'person', 'tense', 'aspect', 'negation', 'degree', 'formation', 'vform', 'form'])


def extract_features(annotation):

    # print(annotations)

    features_tmp = {'category': 'Nan', 'type': 'Nan', 'gender': 'Nan', 'number': 'Nan', 'case': 'Nan', 'definite': 'Nan', 'person': 'Nan', 'tense': 'Nan', 'aspect': 'Nan', 'negation': 'Nan', 'degree': 'Nan', 'formation': 'Nan', 'vform': 'Nan', 'form': 'Nan'}

    category = annotation[0]
    if category in annotations:
        category_data = annotations[category]
        
        i = 0
        for feature in features_tmp.keys():
            if(feature in category_data.keys()):
                features_tmp[feature] = category_data[feature][annotation[i]]
                i += 1
                
    return features_tmp

sentence_id = 1

for s_tag in root.findall('.//tei:s', namespace):
    # print(s_tag)
    
    words = []
    lemmas = []
    anns = []

    for word in s_tag:
        if(word.tag == '{http://www.tei-c.org/ns/1.0}w'):
            print(word.tag, word.get('lemma'))
            words.append(word.text)
            lemmas.append(word.get('lemma'))
            anns.append(word.get('ana'))

    for word, lemma, annotation in zip(words, lemmas, anns):
        features = extract_features(annotation)
        csv_writer.writerow([sentence_id, word, lemma] + list(features.values()))

    sentence_id += 1

csv_file.close()