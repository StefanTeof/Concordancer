import csv
import xml.etree.ElementTree as ET
import json

with open('../annotations.json', 'r', encoding='utf-8') as json_file:
    annotations = json.load(json_file)

# print(annotations)

tree = ET.parse('../oana-mk.ana15.02.xml')
root = tree.getroot()

# print(f"Root Element: {root.tag}")

csv_file = open('../data.csv', 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)


csv_writer.writerow(['sentence_id', 'word', 'lemma', 'category', 'type', 'gender', 'number', 'case', 'definite', 'person', 'tense', 'aspect', 'negation', 'degree', 'formation', 'vform', 'form'])


def extract_features(annotation):

    features = {'category': 'Nan', 'type': 'Nan', 'gender': 'Nan', 'number': 'Nan', 'case': 'Nan', 'definite': 'Nan', 'person': 'Nan', 'tense': 'Nan', 'aspect': 'Nan', 'negation': 'Nan', 'degree': 'Nan', 'formation': 'Nan', 'vform': 'Nan', 'form': 'Nan'}

    category = annotation[0]
    if category in annotations:
        category_data = annotations[category]
        
        i = 0
        for feature in features.keys():
            if(feature in category_data.keys()):
                features[feature] = category_data[feature][annotation[i]]
                i += 1
                
    return features


print(extract_features('Ncnsny'))
