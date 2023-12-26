import csv
import xml.etree.ElementTree as ET


tree = ET.parse('../oana-mk.ana15.02.xml')
root = tree.getroot()

print(f"Root Element: {root.tag}")

csv_file = open('../data.csv', 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)


csv_writer.writerow(['sentence_id', 'word', 'lemma', 'category', 'type', 'gender', 'number', 'case', 'definiteness', 'person', 'tense', 'aspect', 'negation', 'degree', 'formation', 'vform', 'form'])

def extract_features(annotation):
    features = {'category': 'Nan', 'type': 'Nan', 'gender': 'Nan', 'number': 'Nan', 'case': 'Nan', 'definiteness': 'Nan', 'person': 'Nan', 'tense': 'Nan', 'aspect': 'Nan', 'negation': 'Nan', 'degree': 'Nan', 'formation': 'Nan', 'vform': 'Nan', 'form': 'Nan'}
    if 'N' in annotation:
        features.update({'type': 'NaN', 'gender': 'NaN', 'number': 'NaN', 'case': 'NaN', 'definiteness': 'NaN'})
    elif 'V' in annotation:
        features.update({'person': 'NaN', 'tense': 'NaN', 'aspect': 'NaN', 'negation': 'NaN', 'degree': 'NaN', 'formation': 'NaN', 'vform': 'NaN', 'form': 'NaN'})
    elif 'A' in annotation:
        features.update({})
    else:
        print("Unknown feature type")
        
    return features

