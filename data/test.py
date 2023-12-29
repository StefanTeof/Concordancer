import json
import xml.etree.ElementTree as ET

tree = ET.parse('../oana-mk.ana15.02.xml')
root = tree.getroot()

namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}

with open('../annotations.json', 'r', encoding='utf-8') as json_file:
    annotations = json.load(json_file)

def test(): 
    print('test')
    for s_tag in root.findall('.//tei:s', namespace):
        # Process each <s> tag
        for child in s_tag:
            if(child.tag == '{http://www.tei-c.org/ns/1.0}w'):
                print("  ", child.tag, ":", child.get('lemma            '))
test()