import json

with open('../annotations.json', 'r', encoding='utf-8') as json_file:
    annotations = json.load(json_file)

def test():
    annotation = 'Ncnsny'
    features = {'category': 'Nan', 'type': 'Nan', 'gender': 'Nan', 'number': 'Nan', 'case': 'Nan', 'definite': 'Nan', 'person': 'Nan', 'tense': 'Nan', 'aspect': 'Nan', 'negation': 'Nan', 'degree': 'Nan', 'formation': 'Nan', 'vform': 'Nan', 'form': 'Nan'}


    category = annotation[0]
    if category in annotations:
        category_data = annotations[category]
        
        i = 0
        for feature in features.keys():
            if(feature in category_data.keys()):
                features[feature] = category_data[feature][annotation[i]]
                i += 1
                # print(feature)
                # print(category_data[feature][annotation[i]])
                # i += 1

    print(features)



test()