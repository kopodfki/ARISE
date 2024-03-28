import pickle
import json

filepath = 'C:/ChatScene/retrieve/database_v1.pkl'

with open(filepath, "rb") as f:
    data = pickle.load(f)
    with open('C:/ChatScene/retrieve/database_v1.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
print('Unpickling done!')
