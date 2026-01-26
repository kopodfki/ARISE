import json
import pickle

with open('C:/git/ScenicToOSC/ChatScene/db/database_v1_scenic3.json', 'r') as json_file:
    data = json.load(json_file)
    with open('C:/git/ScenicToOSC/ChatScene/db/database_v1_scenic3.pkl', 'wb') as f:
        pickle.dump(data, f)
print('Pickling done!')
