import json
import numpy as np

with open("img/json_out.json", 'r') as load_f:
    data = json.load(load_f)

rects = data['res']
rows = 26
cols = 9
rects = np.array(rects).reshape(rows, cols)
np.savetxt("img/result.csv", rects, fmt="%s", delimiter=',')
