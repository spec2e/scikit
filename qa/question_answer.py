import json
import ast

all_q_a = []

with open("qa_baby.json") as f:

    for line in f:
        json_line = ast.literal_eval(line)
        json_data = json.dumps(json_line)
        json_data = json.loads(json_data)
        all_q_a.append(json_data)




