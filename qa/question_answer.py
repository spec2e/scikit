import json
import ast
from sklearn.model_selection import train_test_split

all_q_a = []
X = []
Y = []

with open("qa_baby.json") as f:

    for line in f:
        json_line = ast.literal_eval(line)
        json_data = json.dumps(json_line)
        json_data = json.loads(json_data)
        all_q_a.append(json_data)


print("number of question/answers %i" % len(all_q_a))

for qa in all_q_a:
    X.append(qa["question"])
    Y.append(qa["answer"])


print("count in X %i" % len(X))
print("count in Y %i" % len(Y))


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

print("count x_train %i" % len(x_train))
print("count y_train %i" % len(y_train))
print("count x_test %i" % len(x_test))
print("count y_test %i" % len(y_test))
print("count x_val %i" % len(x_val))
print("count y_val %i" % len(y_val))


