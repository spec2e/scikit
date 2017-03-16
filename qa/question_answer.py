import json
import ast
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing import sequence
from time import time
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics



all_q_a = []
X = []
y = []

with open("qa_baby.json") as f:

    for line in f:
        json_line = ast.literal_eval(line)
        json_data = json.dumps(json_line)
        json_data = json.loads(json_data)
        all_q_a.append(json_data)


print("number of question/answers %i" % len(all_q_a))

for qa in all_q_a:
    X.append(qa["question"].lower())
    y.append(qa["answer"].lower())


print("count in X %i" % len(X))
print("count in Y %i" % len(y))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

print("count x_train %i" % len(x_train))
print("count y_train %i" % len(y_train))
print("count x_test %i" % len(x_test))
print("count y_test %i" % len(y_test))
print("count x_val %i" % len(x_val))
print("count y_val %i" % len(y_val))


vectorizer = TfidfVectorizer(min_df=1)
X_vec = vectorizer.fit_transform(X)
print(len(X_vec.indices))


print(x_test[0])
question = vectorizer.transform([x_test[0]])
print(question.indices)


def benchmark(clf, x_set, y_set, x_test_set, y_test_set):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(x_set, y_set)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(x_test_set)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test_set, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test_set, pred,
                                            target_names=target_names))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test_set, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


x_q_vectors = []
for question in x_train:
    indices = vectorizer.transform([question]).indices
    x_q_vectors.append(indices)

x_q_vectors = sequence.pad_sequences(x_q_vectors, maxlen=500)
print(len(x_q_vectors))

x_test_q_vectors = []
for question in x_test:
    indices = vectorizer.transform([question]).indices
    x_test_q_vectors.append(indices)

x_test_q_vectors = sequence.pad_sequences(x_test_q_vectors, maxlen=500)

results = []
# for clf, name in (
# (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
# (Perceptron(n_iter=50), "Perceptron"),
# (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
# (KNeighborsClassifier(n_neighbors=10), "kNN"),
# (RandomForestClassifier(n_estimators=100), "Random forest")):
name = "Random forest"
clf = RandomForestClassifier(n_estimators=100)

print('=' * 80)
print(name)

clf.fit(x_q_vectors, y_train)