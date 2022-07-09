import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

SEED = 7

data = pd.read_csv('data/busca2.csv')
data = pd.get_dummies(data, columns=['busca'])

X = data.drop(columns=['comprou'])
Y = data['comprou']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, random_state=SEED, test_size=0.33, stratify=Y,
)

baseline_clf = DummyClassifier(strategy='most_frequent', random_state=SEED)
baseline_clf.fit(X_train, Y_train)

print('Precisão Dummy Most Frequent: %.2f%%' % (baseline_clf.score(X_test, Y_test) * 100))


clf = MultinomialNB()
clf.fit(X_train, Y_train)

print('Precisão MultinomialNB: %.2f%%' % (clf.score(X_test, Y_test) * 100))

clf = AdaBoostClassifier(random_state=SEED)
clf.fit(X_train, Y_train)

print('Precisão AdaBoost: %.2f%%' % (clf.score(X_test, Y_test) * 100))
