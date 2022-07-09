import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

SEED = 42

data = pd.read_csv('data/acesso.csv')

X = data[[
    'acessou_home',
    'acessou_como_funciona',
    'acessou_contato',
]]

Y = data['comprou']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, random_state=SEED, test_size=0.25, stratify=Y,
)

clf = MultinomialNB()
clf.fit(X_train, Y_train)

print('Precisão: %.2f%%' % (clf.score(X_test, Y_test) * 100))
