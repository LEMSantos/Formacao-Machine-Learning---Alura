from sklearn.naive_bayes import MultinomialNB


def get_class_names(predicted):
    return list(map(lambda x: 'porco' if x else 'cachorro', predicted))


# eh gordinho? tem perninha curta? faz auau?
data = [[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 1]]
target = [1, 1, 1, 0, 0, 0]
_inputs = [
    [1, 1, 1],
    [1, 0, 0],
    [0, 0, 1],
]
_input_class = [0, 1, 0]

clf = MultinomialNB()
clf.fit(data, target)

print('Precisão: %.2f%%' % (clf.score(_inputs, _input_class) * 100))
