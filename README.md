# deep-learning

## Logistic Regression

__w__: n * 1, n features

__X__: n * m, m examples, each example contains n features

__b__: 1 * 1

__Z__: 1 * m, m examples. Z = dot(w.T, X) + b

__A__: 1 * m, m examples. A = sigmoid(Z)

__dZ__: 1 * m, m examples. dZ = A - Y

__dw__: n * 1, n features. dw = dot(X, dZ.T) / m
