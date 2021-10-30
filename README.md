# deep-learning

## Logistic Regression

__W__: n * 1, n features

__X__: n * m, m examples, each example contains n features

__b__: 1 * 1

__Z__: 1 * m, m examples. Z = dot(W.T, X) + b

__A__: 1 * m, m examples. A = sigmoid(Z)

__dZ__: 1 * m, m examples. dZ = A - Y

__dW__: n * 1, n features. dW = dot(X, dZ.T) / m

## Neural Network with one Hidden Layer

__W[i]__: L(i) * L(i-1), L(i) indicates the number of features in i-th layer

__A[i]__: L(i) * m, m example, L(i) number of features in i-th layer

- __A[0]__ = __X__: n * m, m example, each example contains n initial features

- __A[i]__ = g(Z[i]), if i > 0

__B[i]__: L(i) * 1

__Z[i]__: L(i) * m, m example. Z[i] = dot(W[i], A[i-1]) + B[i]

__dZ[i]__: L(i) * m

- __dZ[2]__ = A[2] - Y

- __dZ[1]__ = dot(W[2].T, dZ[2]) * g[1]'(Z[1])

__dW[i]__: L(i) * L(i-1). dW[i] = dot(dZ[i], A[i-1].T) / m

__dB[i]__: L(i) * 1. dB[i] = sum(dZ[i]) / m
