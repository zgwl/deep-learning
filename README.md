# deep-learning

## Logistic Regression

**w**: n * 1, n features
**X**: n * m, m examples, each example contains n features
**b**: 1 * 1
**Z**: 1 * m, m examples. Z = dot(w.T, X) + b
**A**: 1 * m, m examples. A = sigmoid(Z)
**dZ**: 1 * m, m examples. dZ = A - Y
**dw**: n * 1, n features. dw = dot(X, dZ.T) / m
