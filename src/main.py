from data.processor import build_dataset

X, y = build_dataset()

print("Shape X:", X.shape)
print("Shape y:", y.shape)
print("Exemplo X:", X[0])
print("Exemplo y:", y[0])
