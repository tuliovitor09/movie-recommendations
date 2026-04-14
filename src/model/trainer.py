import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


def train_model(X, y):

    # separar treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Criar modelo
    model = tf.keras.Sequential(
        [
            layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    # 5. Avaliar modelo
    loss, acc = model.evaluate(X_test, y_test)

    # print(f"Loss: {loss:.4f}")
    # print(f"Accuracy: {acc:.4f}")

    return model
