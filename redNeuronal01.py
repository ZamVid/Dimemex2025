import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('validation.csv')

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['etiqueta'])
val_labels = label_encoder.transform(val_df['etiqueta'])

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['texto'])

train_sequences = tokenizer.texts_to_sequences(train_df['texto'])
val_sequences = tokenizer.texts_to_sequences(val_df['texto'])

max_length = 100  
train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
val_padded = tf.keras.preprocessing.sequence.pad_sequences(val_sequences, maxlen=max_length, padding='post', truncating='post')

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(24, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(
    train_padded, train_labels,
    epochs=10,
    validation_data=(val_padded, val_labels),
    batch_size=32
)

loss, accuracy = model.evaluate(val_padded, val_labels)
print(f'Precisión en validación: {accuracy*100:.2f}%')

def predict_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded)
    predicted_label = label_encoder.inverse_transform([prediction.argmax()])[0]
    return predicted_label, prediction

texto_ejemplo = "te odio"
etiqueta, probabilidades = predict_text(texto_ejemplo)
print(f"Texto: {texto_ejemplo}")
print(f"Etiqueta predicha: {etiqueta}")
print(f"Probabilidades: {probabilidades}")
