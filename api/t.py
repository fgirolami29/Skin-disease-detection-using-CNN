import tensorflow as tf

# Percorso alla cartella del modello
model_path = "/Users/federicogirolami/Documents/GitHub/Skin-disease-detection-using-CNN/models/1"

# Carica il modello senza compilazione
model = tf.keras.models.load_model(model_path, compile=False)

# Ricompila il modello con una funzione di perdita compatibile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')  # Usa una loss adatta al tuo problema


# Salva di nuovo il modello per essere compatibile con la versione corrente di TensorFlow
model.save("new_model.h5")
