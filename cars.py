#building a simple model using TensorFlow  that regenizes a car photo and tell if it is a car or not
import tensorflow as tf 

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
data = tf.keras.utils.image_dataset_from_directory(
    r'C:\Users\ahmid\Documents\data\car vs noncar',
    labels='inferred',
    label_mode='binary',
    image_size=(64, 64),
    batch_size=32
)
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2))) 
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(data, epochs=10, validation_data=(data), verbose=2)

test_loss, test_acc = model.evaluate(data, verbose=2)
print('\nTest accuracy:', test_acc)