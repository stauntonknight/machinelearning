import tensorflow as tf

class AccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if logs.get('accuracy') > 0.95:
            print('stopping now')
            self.model.stop_training = True
callbacks = AccuracyCallback()
data = tf.keras.datasets.fashion_mnist
(training_images, training_labels),(test_images, test_labels) = data.load_data()
# normalizing the images
training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs = 1, callbacks = [callbacks])
model.evaluate(test_images, test_labels)
classifications = model.predict(test_images)
print("here",classifications[0], test_labels[0])



model2 = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(10)
    ]
)
model2.compile(optimizer = 'sgd', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model2.fit(training_images, training_labels, callbacks = [callbacks], epochs = 50)
model2.evaluate(test_images, test_labels)

