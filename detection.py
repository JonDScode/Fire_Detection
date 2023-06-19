
import tensorflow as tf
from PIL import Image

# Ruta al archivo H5 que contiene el modelo
ruta_modelo_h5 = '/content/drive/MyDrive/InceptionResNetV2-forest fire-100.0 /InceptionResNetV2-forest fire-100.0.h5'

# Cargar el modelo desde el archivo H5
model = tf.keras.models.load_model(ruta_modelo_h5)
labels = ["smoke", "fire", "nofire"]

def fire_detection(image_path):
    image = Image.open(image_path)
    image = image.resize((400, 200))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.inception_resnet_v2.preprocess_input(image)

    image = tf.expand_dims(image, axis=0)
    predictions = model.predict(image)[0]
    percentages = tf.nn.softmax(predictions)

    result = {label: percentage.numpy() for label, percentage in zip(labels, percentages)}
    return result

image_path = "/content/WEB10471.jpg"  # Reemplaza con la ruta de tu imagen
prediction = fire_detection(image_path)
print(prediction)