import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

# Ruta al archivo H5 que contiene el modelo
ruta_modelo_h5 = '/content/drive/MyDrive/InceptionResNetV2-forest fire-100.0 /InceptionResNetV2-forest fire-100.0.h5'

# Cargar el modelo desde el archivo H5
model = tf.keras.models.load_model(ruta_modelo_h5)
labels = ["smoke", "fire", "nofire"]

def fire_detection(image):
    image = image.resize((400, 200))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.inception_resnet_v2.preprocess_input(image)

    image = tf.expand_dims(image, axis=0)
    predictions = model.predict(image)[0]
    percentages = tf.nn.softmax(predictions)

    result = {label: percentage.numpy().tolist() for label, percentage in zip(labels, percentages)}
    return result

input_image = gr.inputs.Image(type="pil", label="Input Image")
output_label = gr.outputs.Label(label="Label")

title = "Fire Detection"
description = "Upload an image and get the label prediction with percentages."
examples = [["/content/PublicDataset00749.jpg"], ["/content/WEB10471.jpg"], ["/content/fire1.jpg"], ["/content/fire3.jpg"], ["/content/forest.jpg"]]

gr.Interface(fn=fire_detection, inputs=input_image, outputs=output_label, title=title, description=description, examples=examples).launch(debug=True)


