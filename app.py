import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image as pilimage

IMAGE_SIZE = (128, 128)
CLASS_NAMES = ["AI generated", "real"]

model = tf.keras.models.load_model("model_24-3-18.keras")

def evaluate(input_image):
    image = input_image.convert('RGB')
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image)
    image_array = image_array.flatten()/255.0
    image_array = image_array.reshape(-1,128,128,3)
    label_prediction_confidence = model.predict(image_array)
    label_prediction_class = np.argmax(label_prediction_confidence, axis = 1)[0]
    return "Class: " + CLASS_NAMES[label_prediction_class] + " with a confidence of: " + str(round(np.amax(label_prediction_confidence) * 100., 2))
    
demo = gr.Interface(
    fn=evaluate,
    inputs=gr.Image(type="pil"),
    outputs=["text"],
)

demo.launch()