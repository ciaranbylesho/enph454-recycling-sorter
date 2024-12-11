import cv2
import time
import numpy as np

# Function to process and classify the image
def preprocess_image(image):
    # Resize the image to the model's expected input size (e.g., 224x224)
    image_resized = cv2.resize(image, (128, 128))
    # Normalize the image (adjust this step as needed for your model)
    image_normalized = image_resized / 255.0
    # Expand dimensions to match the model's input shape
    image_input = np.expand_dims(image_normalized, axis=0)
    return image_input

def write_read(x, arduino): 
    val = bytes(x, 'utf-8')
    arduino.write(val) 
    time.sleep(0.5) 
    data = arduino.readline() 
    return str(data)[2]

def write(x, arduino):
    val = bytes(x, 'utf-8')
    arduino.write(val) 
    time.sleep(0.5) 

def read(arduino):
    data = arduino.readline() 
    return str(data)[2]

# 2 is elft
def switchLeft(arduino):
    write("2", arduino=arduino)
def switchRight(arduino):
    write("3", arduino=arduino)
def switchMiddle(arduino):
    write("4", arduino=arduino)
    
def StartMotor(arduino):
    write("1", arduino=arduino)
