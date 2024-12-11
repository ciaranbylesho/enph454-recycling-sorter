import cv2
import tensorflow as tf
import numpy as np
import serial 
import time
import setup_utils as su

port = 'COM7'
fixed_exposure = -7

print(f"GPUs Connected: {len(tf.config.list_physical_devices('GPU'))}")

# Load your pre-trained TensorFlow model (replace with your model path)
model = tf.keras.models.load_model('models/model_ts6_5')

# Function to process and classify the image
def preprocess_image(image):
    # Resize the image to the model's expected input size (e.g., 224x224)
    image_resized = cv2.resize(image, (128, 128))
    # Normalize the image (adjust this step as needed for your model)
    image_normalized = image_resized / 255.0
    # Expand dimensions to match the model's input shape
    image_input = np.expand_dims(image_normalized, axis=0)
    return image_input

# arduino = serial.Serial(port=port, baudrate=115200, timeout=.1)
# # Boot up the arduino and run test runs to ensure it works
# su.write('0', arduino=arduino)
# Value = su.read(arduino=arduino)
# while (Value != '0'):
#     su.write('0', arduino=arduino)
#     time.sleep(0.5)
#     Value = su.read(arduino=arduino)
# su.StartMotor(arduino=arduino)
# su.switchMiddle(arduino=arduino)
# print("Arduino Boot Completed")

# Set up the camera (default camera index 0)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE, fixed_exposure)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Loop to capture frames from the camera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    input_image = preprocess_image(frame)
    # Use the model to predict the class of the frame
    predictions = model.predict(input_image)
    
    # Assuming the model output is a single class prediction (e.g., classification)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    # predicted_class = predictions[0]
    
    # Display the frame with the predicted class label
    label = f"Predicted Class: {predicted_class}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    label2 = f"{[x for x in np.round(predictions[0], 2)]}"
    cv2.putText(frame, label2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow("Camera Feed", frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
# su.write("0", arduino=arduino)
# arduino.close()