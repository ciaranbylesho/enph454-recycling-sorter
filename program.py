import cv2
import tensorflow as tf
import numpy as np
import serial 
import time
import setup_utils as su

##### INPUTS #####
fixed_exposure = -8
port = 'COM7'
##### INPUTS #####

print(f"GPUs Connected: {len(tf.config.list_physical_devices('GPU'))}")

# Load your pre-trained TensorFlow model (replace with your model path)
model = tf.keras.models.load_model('models/model_ts6_4')

arduino = serial.Serial(port=port, baudrate=115200, timeout=.1)

# Set up the camera (default camera index 0)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE, fixed_exposure)    

# Boot up the arduino and run test runs to ensure it works
su.write('0', arduino=arduino)
Value = su.read(arduino=arduino)
while (Value != '0'):
    su.write('0', arduino=arduino)
    time.sleep(0.5)
    Value = su.read(arduino=arduino)
print("Arduino Boot Completed")

time.sleep(0.5)
su.StartMotor(arduino)
su.switchMiddle(arduino=arduino)
time.sleep(0.5)
print("Arduino Setup")


# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Program Commenced")
prev = 0
breakloop = False
# Loop to capture frames from the camera
while True:
    #check if there is anything in the way
    su.write("5", arduino=arduino)
    Value = su.read(arduino=arduino)
    # reminder, the switches send back signals 2, 3, 4
    while (Value != "5" and "5" != '0'):
        Value = su.read(arduino=arduino)
        for i in range(50):
            time.sleep(0.01)
            ret, frame = cap.read()
            cv2.imshow("Camera Feed", frame)
            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                breakloop = True
            if breakloop:
                break
    if breakloop:
        break
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    input_image = su.preprocess_image(frame)
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
    
    if predicted_class != prev:
        if predicted_class == 0:
            continue
        elif predicted_class == 1:
            su.switchRight(arduino=arduino)
        elif predicted_class == 2:
            su.switchLeft(arduino=arduino)
    prev = predicted_class
    time.sleep(1.5)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
su.write("0", arduino=arduino)
arduino.close()