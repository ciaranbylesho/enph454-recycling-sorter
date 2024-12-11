import cv2
import time
import pygame
from tqdm import tqdm
import setup_utils as su
import serial

### SETTINGS ###
file_start = 901
camera_num = 0
delay = 0.05
iters = 100
output = "data/training_data_6/3"
filename = "plastic_"
fixed_exposure = -7
port = 'COM7'
################

arduino = serial.Serial(port=port, baudrate=115200, timeout=.1)
# Boot up the arduino and run test runs to ensure it works
su.write('0', arduino=arduino)
Value = su.read(arduino=arduino)
while (Value != '0'):
    su.write('0', arduino=arduino)
    time.sleep(0.5)
    Value = su.read(arduino=arduino)
print("Arduino Boot Completed")
time.sleep(1)
su.StartMotor(arduino)
time.sleep(0.5)
print("Arduino Setup")

# Open the default camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_EXPOSURE, fixed_exposure)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
pygame.mixer.init()
pygame.mixer.music.load('Sound.wav')

for i in tqdm(range(iters)):
    # Delay
    pygame.mixer.music.play()
    time.sleep(delay)
    ret, frame = cam.read()
    # Display the captured frame and save it
    cv2.imshow('Camera', frame)
    cv2.imwrite(f"{output}/{filename}_{file_start}.jpg", frame)
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break
    file_start += 1
# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()
su.write("0", arduino=arduino)
arduino.close()