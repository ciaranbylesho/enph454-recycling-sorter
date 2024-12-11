import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
import numpy as np
import shutil
import random

# Function to load and process images
def load_image(image_path):
    # Load the image using tf.io.read_file and decode it into a tensor
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)  # Decode to RGB (3 channels)
    img = tf.image.resize(img, (128, 128))  # Resize to 224x224
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

# Function to extract label from directory structure
def extract_label_from_path(image_path):
    # Extract label (class) from the image path
    label = tf.strings.split(image_path, os.path.sep)[-2]  # Get the parent directory as label
    return int(label)

def load_image_ds(image_folder):
    # Get image paths from all subdirectories
    image_paths = [os.path.join(root, fname)
                   for root, dirs, files in os.walk(image_folder)
                   for fname in files
                   if fname.endswith('.jpg') or fname.endswith('.png')]
    # Create a TensorFlow dataset from image file paths
    image_paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    # Map the load_image function to each image path
    image_ds = image_paths_ds.map(lambda x: load_image(x))
    # Get list of labels (for classification) using the above function
    labels_ds = image_paths_ds.map(lambda x: extract_label_from_path(x))
    print(labels_ds)
    return tf.data.Dataset.zip((image_ds, labels_ds))

def extract_label_from_path_multiclass(image_path, classes_n):
    # Extract label (class) from the image path
    label = int(tf.strings.split(image_path, os.path.sep)[-2])  # Get the parent directory as label
    # Create a tensor of zeros with the shape of the number of classes
    output = tf.one_hot(label, depth=classes_n, on_value=1.0, off_value=0.0)
    return output
    
def augment_image(image):
    # Random Horizontal Flip
    image = tf.image.random_flip_left_right(image)

    # Random Vertical Flip
    image = tf.image.random_flip_up_down(image)

    # Random Rotation
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    # Random Zoom (Resize and Crop)
    scale_factor = tf.random.uniform(shape=[], minval=0.8, maxval=1.2)
    new_size = tf.cast(tf.shape(image)[0:2], tf.float32) * scale_factor
    new_size = tf.cast(new_size, tf.int32)
    image = tf.image.resize(image, new_size)
    image = tf.image.resize_with_crop_or_pad(image, target_height=128, target_width=128)

    # Random Brightness
    image = tf.image.random_brightness(image, max_delta=0.1)

    # Random Contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    return image

def load_image_ds_multiclass(image_folder):
    # Get image paths from all subdirectories
    image_paths = [os.path.join(root, fname)
                   for root, dirs, files in os.walk(image_folder)
                   for fname in files
                   if fname.endswith('.jpg') or fname.endswith('.png')]
    classes_n = len(os.listdir(image_folder))
    # Create a TensorFlow dataset from image file paths
    image_paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    # Map the load_image function to each image path
    # image_ds = image_paths_ds.map(lambda x: random_rotate_image(load_image(x)))
    image_ds = image_paths_ds.map(lambda x: augment_image(load_image(x)))
    # Get list of labels (for classification) using the above function
    labels_ds = image_paths_ds.map(lambda x: extract_label_from_path_multiclass(x, classes_n=classes_n))
    # print(labels_ds)
    return tf.data.Dataset.zip((image_ds, labels_ds))

def split_files_kfold(data_path, dest_path, k=5):
    dir_list = os.listdir(data_path) # is folder name only

    # make new folders
    os.mkdir(dest_path)
    for fold_dir in range(k):
        os.mkdir(f"{dest_path}/{fold_dir}fold")
        for dir in dir_list:
            os.mkdir(f"{dest_path}/{fold_dir}fold/{dir}")
       
    for dir in dir_list:
        photos_paths = os.listdir(f"{data_path}/{dir}/")        
        # initialize a numbers array
        numbers = list(range(len(photos_paths)))
        # Shuffle the numbers randomly
        random.shuffle(numbers)
        # Create empty groups
        groups = [[] for _ in range(k)]
        # Distribute numbers into groups
        for i, number in enumerate(numbers):
            groups[i % k].append(number)

        # copy files to other directory
        for i in range(k):
            for photoidx in tqdm(groups[i]):
                shutil.copy(f'{data_path}/{dir}/{photos_paths[photoidx]}', f'{dest_path}/{i}fold/{dir}/{photos_paths[photoidx]}')
                
def calculate_per_class_accuracy(model, dataset):
    """
    Calculates accuracy for each class for a given model and dataset.
    
    Args:
    - model: Trained TensorFlow/Keras model.
    - dataset: TensorFlow dataset containing images and corresponding labels.
    
    Returns:
    - A dictionary with class index as key and per-class accuracy as value.
    """
    # Initialize variables to keep track of correct predictions and total samples per class
    correct_predictions = np.zeros(model.output_shape[-1], dtype=np.int32)
    total_samples = np.zeros(model.output_shape[-1], dtype=np.int32)
    
    # Iterate through the dataset
    for images, labels in dataset:
        # Get model predictions (probabilities or logits)
        predictions = model(images, training=False)
        
        # Convert predictions to class labels (indices of maximum probability)
        predicted_labels = np.argmax(predictions, axis=-1)
        true_labels = np.argmax(labels, axis=-1)  # Convert one-hot to class indices
        
        # Count correct predictions per class
        for i in range(model.output_shape[-1]):  # Loop over the number of classes
            correct_predictions[i] += np.sum((predicted_labels == i) & (true_labels == i))
            total_samples[i] += np.sum(true_labels == i)
    
    # Calculate accuracy for each class
    per_class_accuracy = {}
    for i in range(model.output_shape[-1]):
        if total_samples[i] > 0:
            per_class_accuracy[i] = correct_predictions[i] / total_samples[i]
        else:
            per_class_accuracy[i] = 0.0  # Avoid division by zero for empty classes
    
    return per_class_accuracy

def move_files_with_string(source_dir, target_dir, search_string="copy"):
    """
    Moves all files containing a specific string (default is 'copy') in their filename
    from the source directory to the target directory.

    Parameters:
    - source_dir (str): The directory where the files are located.
    - target_dir (str): The directory to move the files to.
    - search_string (str): The string to search for in filenames (default is 'copy').
    """
    # Check if the source directory exists
    if not os.path.exists(source_dir):
        print(f"The source directory '{source_dir}' does not exist.")
        return

    # Check if the target directory exists, if not, create it
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Target directory '{target_dir}' created.")

    # List all files in the source directory
    for filename in os.listdir(source_dir):
        # Check if the search string is in the file name
        if search_string.lower() in filename.lower():  # case-insensitive search
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)
            
            try:
                # Move the file
                shutil.move(source_path, target_path)
                print(f"Moved: {filename}")
            except Exception as e:
                print(f"Error moving file {filename}: {e}")