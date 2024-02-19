#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('testmodel.h5')

# Create a list of class labels (disease names) used during training
class_labels = ['Tomato_Bacterial_Spot', 'Tomato_Early_Blight', 'Tomato_Healthy', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_Spot', 'Tomato_Spider_mites', 'Tomato_Target_Spot', 'Tomato_mosaic_virus', 'Tomato_Yellow_Leaf_Curl_Virus']  # Corrected class name

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, change it if using an external camera

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Error capturing frame")
        break
    
    # Resize the frame to match the input size of your model
    img = cv2.resize(frame, (224, 224))
    
    # Preprocess the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize the image data
    
    # Make predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    
    # Get the disease name from the class_labels list
    disease_name = class_labels[predicted_class[0]]
    
    # Display the image and disease name
    cv2.imshow("Leaf Disease Detection", frame)
    print(f'Predicted Disease: {disease_name}')
    break
    
    # Press 'q' to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




