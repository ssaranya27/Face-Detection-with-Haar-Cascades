# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows  

### PROGRAM:
```
import cv2
import matplotlib.pyplot as plt
import numpy as np

w_glass = cv2.imread('withglass.jpg', cv2.IMREAD_GRAYSCALE)
wo_glass = cv2.imread('withoutglass.png', cv2.IMREAD_GRAYSCALE)
group = cv2.imread('poster.webp', cv2.IMREAD_GRAYSCALE)

w_glass1 = cv2.resize(w_glass, (1000, 1000))
wo_glass1 = cv2.resize(wo_glass, (1000, 1000)) 
group1 = cv2.resize(group, (1000, 1000))

plt.figure(figsize=(15,10))
plt.subplot(1,3,1);plt.imshow(w_glass1,cmap='gray');plt.title('With Glasses');plt.axis('off')
plt.subplot(1,3,2);plt.imshow(wo_glass1,cmap='gray');plt.title('Without Glasses');plt.axis('off')
plt.subplot(1,3,3);plt.imshow(group1,cmap='gray');plt.title('Group Image');plt.axis('off')
plt.show()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def detect_and_display(image):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 10)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

import cv2
from matplotlib import pyplot as plt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Cascade file not loaded properly!")
else:
    print("Cascade loaded successfully.")
w_glass1 = cv2.imread('withglass.jpg')  # <-- replace with your image filename

if w_glass1 is None:
    print("Error: Image not found. Check the filename or path.")
else:
    print("Image loaded successfully.")
def detect_and_display(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    
    return image
if w_glass1 is not None and not face_cascade.empty():
    result = detect_and_display(w_glass1)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

import cv2
from matplotlib import pyplot as plt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
if face_cascade.empty():
    print("Error: Face cascade not loaded properly!")
if eye_cascade.empty():
    print("Error: Eye cascade not loaded properly!")
# (Change the filenames as per your actual image files)
w_glass = cv2.imread('withglass.jpg')
wo_glass = cv2.imread('withoutglass.png')
group = cv2.imread('poster.webp')
def detect_eyes(image):
    face_img = image.copy()
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(face_img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    
    return face_img
if w_glass is not None:
    w_glass_result = detect_eyes(w_glass)
    plt.imshow(cv2.cvtColor(w_glass_result, cv2.COLOR_BGR2RGB))
    plt.title("With Glasses - Eye Detection")
    plt.axis("off")
    plt.show()

if wo_glass is not None:
    wo_glass_result = detect_eyes(wo_glass)
    plt.imshow(cv2.cvtColor(wo_glass_result, cv2.COLOR_BGR2RGB))
    plt.title("Without Glasses - Eye Detection")
    plt.axis("off")
    plt.show()

if group is not None:
    group_result = detect_eyes(group)
    plt.imshow(cv2.cvtColor(group_result, cv2.COLOR_BGR2RGB))
    plt.title("Group - Eye Detection")
    plt.axis("off")
    plt.show()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("on")
    plt.title("Video Face Detection")
    plt.show()
    break

cap.release()

```

### OUTPUT:

![alt text](image.png)

![alt text](image-1.png)

![alt text](image-2.png)

![alt text](image-3.png)

![alt text](image-4.png)

![alt text](image-5.png)

![alt text](image-6.png)

![alt text](image-7.png)

![alt text](image-8.png)

![alt text](image-9.png)

![alt text](image-10.png)

### RESULT:
    Thus, we executed Face Detection using Haar Cascades with OpenCV and Matplotlib
