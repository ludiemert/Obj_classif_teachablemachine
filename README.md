# Obj_classif_teachablemachine

## Introduction and Application of Teachable Machine

- Teachable Machine by Google is a tool designed to make machine learning accessible and easy to use. Its intuitive interface allows users—even those without programming or data science experience—to train models with their own data, such as images, sounds, or poses.
- The tool's goal is to democratize machine learning, enabling anyone to create innovative projects without needing advanced technical knowledge. Through a straightforward process of "teaching" patterns to the machine, users can develop applications for fields like education, healthcare, and entertainment.
- Teachable Machine also allows trained models to be exported for use in websites, apps, and devices, broadening opportunities for developers and creators. As both an educational resource and a platform for innovation, Teachable Machine invites anyone, regardless of technical skill, to engage with machine learning.

---

# Object Recognition with TensorFlow and OpenCV

 - This project uses **TensorFlow** and **OpenCV** to create a real-time object recognition system, capturing images through the webcam and identifying four distinct classes: **Calculator**, **Remote Control**, **Lamp**, and **Background**.
 - This project is perfect for learning about computer vision and the application of pre-trained AI models.

---

## Project Objective

- The goal is to demonstrate how AI models can be applied to real-time images, turning camera captures into an intelligent interface that recognizes objects.
- This example can be extended to other contexts, such as item classification in surveillance systems or automated assistants interacting with real-world objects.

---


## Project Img

<div align="center">
    <img src="img_Teachable Machine/0-array com obj.jpg" style="width: 45%; margin-right: 5%;" alt="0-array com obj">
    <img src="img_Teachable Machine/1_Teachable Machine.png" style="width: 45%; margin-right: 5%;" alt="1_Teachable Machine">
  <br/>
  <br/>
   <img src="img_Teachable Machine/01-detec obj .jpg" style="width: 45%; margin-right: 5%;" alt="01-detec obj">
   <img src="img_Teachable Machine/2_Teachable Machine.png" style="width: 45%; margin-right: 5%;" alt="2_Teachable Machine">
   <br/>
   <br/>
   <img src="img_Teachable Machine/02-detec ob calc.jpg" style="width: 45%; margin-right: 5%;" alt="02-detec ob calc">
   <img src="img_Teachable Machine/3_Teachable Machine.png" style="width: 45%; margin-right: 5%;" alt="3_Teachable Machine">
   <br/>
   <br/>
   <img src="img_Teachable Machine/03- detec obh.jpg" style="width: 45%; margin-right: 5%;" alt="03- detec obh">
   <img src="img_Teachable Machine/4_Teachable Machine.png" style="width: 45%; margin-right: 5%;" alt="4_Teachable Machine">
   <br/>
 <br/>
   <img src="img_Teachable Machine/5_Teachable Machine.png" style="width: 45%; margin-right: 5%;" alt="5_Teachable Machine">
   <img src="img_Teachable Machine/6_ Teachable Machines_rec_caneta.png" style="width: 45%; margin-right: 5%;" alt="6_ Teachable Machines_rec_caneta">
   <br/>
</div>

---

## Features

- Real-time video capture using the system's webcam
- Image classification into four categories: Calculator, Remote Control, Lamp, and Background
- Display of the identified object’s name overlaid on the captured image

## Technologies Used

- **TensorFlow**: To load and run the deep learning model.
- **OpenCV**: For image manipulation and real-time video capture.
- **NumPy**: For array manipulation and mathematical operations.
- **Keras**: For handling the trained model in `.h5` format.

## Prerequisites

To run the project, you will need:

- **Python 3.8**
- Libraries: TensorFlow, OpenCV, and NumPy. To install, run:

 ```bash
pip install keras==2.9.0
pip install tensorflow==2.9.1
pip install h5py==3.11.0
pip install numpy
pip install opencv-python

 ```
  
 ```bash
 pip install tensorflow opencv-python numpy
Code Structure
The main code is organized as follows:
python
Copiar código
# Import necessary libraries
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
```

## Display the TensorFlow version

```bash
print(tf.__version__)
```

## Load the saved deep learning model from the .h5 file

```bash
model = load_model('Model/Keras_model.h5')
```

## Create a NumPy array for the input image

```bash
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
```

## Initialize video capture

```bash
cap = cv2.VideoCapture(0)
```

## Recognizable classes by the model

```bash
classes = ['CALCULATOR', 'REMOTE', 'LAMP', 'BACKGROUND']
```

## Loop to capture and process the video

 ```bash
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (224, 224))  # Resize image
    image_array = np.asarray(imgS)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1  # Normalize pixels
    data[0] = normalized_image_array  # Set the input array
    prediction = model.predict(data)  # Make a prediction
    indexVal = np.argmax(prediction)  # Find the index of the class with the highest probability
 ```

 ## Display the class name on the video

 ```bash
    cv2.putText(img, str(classes[indexVal]), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    print(classes[indexVal])  # Print to console
```

```bash
    cv2.imshow('img', img)  # Show the video frame
    cv2.waitKey(1)  # Wait 1 millisecond
```
    
    
     
### Detailed Explanation

- Library Import: Imports the libraries required for array manipulation, video, and AI model loading.
- Model Loading: Loads the pre-trained model from an .h5 file.
- Video Capture: Starts video capture using the device’s camera.
- Pre-processing: Resizes and normalizes each video frame to be compatible with the model.
- Prediction: The model processes the image and predicts the class with the highest probability.
- Real-time Display: Overlays the recognized object’s name on the webcam image and displays it.


### How to Run the Project
1.	Clone this repository:
bash
Copiar código
```bash
git clone https://github.com/your_username/repository_name.git
```
3.	Access the project directory:
bash
Copiar código
cd repository_name
4.	Run the script:
bash
Copiar código
python your_script.py
When you run the script, a video window will display the webcam feed. Whenever an object is recognized, the class name will be overlaid on the image.

---

#### Example Result
When detecting a Calculator, the system will display “CALCULATOR” on the screen. Examples:
 - objectivec
 - Copiar código
 - CALCULATOR
 - REMOTE

 ---
 
#### Possible Expansions
This project can be expanded to support more classes and different types of objects. Other ideas include:
•	Adding new object classes in model training
•	Using an external camera to capture objects in larger environments
•	Implementing alerts or automated actions when a specific class is recognized
Contributing
---

### Contributions are welcome! If you have improvement ideas or want to fix an issue, please:
1.	Fork the project.
2.	Create a branch for your feature (git checkout -b my-feature).
3.	Commit your changes (git commit -m 'Add new feature').
4.	Push to the branch (git push origin my-feature).
5.	Open a pull request.
---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---


### 📦 Contribution

 - Feel free to contribute by submitting pull requests or reporting issues.

- #### My LinkedIn - [![Linkedin Badge](https://img.shields.io/badge/-LucianaDiemert-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/lucianadiemert/)](https://www.linkedin.com/in/lucianadiemert/)

#### Contact

<img align="left" src="https://www.github.com/ludiemert.png?size=150">

#### [**Luciana Diemert**](https://github.com/ludiemert)

🛠 Full-Stack Developer <br>
🖥️ Python Enthusiast | Computer Vision | AI Integrations <br>
📍 São Jose dos Campos – SP, Brazil

<a href="https://www.linkedin.com/in/lucianadiemert" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white" alt="LinkedIn Badge" height="25"></a>&nbsp;
<a href="mailto:lucianadiemert@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-D14836?style=flat&logo=gmail&logoColor=white" alt="Gmail Badge" height="25"></a>&nbsp;
<a href="#"><img src="https://img.shields.io/badge/Discord-%237289DA.svg?logo=discord&logoColor=white" title="LuDiem#0654" alt="Discord Badge" height="25"></a>&nbsp;
<a href="https://www.github.com/ludiemert" target="_blank"><img src="https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white" alt="GitHub Badge" height="25"></a>&nbsp;

<br clear="left"/>

