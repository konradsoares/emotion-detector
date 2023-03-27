# emotion-detector
Personal project classifying emotions in real-time using computer vision techniques and a CNN model

The process involves capturing a video stream from a camera, detecting faces using the Haar Cascade Classifier, and then extracting the face region. The extracted face region is then passed through a pre-trained CNN model, which has been trained to classify emotions from facial expressions. The CNN model makes a prediction on the extracted face, and the predicted emotion label is displayed in real-time on the video stream through a python flask webpage.


To understand better, a CNN is a type of neural network that is commonly used for image classification tasks. The CNN model used here consists of multiple layers, including convolutional layers, pooling layers, and fully connected layers. The convolutional layers learn filters that can detect specific features in an image, such as edges, textures, and patterns. The pooling layers down sample the feature maps to reduce the spatial dimensions, and the fully connected layers perform the final classification based on the extracted features.

Overall, this project demonstrates the powerful capabilities of machine learning and computer vision for real-time emotion recognition.

For the implementation, we are going to use a public dataset called fer-2023 and it has 48*48 gray-scale pictures of faces with their emotion labels. The dataset contains the values of pixels that we need to process in upcoming steps.

https://drive.google.com/drive/folders/1hF-GG3XmOJfVi410wwHtBCBqwkEqCTNS?usp=sharing

In model.py we have all the steps bellow.

Dataset Loading

Data preprocessing

Shaping data

Training Data Generator

Model Designing

Compiling Model

Model Training

Saving the Model

If you want to train this model and understand each step/funtion on this module, you must download the dataset to the same folder where is model.py and run it.
This will generate model_arch.json and model_weights.h5 which are already in the project data folder.

app_flask.py will load the model architecture and weights, load the face cascade classifier, read a frame from the camera, detect faces in the frame then use flask to create our web application that serves as the interface for the project. 


