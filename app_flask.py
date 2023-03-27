from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from keras.models import model_from_json

app = Flask(__name__)

# Load the model architecture and weights
model = model_from_json(open(".\data\model_arch.json", "r").read())
model.load_weights('.\data\model_weights.h5')

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('.\data\haarcascade_frontalface_default.xml')

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Define the emotions
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def gen_frames():
    while True:
        # Read a frame from the camera
        success, frame = video_capture.read()
        if not success:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Loop through each face and make a prediction
        for (x,y,w,h) in faces:
            face_img = gray[y:y+h, x:x+w]
            resized_img = cv2.resize(face_img, (48, 48))
            normalized_img = resized_img / 255.0
            reshaped_img = np.reshape(normalized_img, (1, 48, 48, 1))
            prediction = model.predict(reshaped_img)

            # Get the predicted emotion label
            label = EMOTIONS[np.argmax(prediction)]

            # Change the label if the prediction value is below a certain threshold
            if prediction[0][0] < 0.5:
                label = ''

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Convert the frame to JPEG format
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file from the request
    file = request.files['image']
    
    # Read the image file
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to 48x48 pixels
    resized_img = cv2.resize(gray, (48, 48))
    
    # Resize the image to match the input shape of the model
    input_img = cv2.resize(resized_img, (48, 48))
    
    # Reshape the image to match the input shape of the model
    input_img = np.reshape(input_img, (1, 48, 48, 1))
    
    # Normalize the input image
    input_img = input_img / 255.0
    
    # Make a prediction
    prediction = model.predict(input_img)
    
    # Get the predicted emotion label
    label = EMOTIONS[np.argmax(prediction)]
    
    # Draw the predicted emotion label on the image
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
   
    # Convert the image back to JPEG format
    _, jpeg = cv2.imencode('.jpg', img)
    return jpeg.tobytes()

@app.route('/')
def index():
    """Display the camera feed and face recognition results"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route for serving the video stream"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
