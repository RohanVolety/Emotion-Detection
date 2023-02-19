import numpy as np
import cv2
import av
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from keras import models 
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv



emotion_labels = ['Angry', 'Disgust', 'Fear','Happy', 'Neutral', 'Sad', 'Surprise']
classifier = load_model('Emotion_model.h5')
# load face
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        labels = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h),
                          color=(127, 0, 255), thickness=2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(img, label, label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img
        

class VideoProcessor(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        faces = face_classifier.detectMultiScale(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 3)

        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(img, format='bgr24')







def detect():
   
    st.title("Real Time Face Emotion Detection")
    choices = ["Home", "Face Detection", "Emotion Detection", "About"]
    option = st.sidebar.selectbox("Select Activity", choices)

    if option == "Home":
        Home = """<div style="background-color:#1771e6;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(Home, unsafe_allow_html=True)
        st.write("""
                 This Project has two applications.

                 1. Real time face detection.

                 2. Real time face emotion recognization.

                 """)
    elif option == "Emotion Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start detect your face emotion")
        webrtc_streamer(
            key="example", video_transformer_factory=VideoTransformer)
    
    elif option == "Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start detect your face")
        webrtc_streamer(
            key="work", video_transformer_factory=VideoProcessor)

    elif option == "About":
        st.subheader("About this app")
        About = """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(About, unsafe_allow_html=True)

        Developer_info = """
                             		<div style="background-color:#98AFC7;padding:10px">
                            		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(Developer_info, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    detect()


