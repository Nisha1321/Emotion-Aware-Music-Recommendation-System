import cv2
import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from urllib.parse import quote
import time
import os

path = r'D:\emotion'
os.chdir(path)

# Load the CSV files containing song information for each mood category
happy_songs_spotify_df = pd.read_csv('happy.csv')
happy_songs_names_df = pd.read_csv('happy (2).csv')

sad_songs_spotify_df = pd.read_csv('sad.csv')
sad_songs_names_df = pd.read_csv('sad (2).csv')

angry_songs_spotify_df = pd.read_csv('calm.csv')
angry_songs_names_df = pd.read_csv('calm (2).csv')

romantic_songs_spotify_df = pd.read_csv('romantic.csv')
romantic_songs_names_df = pd.read_csv('romantic (2).csv')

fear_songs_spotify_df = pd.read_csv('fear.csv')
fear_songs_names_df = pd.read_csv('fear (2).csv')

# Load custom emotion detection model
model = load_model('model.h5')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    dominant_mood = None
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype("float") / 255.0
            roi_gray = img_to_array(roi_gray)
            roi_gray = np.expand_dims(roi_gray, axis=0)
            
            preds = model.predict(roi_gray)[0]
            emotion_probability = np.max(preds)
            dominant_emotion = emotion_labels[preds.argmax()]

            mood_mapping = {
                'angry': 'angry',
                'sad': 'sad',
                'neutral': 'neutral',
                'happy': 'happy',
                'surprise': 'surprised',
                'disgust': 'disgusted',
                'fear': 'fear'
            }

            dominant_mood = mood_mapping.get(dominant_emotion, 'unknown')

    return dominant_mood

def main():
    st.title("Emotion-Based Song Recommendation")
    # Open webcam
    cap = cv2.VideoCapture(0)
    capture_duration = 5
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        st.error("Cannot open webcam")
        return

    start_time = time.time()  # Track start time for capture duration
    dominant_mood = None
    detected_frame = None
    while True:
        # Read frame from webcam
        ret, frame = cap.read()

        # Check if frame is successfully read
        if not ret:
            print("Error: Cannot read frame")
            break

        # Perform emotion analysis on the frame
        dominant_mood = detect_emotion(frame)

        # If mood is detected, store the frame
        if dominant_mood != 'unknown':
            detected_frame = frame

        # Check for 'q' key press or capture duration reached
        if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time >= capture_duration:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    if dominant_mood:
        st.subheader(f"Detected Dominant Mood: {dominant_mood}")

        # Select appropriate mood dataframes based on the detected mood
        if dominant_mood == 'happy':
            mood_songs_spotify_df = happy_songs_spotify_df
            mood_songs_names_df = happy_songs_names_df
        elif dominant_mood == 'sad':
            mood_songs_spotify_df = sad_songs_spotify_df
            mood_songs_names_df = sad_songs_names_df
        elif dominant_mood == 'angry':
            mood_songs_spotify_df = angry_songs_spotify_df
            mood_songs_names_df = angry_songs_names_df
        elif dominant_mood == 'fear':
            mood_songs_spotify_df = fear_songs_spotify_df
            mood_songs_names_df = fear_songs_names_df
        elif dominant_mood == 'neutral':
            mood_songs_spotify_df = romantic_songs_spotify_df
            mood_songs_names_df = romantic_songs_names_df

        col1, col2, col3 = st.columns(3)

        with col1:
            if detected_frame is not None:
                st.subheader("Captured Frame with Detected Mood:")
                st.image(detected_frame, channels="BGR", use_column_width=True)

        with col2:
            st.subheader("Recommended Spotify Songs:")
            for index, row in mood_songs_spotify_df.head(5).iterrows():
                song_name = row['Title']
                spotify_link = row.get('Song URL', 'Not Available')
                st.write(f"Song Name: {song_name}")
                st.write(f"Spotify Link: {spotify_link}")

        with col3:
            st.subheader("Recommended YouTube Songs:")
            for index, row in mood_songs_names_df.head(5).iterrows():
                song_name = row['song']
                artist_name = row['artist']
                youtube_search_query = f"{song_name} {artist_name} {dominant_mood}"
                youtube_search_query_encoded = quote(youtube_search_query)
                youtube_search_link = f"https://www.youtube.com/results?search_query={youtube_search_query_encoded}"
                st.write(f"Song Name: {song_name} by {artist_name}")
                st.markdown(f"YouTube Search Link: [{song_name} on YouTube]({youtube_search_link})")

    else:
        st.warning("No dominant mood detected.")

if __name__ == "__main__":
    main()
