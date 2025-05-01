# Emotion-Aware Music Recommendation

## Project Overview

The **Emotion-Aware Music Recommendation** system is a web-based application designed to provide personalized music recommendations based on the user's current emotional state. By leveraging **facial emotion recognition technology**, the system analyzes the user's facial expressions and suggests music from popular platforms like **YouTube** and **Spotify** that aligns with the detected mood.

The application provides a seamless, dynamic, and interactive experience, enhancing the music-listening journey by tailoring the recommendations to the user’s emotional context in real-time.

## Features

- **Real-Time Emotion Detection**: The system captures the user’s facial expression every 5 seconds using the webcam and determines the dominant emotion (Happy, Sad, Angry, Neutral, Surprised, or Fearful).
- **Personalized Music Recommendations**: Based on the identified mood, the system fetches 5 YouTube and 5 Spotify song recommendations.
- **Web-Based Interface**: A user-friendly interface built using **Streamlit** for easy interaction with the system.

## Technologies Used

- **Python**: The main programming language.
- **OpenCV**: For capturing video and processing images from the webcam.
- **Streamlit**: To build the interactive web application.
- **DeepFace**: A deep learning-based library for emotion recognition from facial expressions.
- **Pandas**: For data manipulation and analysis.
- **Spotify API & YouTube API**: For fetching music recommendations from Spotify and YouTube.

## Prerequisites

Before you run the application, ensure that you have the following installed:

- **Python 3.x** or higher
- **Libraries**: OpenCV, Streamlit, DeepFace, Pandas

Install required libraries via `pip`:

```bash
pip install opencv-python streamlit deepface pandas
