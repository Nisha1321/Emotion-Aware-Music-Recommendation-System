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
```


## How It Works

### 1. **Grant Webcam Access**
Upon visiting the web application, the user is prompted to allow access to their webcam. This is required for capturing live images of the user’s face, which will be processed to detect their emotional state.

### 2. **Emotion Detection**
Every 5 seconds, the system captures an image from the webcam, processes it using **DeepFace**, and identifies the user's dominant emotion. The emotions detected can include:

- Happy
- Sad
- Angry
- Neutral
- Surprised
- Fearful

The emotion detection is powered by advanced deep learning algorithms, allowing for high accuracy in recognizing facial expressions.

### 3. **Music Recommendation**
Based on the detected emotion, the system fetches personalized music recommendations. It selects 5 songs from **YouTube** and **Spotify**, displaying them with direct links to the platforms. The recommendations aim to match the user’s current mood and enhance their emotional experience.

### 4. **User Interaction**
Users can explore these personalized music recommendations in real-time, adjusting their mood with the music tailored to their detected emotion. They can click on the provided links to listen to the suggested tracks on YouTube or Spotify.

## Future Enhancements

- **Support for More Moods**: Expand the mood categories to include more nuanced emotions, offering a wider variety of music suggestions based on different emotional states.
- **Improved Recommendations**: Implement machine learning techniques to continuously improve the music recommendation algorithm. User feedback will help refine and personalize suggestions over time.
- **Cross-Platform Support**: Make the application available on mobile platforms, broadening access and allowing users to use the system on smartphones and tablets.
- **User Profiles**: Allow users to create personal profiles to store their music preferences and emotional history, enabling even more customized music suggestions in the future.

## References

1. **Facial Expression Recognition**: "Weighted Least Square (WLS), Gabor Filter, and Support Vector Machine (SVM)" - Ketki R. Kulkarni, Sahebrao B. Bagal.
2. **Emotion-Based Recommendation System**: "Image Pyramid, Histogram of Oriented Gradients, and Multiclass SVM" - H. Immanuel James et al.
3. **Smart Music Player**: "Facial Emotion Recognition and Music Mood Recommendation using CNN" - Shlok Gilda et al.



