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

How It Works
Grant Webcam Access: Upon visiting the web application, the user is prompted to allow access to their webcam.

Emotion Detection: Every 5 seconds, the system captures an image from the webcam, analyzes it, and identifies the user's dominant emotion using DeepFace.

Music Recommendation: Based on the detected emotion, the system fetches music recommendations and displays the songs along with links to YouTube and Spotify.

User Interaction: Users can explore personalized music recommendations in real-time, enhancing their mood.

Setup and Installation
Step 1: Clone the repository
bash
Copy
Edit
git clone https://github.com/your-username/emotion-aware-music-recommendation.git
cd emotion-aware-music-recommendation
Step 2: Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Step 3: Run the Application
Start the application with:

bash
Copy
Edit
streamlit run app.py
Your web application will be available at http://localhost:8501.

Project Structure
bash
Copy
Edit
emotion-aware-music-recommendation/
│
├── app.py                 # Main application file
├── requirements.txt       # List of dependencies
├── README.md              # Project documentation
├── data/                  # Data files (e.g., mood-based song CSVs)
└── assets/                # Images and other media
Future Enhancements
Support for more moods: Expanding the mood categories to include more nuanced emotions.

Improved recommendations: Use machine learning to continuously improve the recommendation algorithm based on user feedback.

Cross-platform support: Expand support for mobile platforms to increase accessibility.

User Profile: Allow users to create profiles to store and personalize recommendations.

References
Facial Expression Recognition: "Weighted Least Square (WLS), Gabor Filter, and Support Vector Machine (SVM)" - Ketki R. Kulkarni, Sahebrao B. Bagal.

Emotion-Based Recommendation System: "Image Pyramid, Histogram of Oriented Gradients, and Multiclass SVM" - H. Immanuel James et al.

Smart Music Player: "Facial Emotion Recognition and Music Mood Recommendation using CNN" - Shlok Gilda et al.

License
This project is licensed under the MIT License - see the LICENSE.md file for details.

vbnet
Copy
Edit

This **README.md** provides all the necessary information for anyone setting up or contributing to your Emotion-Aware Music Recommendation project. It includes installation instructions, usage, project structure, and planned enhancements.

Let me know if you need any more changes or additions!
