import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# === Load model, scaler, and dataset ===
model = joblib.load("models/rf_mood_model.joblib")
scaler = joblib.load("models/scaler.joblib")
df = pd.read_csv("train.csv")

st.title("🎵 Emotion-Based Music Classifier")
st.write("Enter a track name to predict its mood!")

# === User input ===
track_name_input = st.text_input("Track name")

if track_name_input:
    # Find matches
    matches = df[df['track_name'].str.lower().str.contains(track_name_input.lower(), na=False)]

    if matches.empty:
        st.warning("❌ No matching tracks found.")
    else:
        # Create a dropdown to choose track
        matches['display'] = matches['track_name'] + " — " + matches['artists'] + " (" + matches['album_name'] + ")"
        selected_display = st.selectbox("Select a track:", matches['display'])
        track = matches[matches['display'] == selected_display].iloc[0]

        # Features used in training
        features = ['danceability', 'energy', 'valence', 'tempo', 
                    'acousticness', 'instrumentalness', 'loudness']
        X = scaler.transform([track[features].values])
        
        # Predict mood
        predicted_mood = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        classes = model.classes_

        # === Display results ===
        st.subheader("🎶 Track Details")
        st.write(f"**Track:** {track['track_name']}")
        st.write(f"**Artist:** {track['artists']}")
        st.write(f"**Album:** {track['album_name']}")
        st.write(f"**Predicted Mood:** 🎯 {predicted_mood}")

        # === Plot probability bar chart ===
        fig, ax = plt.subplots(figsize=(6,3))
        colors = ['skyblue', 'salmon', 'lightgreen', 'orange']
        ax.bar(classes, probs, color=colors)
        ax.set_ylabel("Probability")
        ax.set_title("Mood Probabilities")
        st.pyplot(fig)
