from flask import Flask, request, jsonify
from openai import OpenAI
import numpy as np
from collections import deque
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_KEY=os.getenv("OPENAI_API_KEY")

# Flask app
app = Flask(__name__)

# OpenAI setup
openai_client = OpenAI(api_key=OPENAI_KEY)

# Store recent pulse rate averages (last 3 intervals, ~45 seconds)
pulse_history = deque(maxlen=3)  # Stores dicts: {"pulse": float, "timestamp": float}

# Determine mood based on pulse rate and trend
def infer_mood(pulse, history):
    # Compute trend (rising, stable, falling)
    if len(history) >= 2:
        recent_pulses = [h["pulse"] for h in history]
        trend = "rising" if recent_pulses[-1] > recent_pulses[-2] else "falling" if recent_pulses[-1] < recent_pulses[-2] else "stable"
    else:
        trend = "stable"

    # Mood thresholds
    if pulse > 100 and trend in ["rising", "stable"]:
        return "excited"
    elif pulse < 80 and trend in ["falling", "stable"]:
        return "chill"
    elif 80 <= pulse <= 100:
        return "festive"
    else:
        return "hyped"

# API to receive sensor data (pulse rate)
@app.route('/sensor', methods=['POST'])
def process_sensor():
    try:
        data = request.json
        pulse = float(data.get('pulse', 80))  # Average pulse rate

        # Update pulse history
        pulse_history.append({"pulse": pulse, "timestamp": time.time()})

        # Infer mood
        mood = infer_mood(pulse, pulse_history)

        # LLM: Recommend song, artist, and lighting
        prompt = (
            f"Crowd mood is {mood} based on average pulse rate {pulse} BPM. "
            f"Pulse history: {[h['pulse'] for h in pulse_history]}. "
            "Suggest a song, artist, and lighting color to match the mood."
        )
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a DJ agent that recommends songs, artists, and lighting colors based on crowd mood."},
                {"role": "user", "content": prompt}
            ]
        )
        recommendation = response.choices[0].message.content
        # Parse recommendation (assume format: "Song: [name], Artist: [artist], Lighting: [color]")
        try:
            song = recommendation.split("Song: ")[1].split(",")[0].strip()
            artist = recommendation.split("Artist: ")[1].split(",")[0].strip()
            color = recommendation.split("Lighting: ")[1].strip()
        except IndexError:
            # Fallback if parsing fails
            song, artist, color = "Sweet but Psycho", "Ava Max", "red"

        return jsonify({
            "mood": mood,
            "song": song,
            "artist": artist,
            "lighting": color,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# API to communicate with Spotify server
@app.route('/spotify', methods=['POST'])
def process_spotify():
    try:
        data = request.json
        current_song = data.get('current_song', "Unknown")
        current_artist = data.get('current_artist', "Unknown")
        queue = data.get('queue', [])  # List of {"song": str, "artist": str}

        # Get latest pulse rate and mood
        latest_pulse = pulse_history[-1]["pulse"] if pulse_history else 80
        mood = infer_mood(latest_pulse, pulse_history)

        # LLM: Recommend song/artist to update queue
        queue_str = ", ".join([f"{item['song']} by {item['artist']}" for item in queue])
        prompt = (
            f"Crowd mood is {mood} based on pulse rate {latest_pulse} BPM. "
            f"Current song: {current_song} by {current_artist}. "
            f"Current queue: {queue_str if queue_str else 'empty'}. "
            f"Pulse history: {[h['pulse'] for h in pulse_history]}. "
            "Suggest a song and artist to add to the queue to match the mood."
        )
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a DJ agent that recommends songs and artists to update a Spotify queue based on crowd mood and current playback."},
                {"role": "user", "content": prompt}
            ]
        )
        recommendation = response.choices[0].message.content
        # Parse recommendation (assume format: "Song: [name], Artist: [artist]")
        try:
            song = recommendation.split("Song: ")[1].split(",")[0].strip()
            artist = recommendation.split("Artist: ")[1].strip()
        except IndexError:
            song, artist = "Uptown Funk", "Mark Ronson"

        return jsonify({
            "song": song,
            "artist": artist,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Run Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)