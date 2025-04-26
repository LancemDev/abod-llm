import logging
import re
from flask import Flask, request, jsonify
from openai import OpenAI
import numpy as np
from collections import deque
import time
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
logging.info("Loaded environment variables.")

# Flask app
app = Flask(__name__)
logging.info("Flask app initialized.")

# OpenAI setup
openai_client = OpenAI(api_key=OPENAI_KEY)
logging.info("OpenAI client initialized.")

# Store recent pulse rate averages (last 3 intervals, ~45 seconds)
pulse_history = deque(maxlen=3)  # Stores dicts: {"pulse": float, "timestamp": float}
logging.info("Pulse history deque initialized.")

# Determine mood based on pulse rate and trend
def infer_mood(pulse, history):
    logging.debug(f"Inferring mood for pulse: {pulse}, history: {history}")
    if len(history) >= 2:
        recent_pulses = [h["pulse"] for h in history]
        trend = "rising" if recent_pulses[-1] > recent_pulses[-2] else "falling" if recent_pulses[-1] < recent_pulses[-2] else "stable"
    else:
        trend = "stable"
    logging.debug(f"Computed trend: {trend}")

    if pulse > 100 and trend in ["rising", "stable"]:
        return "excited"
    elif pulse < 80 and trend in ["falling", "stable"]:
        return "chill"
    elif 80 <= pulse <= 100:
        return "festive"
    else:
        return "hyped"

@app.route('/')
def index():
    logging.info("Index route accessed.")
    return "Welcome to the DJ Agent API!"

# API to receive sensor data (pulse rate)
@app.route('/sensor', methods=['POST'])
def process_sensor():
    try:
        data = request.json
        logging.debug(f"Received sensor data: {data}")
        pulse = float(data.get('pulse', 80))  # Average pulse rate

        # Update pulse history
        pulse_history.append({"pulse": pulse, "timestamp": time.time()})
        logging.info(f"Updated pulse history: {list(pulse_history)}")

        # Infer mood
        mood = infer_mood(pulse, pulse_history)
        logging.info(f"Inferred mood: {mood}")

        # LLM: Recommend song, artist, and lighting
        prompt = (
            f"Crowd mood is {mood} based on average pulse rate {pulse} BPM. "
            f"Pulse history: {[h['pulse'] for h in pulse_history]}. "
            "Suggest a song, artist, and lighting color to match the mood in the format: "
            "Song: <song>, Artist: <artist>, Lighting: <color>"
        )
        logging.debug(f"Generated prompt for OpenAI: {prompt}")
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a DJ agent that recommends songs, artists, and lighting colors based on crowd mood."},
                {"role": "user", "content": prompt}
            ]
        )
        recommendation = response.choices[0].message.content
        logging.info(f"Received recommendation from OpenAI: {recommendation}")

        # Parse recommendation with regex
        song_match = re.search(r"Song:\s*([^,]+)", recommendation)
        artist_match = re.search(r"Artist:\s*([^,]+)", recommendation)
        lighting_match = re.search(r"Lighting:\s*(\w+)", recommendation)

        song = song_match.group(1).strip() if song_match else "Sweet but Psycho"
        artist = artist_match.group(1).strip() if artist_match else "Ava Max"
        color = lighting_match.group(1).strip() if lighting_match else "red"
        logging.debug(f"Parsed recommendation - Song: {song}, Artist: {artist}, Lighting: {color}")

        return jsonify({
            "mood": mood,
            "song": song,
            "artist": artist,
            "lighting": color,
            "status": "success"
        })
    except Exception as e:
        logging.error(f"Error in /sensor route: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# API to communicate with Spotify server
@app.route('/spotify', methods=['POST'])
def process_spotify():
    try:
        data = request.json
        logging.debug(f"Received Spotify data: {data}")
        current_song = data.get('current_song', "Unknown")
        current_artist = data.get('current_artist', "Unknown")
        queue = data.get('queue', [])  # List of {"song": str, "artist": str}

        # Get latest pulse rate and mood
        latest_pulse = pulse_history[-1]["pulse"] if pulse_history else 80
        mood = infer_mood(latest_pulse, pulse_history)
        logging.info(f"Latest pulse: {latest_pulse}, inferred mood: {mood}")

        # LLM: Recommend song/artist to update queue
        queue_str = ", ".join([f"{item['song']} by {item['artist']}" for item in queue])
        prompt = (
            f"Crowd mood is {mood} based on pulse rate {latest_pulse} BPM. "
            f"Current song: {current_song} by {current_artist}. "
            f"Current queue: {queue_str if queue_str else 'empty'}. "
            f"Pulse history: {[h['pulse'] for h in pulse_history]}. "
            "Suggest a song and artist to add to the queue in the format: "
            "Song: <song>, Artist: <artist>"
        )
        logging.debug(f"Generated prompt for OpenAI: {prompt}")
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a DJ agent that recommends songs and artists to update a Spotify queue based on crowd mood and current playback."},
                {"role": "user", "content": prompt}
            ]
        )
        recommendation = response.choices[0].message.content
        logging.info(f"Received recommendation from OpenAI: {recommendation}")

        # Parse recommendation with regex
        song_match = re.search(r"Song:\s*([^,]+)", recommendation)
        artist_match = re.search(r"Artist:\s*([^,]+)", recommendation)

        song = song_match.group(1).strip() if song_match else "Uptown Funk"
        artist = artist_match.group(1).strip() if artist_match else "Mark Ronson"
        logging.debug(f"Parsed recommendation - Song: {song}, Artist: {artist}")

        return jsonify({
            "song": song,
            "artist": artist,
            "status": "success"
        })
    except Exception as e:
        logging.error(f"Error in /spotify route: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Run Flask server
if __name__ == "__main__":
    logging.info("Starting Flask server.")
    app.run(host="0.0.0.0", port=5000, debug=True)