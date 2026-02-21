from flask import Flask, render_template, request, redirect, url_for
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the model, scaler, and label encoder
model = load_model('my_model_gpu91.keras')
scaler = joblib.load('scaler2.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Feature extraction function
# Feature extraction function
def extract_features(data, sample_rate):
    try:
        # Extract features
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)

        # Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=sample_rate)
        mel = np.mean(mel_spectrogram.T, axis=0)

        # MFCC (Mel-Frequency Cepstral Coefficients)
        mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)

        # Spectral Contrast and Tonnetz
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=data, sr=sample_rate).T, axis=0)

        # Combine all features
        features = np.concatenate([zcr, chroma_stft, rms, mel, mfcc, spectral_contrast, tonnetz])
        print(f"Extracted features shape: {features.shape}")  # Debug statement
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Function to process uploaded audio
def process_audio(file_path):
    try:
        # Load audio file
        data, sample_rate = librosa.load(file_path, duration=2.5, offset=0.6)
        data = data / np.max(np.abs(data))  # Normalize audio
        print(f"Loaded audio data shape: {data.shape}")  # Debug statement

        # Extract features
        features = extract_features(data, sample_rate)
        if features is not None:
            features = features.reshape(1, -1)  # Reshape for model input
            print(f"Reshaped features shape: {features.shape}")  # Debug statement
            features = scaler.transform(features)  # Normalize features
            print(f"Scaled features shape: {features.shape}")  # Debug statement
            features = features.reshape(features.shape[0], features.shape[1], 1)  # Reshape for CNN-RNN input
            print(f"Final features shape for model: {features.shape}")  # Debug statement
            return features
        else:
            return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process the audio file
        features = process_audio(file_path)
        if features is not None:
            # Make a prediction
            prediction = model.predict(features)[0]
            print(f"Raw prediction: {prediction}")  # Debug statement
            predicted_emotion = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            return render_template('index.html', prediction=predicted_emotion)
        else:
            return render_template('index.html', prediction="Error processing audio file.")
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)