import librosa,os
import numpy as np
import soundfile as sf
from keras.models import load_model
from flask import Flask, request, jsonify

def extract_audio_sample(audio_path, sample_duration=2):
    # Load the audio file
    y, sr = sf.read(audio_path)

    # Find the total duration of the audio
    total_duration = len(y) / sr

    # Check if the total duration is greater than the sample duration
    if total_duration > sample_duration:
        # Calculate the maximum starting point for the random sample
        max_start_time = total_duration - sample_duration

        # Initialize the starting point for the random sample
        start_time = np.random.uniform(0, max_start_time)

        # Extract the random sample
        sample = y[int(start_time * sr):int((start_time + sample_duration) * sr)]

        return sf.write("sample.wav", sample, sr)
    else:
        # If total duration is not greater than sample duration, return original audio
        return sf.write("sample.wav", y, sr)

def extract_features(file_path, target_duration=2, sr=44100):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, sr=sr)

        # Check if the length of audio is greater than 0
        if len(audio) > 0:
            # Ensure target duration
            target_length = target_duration * sr
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            elif len(audio) > target_length:
                audio = audio[:target_length]

            # Feature extraction
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfccs_mean = np.mean(mfccs, axis=1)

            mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr), axis=1)
            chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12).T, axis=0)
            rms = np.mean(librosa.feature.rms(y=audio))
            zero_crossings = np.mean(librosa.feature.zero_crossing_rate(y=audio))

            # Concatenate all features into a single array
            features = np.concatenate((mfccs_mean, mel_spectrogram, chroma, [rms, zero_crossings]))

            return features
        else:
            print("Error: Audio length is 0.")
            return None

    except Exception as e:
        # Handle any exceptions that might occur during feature extraction
        print(f"Error in feature extraction: {e}")
        return None


mdl = load_model(r'audio_Class.h5')

app = Flask(__name__)

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    # Check if the request contains a file
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['audio']
    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    # Example: Extract features from audio and make predictions
    file = extract_audio_sample(file)
    audio_features = extract_features('sample.wav')
    os.remove('sample.wav')
    audio_features = np.expand_dims(audio_features, axis=0)  # Add batch dimension
    predict = mdl.predict(audio_features)
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predict)
    # Define the labels
    labels = ["real", "fake"]
    # Get the label corresponding to the predicted class
    predicted_label = labels[predicted_class_index]
    # Get the confidence percentage
    confidence_percentage = np.max(predict) * 100
    # You can return the prediction result and confidence percentage
    return jsonify({'result': predicted_label, 'confidence_percentage': confidence_percentage})


if __name__ == '__main__':
    app.run(port=8080)

#https://ttsmaker.com/