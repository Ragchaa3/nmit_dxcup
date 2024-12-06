from flask import Flask, request, render_template
import librosa
import numpy as np
import os
import tempfile
from tensorflow.keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

# Load TensorFlow Lite model
model = load_model("my_model.h5")

def preprocessing(audio_file, mode):
    # we want to resample audio to 16 kHz
    sr_new = 16000 # 16kHz sample rate
    x, sr = librosa.load(audio_file, sr=sr_new)

    # padding sound 
    # because duration of sound is dominantly 20 s and all of sample rate is 22050
    # we want to pad or truncated sound which is below or above 20 s respectively
    max_len = 5 * sr_new  # length of sound array = time x sample rate
    if x.shape[0] < max_len:
      # padding with zero
      pad_width = max_len - x.shape[0]
      x = np.pad(x, (0, pad_width))
    elif x.shape[0] > max_len:
      # truncated
      x = x[:max_len]
    
    if mode == 'mfcc':
      feature = librosa.feature.mfcc(y=x, sr=sr_new)
    
    # elif mode == 'log_mel':
    #   feature = librosa.feature.melspectrogram(y=x, sr=sr_new, n_mels=128, fmax=8000)
    #   feature = librosa.power_to_db(feature, ref=np.max)
    
    feature = feature.reshape((-1, 20, 157, 1))
    return feature
def predict(features):
    res = model.predict(features)
    
    # Return the prediction (assuming single output)
    return res

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Receive the .wav file
        file = request.files['fileToUpload']

        if file:
            # Save the file to a temporary location
          with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            file.save(temp_file.name)
        
            # Now that the file is saved and the temp file is closed, you can delete it
            os.remove(temp_file.name)

            # Render the result
            
        return render_template('result.html', prediction="prediction[0]")

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)