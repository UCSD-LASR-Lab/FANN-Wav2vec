import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import pandas as pd
import numpy as np
import os

# Loading Wav2Vec pretrained model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def load_audio(filename, sample_rate=16000):
    audio, rate = librosa.load(filename, sr = sample_rate)
    return audio

def transcribe_each(audio):
    # Taking an input value
    input_values = tokenizer(audio, return_tensors = "pt").input_values

    # Storing logits (non-normalized prediction values)
    logits = model(input_values).logits

    # Storing predicted ids
    prediction = torch.argmax(logits, dim = -1)

    # Passing the prediction to the tokenzer decode to get the transcription
    transcription = tokenizer.batch_decode(prediction)[0]
    
    return transcription

def transcribe(input_folder, output_excel):
    # List audio files in input folder
    audio_files = os.listdir(input_folder)

    # Create empty datadrame
    df = pd.DataFrame(columns=['File', 'Actual', 'Predicted'])
    files = []
    actuals = []
    predicted = []

    for audio_file in audio_files:
        input_audio_file = os.path.join(input_folder, audio_file)

        # Check if it's a wav file (not a folder)
        if os.path.isfile(input_audio_file) and ".wav" in input_audio_file:

            try: 
                # Transcribe .wav files 
                audio = load_audio(input_audio_file)
                transcription_each = transcribe_each(audio)
                files.append(input_audio_file)

                # Extract actual word
                actual = input_audio_file[input_audio_file.rfind('_')+1:input_audio_file.rfind('.')]    # KIDTALK & AP

                actuals.append(actual)
                predicted.append(transcription_each)
            except RuntimeError as error:
                print("Can't transcribe", audio_file)

    # Write to excel file 
    df['File'] = files
    df['Actual'] = actuals
    df['Predicted'] = predicted
    df.to_excel(output_excel)

def transcribe_all(folder_of_folders):
    folders = os.listdir(folder_of_folders)

    for folder in folders:
        dir = os.path.join(folder_of_folders, folder)
        if os.path.isdir(dir):
            excel = folder + ".xlsx"
            transcribe(dir, excel)

folder = input("Enter folder name:")
output = input("Enter output Excel name:")
transcribe(folder, output)
print("All files transcribed.")