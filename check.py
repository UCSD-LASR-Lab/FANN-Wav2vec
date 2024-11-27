import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import numpy as np
import warnings
import os
import math
import csv

os.environ['PYTORCH_MPS_DEVICE_OVERRIDE'] = '0'
warnings.filterwarnings("ignore")  

def get_predictions_and_probabilities(audio_path, candidate_words, processor, model):
    try:
        # Load and process audio
        print(f"Processing audio: {os.path.basename(audio_path)}...")
        audio, rate = librosa.load(audio_path, sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = inputs.to('cpu')
        
        with torch.no_grad():
            # Get logits and convert to log probabilities
            logits = model(inputs.input_values).logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Get the model's predicted transcription
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_transcription = processor.batch_decode(predicted_ids)[0]
            print(f"Predicted transcription: '{predicted_transcription}'")
            
            # Prepare log_probs for CTC loss
            log_probs = log_probs.transpose(0, 1)  # [sequence_length, batch_size, num_labels]
            
            # Get vocabulary
            vocab = processor.tokenizer.get_vocab()
            
            results = {}
            for word in candidate_words:
                print(f"Analyzing '{word}':")
                word_upper = word.upper()  # Convert to uppercase for wav2vec2
                
                # Convert word to character IDs
                char_ids = []
                for char in word_upper:
                    if char in vocab:
                        char_ids.append(vocab[char])
                    else:
                        print(f"Warning: Character '{char}' not in vocabulary")
                
                if not char_ids:
                    print(f"No valid characters found for word '{word}'")
                    results[word] = float('-inf')
                    continue
                
                # Prepare tensors for CTC loss
                target = torch.tensor([char_ids], dtype=torch.long)
                target_length = torch.tensor([len(char_ids)], dtype=torch.long)
                input_length = torch.tensor([log_probs.size(0)], dtype=torch.long)
                
                # Compute CTC loss
                ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
                loss = ctc_loss(log_probs, target, input_length, target_length)
                
                # Convert loss to probability
                word_prob = math.exp(-loss.item())
                results[word] = word_prob
                print(f"Probability for '{word}': {word_prob:.4f}")
            
            # Sort results by probability
            sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
            
            return predicted_transcription, sorted_results
            
    except Exception as e:
        print(f"\nError processing {audio_path}: {str(e)}")
        return None, None

def process_dataset(audio_dir, word_list_file, output_file):
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load model and processor once
    print("Loading model and processor...")
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h",
        cache_dir="./model_cache"
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base-960h",
        cache_dir="./model_cache"
    ).to(device)
    print("Model loaded successfully")

    # Read word list file
    results = []
    with open(word_list_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:  # Skip empty rows
                continue
            audio_file = row[0]
            candidate_words = row[1:]
            
            audio_path = os.path.join(audio_dir, audio_file)
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}")
                continue
                
            # Process audio file
            transcription, probabilities = get_predictions_and_probabilities(
                audio_path, 
                candidate_words,
                processor,
                model
            )
            
            if probabilities:
                # Store results
                result = {
                    'audio_file': audio_file,
                    'transcription': transcription,
                    'probabilities': probabilities
                }
                results.append(result)
                
                # Print interim results
                print(f"\n=== Results for {audio_file} ===")
                print(f"Transcription: '{transcription}'")
                print("Probabilities:")
                for word, prob in probabilities.items():
                    print(f"{word}: {prob:.4f}")
            
    # Write results to output file
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['audio_file', 'transcription', 'candidate_word', 'probability'])
        # Write results
        for result in results:
            audio_file = result['audio_file']
            transcription = result['transcription']
            for word, prob in result['probabilities'].items():
                writer.writerow([audio_file, transcription, word, f"{prob:.4f}"])

if __name__ == "__main__":
    audio_dir = "/Users/nathanko/charsiu/local/audio"
    word_list_file = "/Users/nathanko/charsiu/local/text.csv"
    output_file = "/Users/nathanko/charsiu/local/results.csv"
    
    process_dataset(audio_dir, word_list_file, output_file)