def rank(audio, proc, model, targets):
    audio = load_audio(audio)
    inputs = proc(
        audio, 
        return_tensors = 'pt',
        sampling_rate = 16000
    ).input_values

    with torch.no_grad():
        logits = logits = model(inputs).logits 

    probs = torch.nn.functional.softmax(logits, dim = -1)

    def calc_word_prob(word):
        # Tokenize the word into subword tokens
        tokenized = proc.tokenizer(word, add_special_tokens=False, return_tensors="pt")
        print(tokenized)
        token_ids = tokenized.input_ids[0]  # Get the token ids for the word
        print(f"Tokenized word '{word}' -> Token IDs: {token_ids}")

        # Remove special tokens (such as padding, begin/end-of-sequence)
        token_ids = [token_id for token_id in token_ids if token_id not in proc.tokenizer.all_special_ids]
        print(f"Filtered Token IDs for '{word}': {token_ids}")

        if not token_ids:
            return 0.0  # Return 0 if no valid tokens

        word_prob = 1.0 

        # Now, iterate through token_ids and find the corresponding probability
        for token_id in token_ids:
            # Get the probabilities for each token across all frames
            token_prob = probs[0, :, token_id].max().item()  # max across frames (axis 1)
            word_prob *= token_prob

        return word_prob 

    word_probs = {word: calc_word_prob(word) for word in targets} 
    sorted_word_probs = dict(sorted(word_probs.items(), key = lambda item: item[1], reverse = True))

    return sorted_word_probs
