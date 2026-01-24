import os
import torch
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

def load_brown_words(dataset_path):
    all_words = []
    # The Brown corpus files are often in a subfolder named 'brown'
    brown_dir = os.path.join(dataset_path, 'brown') if 'brown' not in dataset_path else dataset_path
    
    # Files are usually named like 'ca01', 'cb01' etc.
    for filename in sorted(os.listdir(brown_dir)):
        # Skip non-data files
        if filename.startswith('.') or filename.endswith('.zip') or filename == 'README':
            continue
            
        file_path = os.path.join(brown_dir, filename)
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # The Brown corpus often has 'word/tag' format. 
            # We only want the words.
            for line in f:
                for tagged_word in line.strip().split():
                    if '/' in tagged_word:
                        word = tagged_word.rsplit('/', 1)[0]
                        all_words.append(word)
                    else:
                        all_words.append(tagged_word)
    return all_words

def reconstruct_sentences(flat_list):
    sentences = []
    current_sentence = []
    
    # sign of end of sentence can be ".", "!" or "?"
    sentence_enders = {'.', '!', '?'}
    
    for word in flat_list:
        current_sentence.append(word)
        if word in sentence_enders:
            sentences.append(current_sentence)
            current_sentence = []
    return sentences

def replace_rare_words(sentences, min_freq=3):
    counts = Counter(word for sent in sentences for word in sent)
    keep_words = {word for word, count in counts.items() if count >= min_freq}
    
    processed_sentences = [
        [word if word in keep_words else '<UNK>' for word in sent]
        for sent in sentences
    ]
    return processed_sentences

# substitutes OOV words through UNK
def clean_sentences(sents, vocab):
    sents_clean = []
    for sent in sents:
        # Use training vocabulary V to determine UNK substitution
        safe_sent = [w if w in vocab else '<UNK>' for w in sent]
        sents_clean.append(safe_sent)
    return sents_clean


def build_vocab(train_sentences):
    
    vocab = set(word for sent in train_sentences for word in sent)
    vocab.update(['<s>', '</s>'])
    return vocab, len(vocab)

def create_mapping(vocab):
    sorted_vocab = sorted(list(vocab))
    word_to_idx = {word: i for i, word in enumerate(sorted_vocab)}
    idx_to_word = {i: word for i, word in enumerate(sorted_vocab)}
    return word_to_idx, idx_to_word

def split_data(sentences, shuffle):
    #sentences = list(sentences)

    if shuffle:
        random.seed(42)
        random.shuffle(sentences)
    else:
        pass
    
    num_sents = len(sentences)
    train_end = int(num_sents * 0.8)
    val_end = int(num_sents * 0.9)

    train_sents = sentences[:train_end]
    val_sents = sentences[train_end:val_end]
    test_sents = sentences[val_end:]

    return train_sents, val_sents, test_sents

def create_mlp_dataset(sentences, word_to_idx, context_size=5):
    """
    Converts a list of sentences into context-target pairs for the MLP.
    X: Matrix of context word indices.
    y: Vector of target word indices.
    """

    X = []
    y = []

    for sent in sentences:
        # Add padding as defined in the paper (2 start tokens, 1 end token)
        # Note: We use the same context as the Trigram baseline
        tokens = ['<s>'] * context_size + sent + ['</s>']

        # Sliding window over the tokens
        for i in range(context_size, len(tokens)):
            # Get indices for the context (previous n-1 words)
            context = [word_to_idx[tokens[j]] for j in range(i - context_size, i)]
            # Get index for the current target word
            target = word_to_idx[tokens[i]]

            X.append(context)
            y.append(target)

    return np.array(X), np.array(y)

def plot_training_results(history, test_loss, test_pp):
    epochs = range(len(history['train_loss']))

    # Create a figure with 3 subplots
    fig, ax = plt.subplots(1, 2, figsize=(21, 6))

    # 1. Plot Loss (Train vs Val)
    ax[0].plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', lw=2)
    ax[0].plot(epochs, history['val_loss'], label='Val Loss', color='#ff7f0e', lw=2)

    if test_loss is not None:
        ax[0].axhline(
            test_loss,
            color='black',
            linestyle='--',
            lw=2,
            label='Test Loss'
        )

    ax[0].set_title('Loss Develoment', fontsize=14)
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('cross entropy')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # 2. Plot Perplexity (Train vs Val) - Log Scale
    ax[1].plot(epochs, history['train_pp'], label='Train PP', color='#1f77b4', lw=2)
    ax[1].plot(epochs, history['val_pp'], label='Val PP', color='#ff7f0e', lw=2)

    if test_pp is not None:
        ax[1].axhline(
            test_pp,
            color='black',
            linestyle='--',
            lw=2,
            label='Test PP'
        )

    ax[1].set_yscale('log') # Log scale is best practice for Perplexity
    ax[1].set_title('Perplexity (Log Scale)', fontsize=14)
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('perplexity')
    ax[1].legend()
    ax[1].grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.show()