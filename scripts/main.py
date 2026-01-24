import torch
import numpy as np
import argparse
from src.config import Config
from src.data_utils import load_brown_words, reconstruct_sentences, replace_rare_words, clean_sentences, build_vocab, create_mapping, split_data, create_mlp_dataset
from src.model import init_parameters, train_one_epoch
from src.evaluate import evaluate, sample_next_word_idx
from src.baseline import train_em, calculate_perplexity, get_counts

import matplotlib.pyplot as plt

# download data set
'''
# Download latest version
path = kagglehub.dataset_download("nltkdata/brown-corpus")

print("Path to dataset files:", path)
'''

def get_args():
    parser = argparse.ArgumentParser(description="Bengio Trigram/MLP Experiment Runner")
    parser.add_argument('--device', type=str, default=None, help='Device to use on (cuda, mps, cpu); otherwise auto-detected ')
    
    # Model Architecture
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'baseline'])
    parser.add_argument('--emb_dim', type=int, default=30, help='Dimension of word embeddings')
    parser.add_argument('--hidden_dim', type=int, default=100, help='Size of hidden layer')
    
    # Training Hyperparameters
    parser.add_argument('--epsilon_t', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=1e-8)
    
    # Data Processing
    parser.add_argument('--shuffle', action="store_true", help='Shuffle sentences before split')
    parser.add_argument('--no-shuffle', dest="shuffle", action="store_false", help='Do not shuffle sentences before split')
    parser.set_defaults(shuffle=True)
    parser.add_argument('--context_size', type=int, default=5, help='Context size')

    args = parser.parse_args()
    return Config(**vars(args))

def get_device(requested_device=None):
    """
    Determines the best available device or uses the requested one.
    """
    if requested_device:
        return torch.device(requested_device)

    # Check for NVIDIA GPU (Cloud/Linux/Windows)
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # Check for Apple Silicon (Local Mac development)
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    
    # Fallback to CPU
    return torch.device("cpu")

def run(cfg):

    g = torch.Generator().manual_seed(2147483647)

    # 1. hyperparameters and setup
    print("Starting ...")
    device = get_device(cfg.device)
    print(f"Using device: {device}")

    # 2. Data pipeline
    words = load_brown_words("./brown")
    print(f"Size of full dataset on word level: {len(words)}")
    sentences = reconstruct_sentences(words)
    print(f"Size of full dataset on sentence level: {len(sentences)}")
    train_sents, val_sents, test_sents = split_data(sentences, shuffle=cfg.shuffle)
    train_sents_clean = replace_rare_words(train_sents, min_freq=3)

    vocab, vocab_size = build_vocab(train_sents_clean)

    print(f"Size of vocabulary: {vocab_size}")
    val_sents_clean, test_sents_clean = clean_sentences(val_sents, vocab), clean_sentences(test_sents, vocab)
    print(f"Size of train, val, test sets on sentence level: {len(train_sents_clean), len(val_sents_clean), len(test_sents_clean)}")
    
    w2i, i2w = create_mapping(vocab)

    # 3. model decision
    if cfg.model == "mlp":
        # MLP
        # 3.1 dataset preparation
        X_train, y_train = create_mlp_dataset(train_sents_clean, w2i)
        X_val, y_val = create_mlp_dataset(val_sents_clean, w2i)
        X_test, y_test = create_mlp_dataset(test_sents_clean, w2i)
        print("=================")
        print(f"Size of X_train, X_val, X_test using context size {cfg.context_size}: {X_train.shape, X_val.shape, X_test.shape}")

        X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
        X_val, y_val = torch.tensor(X_val), torch.tensor(y_val)
        X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)
        
        # 3.2 init mlp
        parameters = init_parameters(vocab_size, cfg.context_size, cfg.emb_dim, cfg.hidden_dim, device, g)
        
        # 3.3 train
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_pp': [],
            'val_pp': [],
            'lr': []
        }

        t = 0
        r = cfg.lr_decay * cfg.batch_size
        epsilon_0 = cfg.epsilon_t

        for epoch in range(cfg.epochs):
            
            # train for one epoch (that is len(X_train)/batch_size many steps)
            parameters, epoch_losses, lr_t, t = train_one_epoch(
                X_train, y_train, parameters,
                cfg.batch_size, cfg.context_size, cfg.emb_dim,
                epsilon_0, cfg.weight_decay, r, g, t, device)

            # compute and track train loss and perplexity
            avg_train_loss = np.mean(epoch_losses)
            history['train_loss'].append(avg_train_loss)
            history['train_pp'].append(np.exp(avg_train_loss))
            history['lr'].append(lr_t)

            # compute and track val loss and perplexity
            avg_val_loss, val_pp = evaluate(X_val, y_val, device, cfg.context_size, cfg.emb_dim, parameters)
            history['val_loss'].append(avg_val_loss)
            history['val_pp'].append(val_pp)

            # print progress
            print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Train PP {np.exp(avg_train_loss)} | Val Loss {avg_val_loss:.4f} | Val PP {val_pp:.2f}")
            print("==========================")
        
        # report final test loss
        avg_loss, perplexity = evaluate(X_test, y_test, device, cfg.context_size, cfg.emb_dim, parameters)
        print(f"Final test loss: {avg_loss} | Final test perplexity: {perplexity}")

        # 3.4 sample from model
        ix = torch.randint(0, X_val.shape[0], (1,)).item()
        context = X_val[ix].tolist()
        context = [w2i['<s>']] * 5
        print(f"Start sequence is: {[i2w[i] for i in context]}")

        generated_text = sample_next_word_idx(context, parameters, cfg.context_size, cfg.emb_dim, i2w)
        print(" ".join(generated_text))

        return history, avg_loss, perplexity

    else:
        # TRIGRAM
        # 3.1 train

        count_results = get_counts(train_sents_clean)
        T_counts = count_results[4]

        # optimize a(q_t) (alpha of q_t)
        alphas = train_em(val_sents_clean, count_results, vocab, vocab_size, T_counts)

        # report final results
        train_perplexity = calculate_perplexity(train_sents, count_results, alphas, vocab, vocab_size, T_counts)
        val_perplexity = calculate_perplexity(val_sents, count_results, alphas, vocab, vocab_size, T_counts)
        test_perplexity = calculate_perplexity(test_sents, count_results, alphas, vocab, vocab_size, T_counts)
        print(f"The final train perplexity of the interpolated trigram model is: {train_perplexity}")
        print(f"The final test perplexity of the interpolated trigram model is: {test_perplexity}")
        print(f"The final validation perplexity of the interpolated trigram model is: {val_perplexity}")
    
def main():
    cfg = get_args()
    history, avg_loss, perplexity = run(cfg)

if __name__ == "__main__":
    main()