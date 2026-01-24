import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def evaluate(X, y, device, context_size, embedding_dims, parameters):
    # Switch to evaluation mode (not strictly necessary for raw tensors but good practice)
    model_loss = []
    C, W1, b1, W2, b2 = parameters

    # Disable gradient tracking for speed and memory efficiency
    with torch.no_grad():
        # Process in larger batches to speed up evaluation
        eval_batch_size = 1024
        for i in range(0, X.shape[0], eval_batch_size):
            X_batch = X[i:i+eval_batch_size].to(device)
            y_batch = y[i:i+eval_batch_size].to(device)

            # 1. Forward Pass
            emb = C[X_batch]
            h = torch.tanh(emb.view(-1,context_size * embedding_dims) @ W1 + b1)
            # Using the architecture including Direct Connections if applicable
            logits = h @ W2 + b2

            # 2. Calculate Loss
            loss = F.cross_entropy(logits, y_batch)
            model_loss.append(loss.item())

    avg_loss = sum(model_loss) / len(model_loss)
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity

def sample_next_word_idx(context, parameters, context_size, emb_dim, i2w, temperature=1.0):
    
    C, W1, b1, W2, b2 = parameters
    generated_text = []
    for _ in range(20):
        with torch.no_grad():
            x = torch.tensor([context])

            emb = C[x]
            h = torch.tanh(emb.view(-1, context_size * emb_dim) @ W1 + b1)
            logits = (h @ W2 + b2)

            # temperature scaling = multinomial sampling
            logits = logits / temperature
            probs = F.softmax(logits, dim=1)

            next_word_idx = torch.multinomial(probs, num_samples=1).item()
            generated_text.append(i2w[next_word_idx])
            context = context[1:] + [next_word_idx]
            
    return generated_text