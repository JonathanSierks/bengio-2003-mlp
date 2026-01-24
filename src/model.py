import torch
import torch.nn.functional as F


def init_parameters(vocab_size, context_size, emb_dim, hidden_dim, device, g):
    # create parameter tensors
    C = torch.randn((vocab_size, emb_dim), generator=g)
    W1 = torch.randn((emb_dim * context_size, hidden_dim), generator=g)
    b1 = torch.randn(hidden_dim, generator=g)
    W2 = torch.randn(hidden_dim, vocab_size, generator=g)
    b2 = torch.randn(vocab_size, generator=g)

    # override tensors with kaiming init
    torch.nn.init.xavier_normal_(W1)
    torch.nn.init.xavier_normal_(W2)

    torch.nn.init.normal_(C, mean=0, std=0.1)

    # biases to 0
    with torch.no_grad():
        b1.fill_(0)
        b2.fill_(0)

    C = C.to(device)
    W1 = W1.to(device)
    b1 = b1.to(device)
    W2 = W2.to(device)
    b2 = b2.to(device)

    parameters = [C, W1, b1, W2, b2]

    for p in parameters:
        p.requires_grad = True
    
    return parameters

def train_one_epoch(
        X_train, y_train, parameters, 
        batch_size, context_size, emb_dim,
        epsilon_0, weight_decay, r, g, t, device):
    
    C, W1, b1, W2, b2 = parameters

    epoch_losses = []
    lr_t_history = []

    for _ in range(len(X_train)//batch_size):
        t += 1
        lr_t = epsilon_0 / (1+r*t)
        lr_t_history.append(lr_t)

        # batch selection
        start = torch.randint(0, X_train.shape[0], (batch_size,), generator=g)
        X_batch, y_batch = X_train[start], y_train[start]
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # forward pass
        emb = C[X_batch]
        h = torch.tanh(emb.view(-1,context_size * emb_dim) @ W1 + b1)
        logits = (h @ W2 + b2)

        # calculate loss (negative log likelihood)
        loss = F.cross_entropy(logits, y_batch)
        epoch_losses.append(loss.item())

        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # update
        with torch.no_grad():
            C.data += -lr_t * (C.grad + weight_decay * C.data)
            W1.data += -lr_t * (W1.grad + weight_decay * W1.data)
            b1.data += -lr_t * b1.grad
            W2.data += -lr_t * (W2.grad + weight_decay * W2.data)
            b2.data += -lr_t * b2.grad
    parameters = [C, W1, b1, W2, b2]

    return parameters, epoch_losses, lr_t_history[-1], t


