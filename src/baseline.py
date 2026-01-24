from collections import Counter
import math

def get_counts(sentences):
    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    context_freq = Counter() # for freq(w_{t-1}, w_{t-2})
    
    total_tokens = 0
    for sent in sentences:
        # Padding: Start- und End-Tokens
        tokens = ['<s>', '<s>'] + sent + ['</s>']
        for i in range(2, len(tokens)):
            w1, w2, w3 = tokens[i-2], tokens[i-1], tokens[i]
            
            unigrams[w3] += 1
            bigrams[(w2, w3)] += 1
            trigrams[(w1, w2, w3)] += 1
            context_freq[(w1, w2)] += 1
            total_tokens += 1
            
    return unigrams, bigrams, trigrams, context_freq, total_tokens

def get_base_probas(w1, w2, w3, counts, vocab_size, T):
    uni, bi, tri, ctx, total_tokens = counts
    
    p0 = 1.0 / vocab_size
    p1 = uni[w3] / T
    p2 = bi[(w2, w3)] / uni[w2] if uni[w2] > 0 else 0
    p3 = tri[(w1, w2, w3)] / ctx[(w1, w2)] if ctx[(w1, w2)] > 0 else 0
    
    return [p0, p1, p2, p3]

def get_bin(w1, w2, context_freq, T):
    freq = context_freq[(w1, w2)]
    # Formula: ceil(-log((1 + freq)/T))
    val = -math.log((1 + freq) / T)
    return math.ceil(val)

def train_em(val_data, train_counts, vocab, vocab_size, T, iterations=5):
    uni, bi, tri, ctx, _ = train_counts
    
    # 1. Trigramme im Val-Set vorbereiten & Bins zuordnen
    val_samples = []
    bins_in_val = set()
    
    for sent in val_data:
        # substitute every word that is no in the train V but in the test set with <UNK> to simplify
        safe_sent = [w if w in vocab else '<UNK>' for w in sent]
        tokens = ['<s>', '<s>'] + safe_sent + ['</s>']
        
        for i in range(2, len(tokens)):
            w1, w2, w3 = tokens[i-2], tokens[i-1], tokens[i]
            q = get_bin(w1, w2, ctx, T)
            bins_in_val.add(q)
            val_samples.append((w1, w2, w3, q))
            
    # 2. Initialisiere Alphas für jeden gefundenen Bin
    alphas = {q: [0.25, 0.25, 0.25, 0.25] for q in bins_in_val}
    
    # 3. EM-Schleife
    for _ in range(iterations):
        expected_counts = {q: [0.0, 0.0, 0.0, 0.0] for q in bins_in_val}
        
        for w1, w2, w3, q in val_samples:
            p_bases = get_base_probas(w1, w2, w3, train_counts, vocab_size, T)
            a = alphas[q]
            
            # P_total = sum(alpha_i * p_i)
            total_p = sum(a[i] * p_bases[i] for i in range(4))
            
            if total_p > 0:
                for i in range(4):
                    # Expectation step: Wie viel "Schuld" hat p_i an total_p?
                    expected_counts[q][i] += (a[i] * p_bases[i]) / total_p
        
        # Maximization step: Update alphas pro Bin
        for q in bins_in_val:
            row_sum = sum(expected_counts[q])
            if row_sum > 0:
                alphas[q] = [expected_counts[q][i] / row_sum for i in range(4)]
                
    return alphas

def calculate_perplexity(data, counts, alphas, vocab, vocab_size, T):
    log_prob_sum = 0
    N = 0
    ctx = counts[3]
    
    for sent in data:
        # substitute every word that is no in the train V but in the test set with <UNK> to simplify
        safe_sent = [w if w in vocab else '<UNK>' for w in sent]
        tokens = ['<s>', '<s>'] + safe_sent + ['</s>']
        
        for i in range(2, len(tokens)):
            w1, w2, w3 = tokens[i-2], tokens[i-1], tokens[i]
            q = get_bin(w1, w2, ctx, T)
            
            # Falls ein Bin im Testset vorkommt, den wir im Val-Set nicht hatten:
            # Nutze einen Default-Bin oder den nächsten Nachbarn
            if q not in alphas:
                # Einfacher Fallback: Uniforme Gewichte oder globaler Durchschnitt
                a = [0.25, 0.25, 0.25, 0.25]
            else:
                a = alphas[q]
                
            p_bases = get_base_probas(w1, w2, w3, counts, vocab_size, T)
            p_interp = sum(a[j] * p_bases[j] for j in range(4))
            
            log_prob_sum += math.log2(p_interp)
            N += 1
            
    return 2**(-log_prob_sum / N)