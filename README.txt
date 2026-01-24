

Context: N-gram models glue many mall pieces of characters/ words together, thus leaving out a ton of available information that spans over more than n characters/ words. N-gram models thus
1) Do not take a context size of greater then n words/ characters into account
2) Do not take the similarity between words into account. 

Problem: N-gram model fail to predict the next word in sentences like "A dog was running ..." if they only saw sentences close to it (e.g. "The cat is walking ...") but not the exact smae sentence in training. Furthermore the curse of dimensionalty applies since there are extremely many possible combinations of words in a sentence if every word is seen as 100% unique/distinct.

Solution: 
1) Words like "cat" and "dog" or "running" and "walking" receive similiar feature-vectors (embeddings). This will enable the model to generalize even if the training context was quite narrow (since through the appearance of a word like "cat" in a specific context we somewhat learn something about all neighbors of "cat" e.g. "dog"). Through this we tackle the curse of dimensionality since we distribute the probability mass intelligently in the space of possible predictions; therefore we also have to learn the word feature vector.
2) We increase the context size by using more nodes in a MLP

Finding: xxx


Bytes level approach (256)

daten laden und preprocessen
baseline modell trainnieren --> chat 
actual MLP modell implementieren; übernehmen aus dem was bisher gemacht
    aufteilen in MLP, train, eval

readme aufbereiten github; struktur überlegen wie das ganze mit command line anweisungen zum laufen gebracht werden kann; chat fragen
