import wordfreq
import stanza
import json
import os

def main():
    print("Loading Stanza Pipeline...")
    # use_gpu=True for Colab
    nlp = stanza.Pipeline('de', processors='tokenize,pos', use_gpu=True, logging_level='WARN')
    
    print("Gathering German vocabulary from wordfreq...")
    words = []
    # Loop over the wordlist. Zipf >= 2.5 is a very safe lower bound (covers > 100k words)
    for w in wordfreq.iter_wordlist('de'):
        if wordfreq.zipf_frequency(w, 'de') >= 2.5:
            words.append(w)
            
    print(f"Total words to pre-compute: {len(words)}")
    
    pos_cache = {}
    batch_size = 1000 # Safe batch size
    
    print("Tagging words in batches...")
    for i in range(0, len(words), batch_size):
        chunk = words[i:i+batch_size]
        # CAPITALIZE before passing to Stanza to accurately detect Nouns
        in_docs = [stanza.Document([], text=w.capitalize()) for w in chunk]
        
        try:
            out_docs = nlp(in_docs)
            for w, doc in zip(chunk, out_docs):
                upos = 'X'
                if doc.sentences and doc.sentences[0].words:
                    upos = doc.sentences[0].words[0].upos
                
                pos_cache[w.lower()] = upos
        except Exception as e:
            print(f"Failed batch {i}: {e}")
            for w in chunk:
                pos_cache[w.lower()] = 'X'
                
        if i % 10000 == 0 and i > 0:
            print(f"Tagged {i}/{len(words)} words...")
            
    # Save the json
    with open("german_pos_cache.json", "w", encoding="utf-8") as f:
        json.dump(pos_cache, f, ensure_ascii=False)
        
    print("Done! Saved to german_pos_cache.json")

if __name__ == "__main__":
    main()
