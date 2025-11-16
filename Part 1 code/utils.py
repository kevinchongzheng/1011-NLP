import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    text = example["text"]

    # Tokenize into words
    tokens = word_tokenize(text)
    new_tokens = []

    # Simple map of neighboring keys on QWERTY keyboard for typos
    neighbor_map = {
        "a": ["s"],
        "s": ["a", "d"],
        "d": ["s", "f"],
        "f": ["d", "g"],
        "g": ["f", "h"],
        "h": ["g", "j"],
        "j": ["h", "k"],
        "k": ["j", "l"],
        "l": ["k"],
        "e": ["w", "r"],
        "r": ["e", "t"],
        "t": ["r", "y"],
        "y": ["t", "u"],
        "u": ["y", "i"],
        "i": ["u", "o"],
        "o": ["i", "p"],
        "n": ["b", "m"],
        "m": ["n"]
    }

    for w in tokens:
        # Only consider alphabetic words for modification
        if w.isalpha() and random.random() < 0.2:
            # Decide whether to do synonym replacement or typo
            if random.random() < 0.5:
                # ---- Synonym replacement ----
                synsets = wordnet.synsets(w)
                lemmas = [
                    lemma.name().replace("_", " ")
                    for s in synsets
                    for lemma in s.lemmas()
                ]
                candidates = [l for l in lemmas if l.lower() != w.lower()]

                if candidates:
                    replacement = random.choice(candidates)
                    # Preserve capitalization
                    if w[0].isupper():
                        replacement = replacement.capitalize()
                    new_tokens.append(replacement)
                else:
                    new_tokens.append(w)
            else:
                # ---- Keyboard-neighbor typo ----
                chars = list(w)
                # indices whose letter has neighbors
                candidate_idxs = [
                    i for i, ch in enumerate(chars) if ch.lower() in neighbor_map
                ]
                if candidate_idxs:
                    idx = random.choice(candidate_idxs)
                    ch = chars[idx]
                    neighbors = neighbor_map[ch.lower()]
                    new_ch = random.choice(neighbors)
                    if ch.isupper():
                        new_ch = new_ch.upper()
                    chars[idx] = new_ch
                    new_tokens.append("".join(chars))
                else:
                    new_tokens.append(w)
        else:
            new_tokens.append(w)

    new_text = TreebankWordDetokenizer().detokenize(new_tokens)
    example["text"] = new_text

    ##### YOUR CODE ENDS HERE ######

    return example

