from datasets import load_dataset, concatenate_datasets
import pandas as pd

def get_topics():
    sum = load_dataset("ThatsGroes/dialog-topics")
    sum = sum["train"]["topic_da"]
    sum = list(set(sum))
    wiki = load_dataset("ThatsGroes/wiki_views")
    wiki = wiki["train"]["article"]
    sum.extend(wiki)
    return sum
