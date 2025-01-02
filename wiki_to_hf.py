from datasets import Dataset
import pandas as pd

df = pd.read_csv("wiki_views/all_wiki_views.csv")
dataset = Dataset.from_pandas(df)
dataset.push_to_hub("ThatsGroes/wiki_views")