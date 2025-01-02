import deepl
from dotenv import load_dotenv
import os
from get_topics import get_topics
import pandas as pd
from datasets import load_dataset, Dataset

load_dotenv()

topics = load_dataset("knkarthick/dialogsum")
topics = topics["train"]["topic"]
topics = list(set(topics))

df_topics = pd.DataFrame({"topic_en" : topics})

token = os.getenv("DEEPL_TOKEN")
hf_token = os.getenv("HF_TOKEN")

#subset = df_topics[:5]

translator = deepl.Translator(token)

df_topics['topic_da'] = df_topics['topic_en'].apply(lambda x: translator.translate_text(x, target_lang="DA"))

df_topics.to_csv("topics_da.csv", index=False)

def unserialize(text):
    return text.text  # Extract the translated text as a plain string

df_topics['topic_da'] = df_topics['topic_da'].apply(unserialize)

hf_topics = Dataset.from_dict(df_topics)

hf_topics.push_to_hub("ThatsGroes/dialog-topics", token=hf_token)

#result = translator.translate_text("Hello, world!", target_lang="DA")

#subset['topic_da'] = subset['topic_en'].apply(lambda x: translator.translate_text(x, target_lang="DA"))


#total_characters = df_topics['topic_en'].str.len().sum()

#print(result.text)  # "Bonjour, le monde !"

#df_topics.drop(["topic_da_processed"], axis=1, inplace=True)