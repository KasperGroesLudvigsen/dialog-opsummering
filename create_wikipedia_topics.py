import pandas as pd
import os

# Path to the folder containing CSV files
folder_path = "wiki_views"

# Get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Read all CSV files and concatenate them into a single DataFrame
dataframes = [pd.read_csv(os.path.join(folder_path, file)) for file in csv_files]
combined_df = pd.concat(dataframes, ignore_index=True)

# Remove duplicates based on the "article" column
deduplicated_df = combined_df.drop_duplicates(subset="article")

# View the resulting DataFrame
print(deduplicated_df.head())

# Optionally, save the deduplicated DataFrame to a new CSV file
#deduplicated_df.to_csv("deduplicated_wiki_views.csv", index=False)

# Remove rows where "article" contains a semicolon (;) or an underscore (_)
filtered_df = deduplicated_df[~deduplicated_df["article"].str.contains(r"[;:]", na=False)]

filtered_df["article"] = filtered_df["article"].str.replace("_", " ")

filtered_df = filtered_df[filtered_df["article"] != "Main Page"]
filtered_df = filtered_df[filtered_df["article"] != "Forside"]
filtered_df = filtered_df[filtered_df["article"] != "Speciel:Søgning"]
filtered_df = filtered_df[~filtered_df['article'].str.contains(r'^Speciel:.*', na=False)]


#Speciel:Søgning


filtered_df = filtered_df.drop_duplicates(subset="article")


filtered_df.to_csv("wiki_views/all_wiki_views.csv", index=False)

# View the resulting DataFrame
print(filtered_df.head())

len(filtered_df)
