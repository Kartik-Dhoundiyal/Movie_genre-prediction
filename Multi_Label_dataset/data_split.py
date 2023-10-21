import pandas as pd
import random

# Load the dataset from the provided CSV file
df = pd.read_csv('C:/Users/Karti/Desktop/drive/Multi_Label_dataset/whole.csv')

# Shuffle the dataset randomly
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Define the number of entries for each split
train_size = 5000
val_size = 1000

# Split the dataset
train_df = df[:train_size]
val_df = df[train_size:(train_size + val_size)]
test_df = df[(train_size + val_size):]

# Save the splits to separate CSV files
train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)
test_df.to_csv('test.csv', index=False)
