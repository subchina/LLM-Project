# "convert_data.py"
# This file handles processing of data after generating questions
# Author: Andjela Matic (S5248736)

import pandas as pd

df = pd.read_csv('cwe_filtered_and_questions_matched.csv', encoding='latin-1', encoding_errors='ignore', on_bad_lines='warn')

# Change the name of columns
df.columns = ["cwe-id", "completion", "prompt"]

# Drop the cwe-id column, as we only need the descriptionss and questions
df = df.drop(columns=["cwe-id"])

# Change order of columns
df = df[["prompt", "completion"]]

df.to_csv("json_ready.csv", index = False)