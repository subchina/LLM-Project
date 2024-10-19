# "cleanup_data.py"
# This file handles processing of data after generating questions
# Author: Andjela Matic (S5248736)

import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cwe_filtered_and_questions.csv', encoding='latin-1', encoding_errors='ignore', on_bad_lines='warn')

# Remove commas from the 'DESCRIPTION' column
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(',', '')

# Drop duplicates that have the same CWE-ID and DESCRIPTION
df = df.drop_duplicates(subset=['CWE-ID', 'DESCRIPTION'], keep=False)

# 
cwe_filtered = pd.read_csv('cwe_filtered.csv', encoding='latin-1', encoding_errors='ignore', on_bad_lines='warn')

ordered_df = pd.merge(df[['CWE-ID', 'DESCRIPTION']], cwe_filtered, 
                      on=['CWE-ID', 'DESCRIPTION'], how='left')

df.to_csv('cwe_filtered_and_questions_matched.csv', index=False)


