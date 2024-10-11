# "generate_questions.py" 
# This file handles the use of the GPT-4o-mini model to generate prompts from CVE summaries
# Author: Andjela Matic (S5248736)

import pandas as pd
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key="") # Private API key is placed here

# Load the CSV file
df = pd.read_csv('cve_filtered.csv', encoding='latin-1', encoding_errors="ignore")

# Add a column for the questions
df['Generated Question'] = None

# Iterate over all the rows
for index, row in df.iterrows():

    # Store the summary of current row
    summary = row[df.columns[1]] 
    # Call the model to generate a question from the stored summary
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Create a question from the following summary: " + summary},
        ]
    )
    
    # Get the generated question
    question = completion.choices[0].message.content
    df.at[index, 'Generated Question'] = question
    
    # Save the current row to CSV iteratively
    df.iloc[[index]].to_csv('cve_filtered_and_questions.csv', mode='a', header=(index == 0), index=False)
    # Print statement to keep track of current index
    print("Question " + str(index + 1) + " added.")

print("Questions generated and saved iteratively.")
