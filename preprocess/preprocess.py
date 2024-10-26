# "preprocess.py" 
# Deals with cleaning up and filtering the CWE dataset
# Author: Andjela Matic (S5248736)

import pandas as pd

# Load the full CWE dataset
cwe = pd.read_csv('cwe.csv')

# Extract the year from the 'CVE-ID' (e.g., 'CVE-1999-0001' -> 1999)
cwe['YEAR'] = cwe['CVE-ID'].str.extract(r'CVE-(\d{4})')[0].astype(int)

# Filter rows for the years 2019-2021
cwe_filtered = cwe[cwe['YEAR'].between(2020, 2021)]

# Remove rows that contain a tag such as rejected or disputed
cwe_filtered = cwe_filtered[~cwe_filtered['DESCRIPTION'].str.startswith('**')]

# Filter for columns CWE-ID, and the description of vulnerability
cwe_filtered = cwe_filtered[['CWE-ID','DESCRIPTION']]

# Remove duplicates if any
cwe_filtered = cwe_filtered.drop_duplicates()

# Save to new csv file
cwe_filtered.to_csv('cwe_filtered.csv', index=False)
