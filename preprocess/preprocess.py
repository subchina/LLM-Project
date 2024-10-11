# "preprocess.py" 
# This file handles preprocessing of the CVE dataset
# Author: Andjela Matic (S5248736)

import pandas as pd

# Load the full CVE dataset
cve = pd.read_csv('cve.csv') 
# Filter to only contain columns with the vulnerability name and summary
cve_filtered = cve[['cwe_name', 'summary']] 
# Remove rows that contain a tag such as rejected or disputed
cve_filtered = cve_filtered[~cve_filtered['summary'].str.startswith('**')] 
# Remove duplicates if any
cve_filtered = cve_filtered.drop_duplicates() 
# Save to new csv file
cve_filtered.to_csv('cve_filtered.csv', index=False) 
