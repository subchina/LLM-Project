<h3 align="center">LLM Final Project</h3>

  <p align="center">
    This repository is a collection of files for the final project in the LLM course.
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project
Our project focuses on utilizing LLMs in the field of cybersecurity. The idea is to train several LLMs using a domain-specific dataset, namely the Common Weaknesses         Enumeration (CWE) list. Subsequently, we compare the models' performances against each other and analyze the results. 

## Description

### Prerequisites

The list of needed Python libraries: 
* openai
* pandas
* langchain
* asyncio
* argparse
* typing
* dotenv
* json
* pathlib
* os
* logging
  
To run the question-generating (preprocess/generate_questions.py) code using the GPT-4o-mini model, you need a personal OpenAI API key.

<!-- USAGE EXAMPLES -->
### Directory
* data
    * processed - contains intermediate versions of the CWE dataset during the pre and postprocessing steps.
    * raw - contains the raw CWE dataset.
* evaluation - contains the evaluation Python file and the final JSON file
* mistral-qlora-finetune
* notebooks - contains Jupyter notebooks of the finetuning process
* preprocess - contains Python files used to process the CWE dataset

## Authors
 Andrei Foitos (S5233836), Andjela Matic (S5248736), Davide Zani (S5054702)
