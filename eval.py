import argparse
from typing import Any, Dict, List
from dotenv import load_dotenv
import asyncio
import logging
import os
import json
import re
from pathlib import Path
import time

load_dotenv()
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chat_models import AzureChatOpenAI
from langchain.llms.textgen import TextGen
from langchain.schema.language_model import BaseLanguageModel
from langchain.adapters.openai import convert_message_to_dict
from langchain.schema.messages import (
    AIMessage,
    SystemMessage,
    HumanMessage,
)
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache

# Set up LLM cache
set_llm_cache(
    SQLiteCache(
        database_path=str(Path(__file__).parent.parent / ".langchain.db")
    )
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler(f"./eval-{int(time.time())}.log", "w"))

# Instruction and few-shot examples
instruction = "Please select the correct letter(s) (A, B, C, D) for the following cybersecurity questions. Provide only the selected letter(s) without any additional text or explanation."

chat_few_shot = [
    HumanMessage(
        content="Question: Which mitigation prevents stack overflow bugs? A: Stack Canary. B: ALSR. C: CFI. D: Code Signing."
    ),
    AIMessage(content="Answer: A"),  # Answer with a single letter
]

few_shot = """
Question: What is the most effective way to prevent NULL pointer dereferences? A: Use a programming language that manages pointers. B: Always initialize pointers before use. C: Implement a custom memory management system. D: Use manual memory allocation.
Answer: B

Question: In a multithreaded application, what should you do to prevent race conditions? A: Use proper locking mechanisms. B: Rely on compiler optimizations. C: Avoid using multiple threads. D: Always use global variables.
Answer: A

Question: Which security measure is crucial when creating bank accounts without user identity verification? A: Implement strong password policies. B: Duplicate client-side authentication checks. C: Use a vetted library for authentication. D: Skip authentication for user convenience.
Answer: C 

Question: What are the best practices for securing sensitive data access? A: Require strong authentication. B: Use encryption for data at rest. C: Allow unrestricted access for better user experience. D: Implement logging for data access.
Answer: AB

Question: To securely handle untrusted data in Java, which practices should be implemented? A: Use signing features for data integrity. B: Avoid using serialization altogether. C: Implement input validation for deserialized data. D: Make fields transient to protect them from deserialization.
Answer: ACD
"""

# Initialize HuggingFace LLM
def init_hf_llm(model_id: str):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    except ImportError:
        raise ImportError("Please install transformers with `pip install transformers`")
    
    try:
        import torch
        flash_attn_enable = torch.cuda.get_device_capability()[0] >= 8
    except ImportError:
        raise ImportError("Please install torch with `pip install torch`")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'left'
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)

    llm = HuggingFacePipeline(pipeline=pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, max_new_tokens=5))

    return llm

# Initialize TextGen LLM
def init_textgen_llm(model_id: str):
    if os.environ.get("TEXTGEN_MODEL_URL") is None:
        raise RuntimeError("Please set TEXTGEN_MODEL_URL")
    return TextGen(model_url=os.environ["TEXTGEN_MODEL_URL"])

# Initialize Azure OpenAI LLM
def init_azure_openai_llm(model_id: str):
    if os.environ.get("OPENAI_API_ENDPOINT") is None:
        raise RuntimeError("Please set OPENAI_API_ENDPOINT")
    if os.environ.get("OPENAI_API_KEY") is None:
        raise RuntimeError("Please set OPENAI_API_KEY")
    azure_params = {
        "model": model_id,
        "openai_api_base": os.environ["OPENAI_API_ENDPOINT"],
        "openai_api_key": os.environ["OPENAI_API_KEY"],
        "openai_api_type": os.environ.get("OPENAI_API_TYPE", "azure"),
        "openai_api_version": "2023-07-01-preview",
    }
    return AzureChatOpenAI(**azure_params)

# Load dataset from file
def load_dataset(dataset_path: str):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    return dataset

# Perform batch inference
async def batch_inference_dataset(
    llm: BaseLanguageModel, batch: List[Dict[str, Any]], chat=False
):
    results = []
    llm_inputs = []
    
    for dataset_row in batch:
        question_text = f"Question: {dataset_row['question']} Choices: {' '.join(dataset_row['choices'])} Respond with only A, B, C, or D."
        
        if chat:
            llm_input = (
                [SystemMessage(content=instruction)]
                + chat_few_shot
                + [HumanMessage(content=question_text)]
            )
        else:
            llm_input = instruction + few_shot + question_text + "\n"

        llm_inputs.append(llm_input)

    try:
        llm_outputs = await llm.abatch(llm_inputs)
    except Exception as e:
        logging.error(f"Error in processing batch: {e}")
        llm_outputs = [f"{e}" for _ in llm_inputs]

    for idx, llm_output in enumerate(llm_outputs):
        if isinstance(llm_output, AIMessage):
            llm_output = llm_output.content.strip()  # Ensure to strip whitespace
        
        # Log the LLM output for debugging
        logger.info(f"LLM Output: {llm_output} for question: {batch[idx]['question']}")
        
        if llm_output == "ABCD":
            logger.warning(f"Default response 'ABCD' received for question: {batch[idx]['question']}")

        if "Answer:" in llm_output:
            llm_output = llm_output.replace("Answer:", "").strip()
        
        # Prepare the results
        batch[idx]["llm_input"] = convert_message_to_dict(llm_inputs[idx]) if chat else llm_inputs[idx]
        batch[idx]["llm_output"] = llm_output
        batch[idx]["llm_answer"] = "".join(sorted(list(set(re.findall(r"[A-D]", llm_output)))))
        batch[idx]["score"] = int(batch[idx]["llm_answer"].lower() == batch[idx]["answer"].lower())
        
        logging.info(
            f'llm_output: {llm_output}, parsed answer: {batch[idx]["llm_answer"]}, answer: {batch[idx]["answer"]}'
        )
        results.append(batch[idx])
    return results

# Process the dataset in batches
def inference_dataset(
    llm: BaseLanguageModel,
    dataset: List[Dict[str, Any]],
    batch_size: int = 1,
    chat: bool = False,
):
    # Prepare the batched inference
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    loop = asyncio.get_event_loop()
    batches = list(chunks(dataset, batch_size))
    results = []
    for idx, batch in enumerate(batches):
        logger.info(f"Processing batch {idx + 1}/{len(batches)}")
        results += loop.run_until_complete(batch_inference_dataset(llm, batch, chat))
    return results

# Count scores by topic
def count_score_by_topic(dataset: List[Dict[str, Any]]):
    score_by_topic = {}
    total_score_by_topic = {}
    score = 0
    for dataset_row in dataset:
        for topic in dataset_row["topics"]:
            if topic not in score_by_topic:
                score_by_topic[topic] = 0
                total_score_by_topic[topic] = 0
            score_by_topic[topic] += dataset_row["score"]
            total_score_by_topic[topic] += 1
        score += dataset_row["score"]
    score_fraction = {
        k: f"{v}/{total_score_by_topic[k]}" for k, v in score_by_topic.items()
    }
    score_float = {
        k: round(100 * float(v) / float(total_score_by_topic[k]), 4)
        for k, v in score_by_topic.items()
    }
    score_float["Overall"] = round(100 * float(score) / float(len(dataset)), 4)
    score_fraction["Overall"] = f"{score}/{len(dataset)}"
    return score_fraction, score_float

# Main function for CLI
def main():
    parser = argparse.ArgumentParser(description="SecEval Evaluation CLI")

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="/tmp",
        help="Output directory for results",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        choices=["hf", "azure_openai", "textgen"],
        default="hf",
        help="Type of model to use",
    )
    parser.add_argument(
        "-d", "--dataset_path", type=str, required=True, help="Path to the dataset JSON file"
    )

    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset_path)

    # Initialize the appropriate LLM based on model type
    if args.model_type == "hf":
        llm = init_hf_llm("your_huggingface_model_id")
    elif args.model_type == "azure_openai":
        llm = init_azure_openai_llm("your_azure_model_id")
    else:
        llm = init_textgen_llm("your_textgen_model_id")

    # Perform inference on the dataset
    results = inference_dataset(llm, dataset, batch_size=args.batch_size)

    # Count scores by topic
    score_fraction, score_float = count_score_by_topic(results)

    # Save the results to a file
    output_file = Path(args.output_dir) / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    # Print the score summary
    logger.info("Score Summary:")
    for topic, fraction in score_fraction.items():
        logger.info(f"{topic}: {fraction} ({score_float[topic]}%)")

if __name__ == "__main__":
    main()
