import argparse
from typing import Any, Dict, List
from dotenv import load_dotenv
import asyncio
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
import json
import re
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)
import time
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache

set_llm_cache(
    SQLiteCache(
        database_path=str(Path(__file__).parent.parent / ".langchain.db")
    )
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger.addHandler(logging.FileHandler(f"./eval-{int(time.time())}.log", "w"))
instruction = (
    "Below are multiple-choice questions concerning cybersecurity. "
    "Please select the correct answer and respond with 'Answer: [x]', where [x] is A, B, C, D, or a combination of the letters. "
    "If the question allows for multiple correct answers, respond with all applicable letters in alphabetical order (e.g., AB, CD). "
    "For example, if the answer is A, simply respond with 'Answer: A'; if both A and B are correct, respond with 'Answer: AB'. "
)


chat_few_shot = [
]

few_shot = """
"""





def init_hf_llm(model_id: str):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    except ImportError:
        raise ImportError("Please install transformers with pip install transformers")
    
    try:
        import torch
        flash_attn_enable = torch.cuda.get_device_capability()[0] >= 8
    except ImportError:
        raise ImportError("Please install torch with pip install torch")

    # Load tokenizer and set padding side to 'left'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'left'  # Set padding side to left

    # If pad_token_id is not set, set it to eos_token_id (or another token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # You can assign another token ID if you prefer

    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    
    base_dir = os.path.abspath(model_id)  # Convert to absolute path
    adapter_path = os.path.join(base_dir, "adapter_config.json")
      # List contents of the base directory
    print("Contents of base directory:")
    #print(os.listdir(base_dir))

    # Check if adapter path exists and is accessible
    #print(f"Adapter exists: {os.path.isfile(adapter_path)}")
    #print(f"Adapter path is accessible: {os.access(adapter_path, os.R_OK)}")

    print(adapter_path)
    # Check if adapter_path is provided and exists, then apply PEFT adapter
    if adapter_path and os.path.isfile(adapter_path):
        try:
            from peft import PeftModel  # Importing only if needed
            model = PeftModel.from_pretrained(base_model, base_dir)
            print("Loaded model with PEFT adapter.")
            model.config.architectures = ["AutoModelForCausalLM"]
            print(model.config)
            print("Model State Dict Keys:", list(model.state_dict().keys()))

        except ImportError:
            raise ImportError("Please install peft with `pip install peft` to use adapters.")
        except Exception as e:
            print(f"Error loading adapter: {e}")
            model = base_model  
    else:
        model = base_model
        print("Loaded base model without adapter.")
    # Create HuggingFace pipeline with the updated tokenizer
    llm = HuggingFacePipeline(pipeline=pipeline("text-generation", 
                                                model=model, 
                                                tokenizer=tokenizer, 
                                                device=0, 
                                                max_new_tokens=5))

    return llm


def init_textgen_llm(model_id: str):
    if os.environ.get("TEXTGEN_MODEL_URL") is None:
        raise RuntimeError("Please set TEXTGEN_MODEL_URL")
    llm = TextGen(model_url=os.environ["TEXTGEN_MODEL_URL"])  # type: ignore
    return llm


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
    return AzureChatOpenAI(**azure_params)  # type: ignore


def load_dataset(dataset_path: str):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    return dataset


import json  # Assuming you save the output in JSON format

async def batch_inference_dataset(
    llm: BaseLanguageModel, batch: List[Dict[str, Any]], chat=False
):
    results = []
    llm_inputs = []
    for dataset_row in batch:
        question_text = (
            "Question: " + dataset_row["question"] + " ".join(dataset_row["choices"])
        )
        question_text = question_text.replace("\n", " ")
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
        logging.error(f"error in processing batch {e}")
        llm_outputs = [f"{e}" * len(llm_inputs)]

    for idx, llm_output in enumerate(llm_outputs):
        if type(llm_output) == AIMessage:
            llm_output: str = llm_output.content  # type: ignore
            
        logging.info(f"Raw LLM Output: {llm_output}")

        # New logic to ensure we extract the last "Answer: [x]"
        # Using regex to match and extract the final "Answer: [x]"
        answer_match = re.findall(r'Answer:\s*([A-D]+)', llm_output)

        if answer_match:
            # We take the last match to ensure we capture the correct one
            parsed_answer = answer_match[-1].strip()

        # Clean output by matching only valid letters (A-D)
        multi_letter_match = re.search(r'^[A-D]+$', parsed_answer)

        if multi_letter_match:
            parsed_answer = multi_letter_match.group(0).strip()

        if chat:
            batch[idx]["llm_input"] = convert_message_to_dict(llm_inputs[idx])
        else:
            batch[idx]["llm_input"] = llm_inputs[idx]

        batch[idx]["llm_output"] = llm_output  # Adding raw LLM output to the result
        batch[idx]["parsed_answer"] = parsed_answer  # Adding the parsed answer
        match = re.search(r"[A-D]+", parsed_answer)
        batch[idx]["llm_answer"] = match.group(0) if match else "Invalid"
        batch[idx]["score"] = int(
            set(batch[idx]["llm_answer"]) == set(batch[idx]["answer"])
        )  # Compare sets for multiple letters
        logging.info(
            f'llm_output: {llm_output}, parsed answer: {batch[idx]["llm_answer"]}, answer: {batch[idx]["answer"]}'
        )
        results.append(batch[idx])

    # Save results to a file
    with open('output_with_llm_output.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)  # Save with pretty JSON format

    return results







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

    # Asynchronously process dataset in batches
    loop = asyncio.get_event_loop()
    batches = list(chunks(dataset, batch_size))
    results = []
    for idx, batch in enumerate(batches):
        logger.info(f"processing batch {idx+1}/{len(batches)}")
        results += loop.run_until_complete(batch_inference_dataset(llm, batch, chat))
    return results


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


def main():
    parser = argparse.ArgumentParser(description="SecEval Evaluation CLI")

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="/tmp",
        help="Specify the output directory.",
    )
    parser.add_argument(
        "-d",
        "--dataset_file",
        type=str,
        required=True,
        help="Specify the dataset file to evaluate on.",
    )
    parser.add_argument(
        "-c",
        "--chat",
        action="store_true",
        default=False,
        help="Evaluate on chat model.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="Specify the batch size.",
    )
    parser.add_argument(
        "-B",
        "--backend",
        type=str,
        choices=["remote_hf", "azure", "textgen", "local_hf"],
        required=True,
        help="Specify the llm type. remote_hf: remote huggingface model backed, azure: azure openai model, textgen: textgen backend, local_hf: local huggingface model backed",
    )
    parser.add_argument(
        "-m",
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Specify the models.",
    )

    args = parser.parse_args()

    models = list(args.models)
    logging.info(f"evaluating models: {models}")
    for model_id in models:
        if args.backend == "remote_hf":
            llm = init_hf_llm(model_id)
        elif args.backend == "local_hf":
            model_dir = os.environ.get("LOCAL_HF_MODEL_DIR")
            if model_dir is None:
                raise RuntimeError(
                    "Please set LOCAL_HF_MODEL_DIR when using local_hf backend"
                )
            #model_id = os.path.join(model_dir, model_id)
            llm = init_hf_llm(model_id)
        elif args.backend == "textgen":
            llm = init_textgen_llm(model_id)
        elif args.backend == "azure":
            llm = init_azure_openai_llm(model_id)
        else:
            raise RuntimeError("Unknown backend")

        dataset = load_dataset(args.dataset_file)
        result = inference_dataset(
            llm, dataset, batch_size=args.batch_size, chat=args.chat
        )
        score_fraction, score_float = count_score_by_topic(result)
        result_with_score = {
            "score_fraction": score_fraction,
            "score_float": score_float,
            "detail": result,
        }
        output_path = (
            Path(args.output_dir)
            / f"{Path(args.dataset_file).stem}_{os.path.basename(model_id)}.json"
        )
        logger.info(f"writing result to {output_path}")
        with open(output_path, "w") as f:
            json.dump(result_with_score, f, indent=4)
        del llm


if __name__ == "__main__":
    main()
