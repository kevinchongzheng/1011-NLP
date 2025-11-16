import os, argparse, random
from tqdm import tqdm

import torch
from transformers import GemmaTokenizerFast, GemmaForCausalLM
from transformers import GemmaTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from utils import set_random_seeds, compute_metrics, save_queries_and_records, compute_records
from prompting_utils import read_schema, extract_sql_query, save_logs
from load_data import load_prompting_data

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Maximum new tokens to generate
MAX_NEW_TOKENS = 128


def get_args():
    '''
    Arguments for prompting. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(
        description='Text-to-SQL experiments with prompting.')

    parser.add_argument('-s', '--shot', type=int, default=0,
                        help='Number of examples for k-shot learning (0 for zero-shot)')
    parser.add_argument('-p', '--ptype', type=int, default=0,
                        help='Prompt type')
    parser.add_argument('-m', '--model', type=str, default='gemma',
                        help='Model to use for prompting: gemma (gemma-1.1-2b-it) or codegemma (codegemma-7b-it)')
    parser.add_argument('-q', '--quantization', action='store_true',
                        help='Use a quantized version of the model (e.g. 4bits)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed to help reproducibility')
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")
    args = parser.parse_args()
    return args


def create_prompt(sentence, k, schema_text="", examples=None):
    '''
    Function for creating a prompt for zero or few-shot prompting.
    
    Args:
        sentence (str): Natural language query
        k (int): Number of examples in k-shot prompting
        schema_text (str): Database schema information (optional)
        examples (list): List of (nl, sql) tuples for few-shot examples
    
    Returns:
        str: Formatted prompt
    '''
    # System message for the model
    system_msg = "You are a SQL expert. Convert the natural language query to a SQL query."
    
    # Add schema if provided
    if schema_text:
        system_msg += f"\n\nDatabase Schema:\n{schema_text}"
    
    prompt_parts = [system_msg]
    
    # Add few-shot examples if k > 0
    if k > 0 and examples:
        prompt_parts.append("\n\nHere are some examples:")
        for nl, sql in examples[:k]:
            prompt_parts.append(f"\nNatural Language: {nl}")
            prompt_parts.append(f"SQL: {sql}")
    
    # Add the actual query
    prompt_parts.append(f"\n\nNatural Language: {sentence}")
    prompt_parts.append("SQL:")
    
    return "\n".join(prompt_parts)


def exp_kshot(tokenizer, model, inputs, k, train_x=None, train_y=None, schema_text=""):
    '''
    k-shot prompting experiments using the provided model and tokenizer. 
    This function generates SQL queries from text prompts and evaluates their accuracy.
    
    Args:
        tokenizer: Model tokenizer
        model: Language model
        inputs (List[str]): A list of text strings (NL queries)
        k (int): Number of examples in k-shot prompting
        train_x (List[str]): Training NL queries for few-shot examples
        train_y (List[str]): Training SQL queries for few-shot examples
        schema_text (str): Database schema
    
    Returns:
        raw_outputs: List of raw model outputs
        extracted_queries: List of extracted SQL queries
    '''
    raw_outputs = []
    extracted_queries = []

    # Prepare few-shot examples if k > 0
    examples = None
    if k > 0 and train_x and train_y:
        # Randomly sample k examples from training data
        indices = random.sample(range(len(train_x)), min(k, len(train_x)))
        examples = [(train_x[i], train_y[i]) for i in indices]

    for i, sentence in enumerate(tqdm(inputs, desc=f"{k}-shot prompting")):
        prompt = create_prompt(sentence, k, schema_text, examples)

        input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**input_ids, max_new_tokens=MAX_NEW_TOKENS)
        response = tokenizer.decode(outputs[0])
        raw_outputs.append(response)

        # Extract the SQL query
        extracted_query = extract_sql_query(response)
        extracted_queries.append(extracted_query)
        
    return raw_outputs, extracted_queries


def eval_outputs(extracted_queries, eval_y, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    Evaluate the outputs of the model by computing the metrics.
    
    Args:
        extracted_queries: List of SQL queries extracted from model outputs
        eval_y: Ground truth SQL (can be None for test set)
        gt_sql_pth: Path to ground truth SQL file
        model_sql_path: Path to save model SQL
        gt_record_path: Path to ground truth records
        model_record_path: Path to save model records
    
    Returns:
        sql_em, record_em, record_f1, model_error_msgs, error_rate
    '''
    # Save model queries and compute records
    save_queries_and_records(extracted_queries, model_sql_path, model_record_path)
    
    # Compute metrics (only if we have ground truth)
    if eval_y is not None:
        sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
            gt_sql_pth, model_sql_path, gt_record_path, model_record_path
        )
        
        # Calculate error rate
        num_errors = sum(1 for msg in model_error_msgs if msg is not None and msg != "")
        error_rate = num_errors / max(len(model_error_msgs), 1)
        
        return sql_em, record_em, record_f1, model_error_msgs, error_rate
    else:
        # For test set without ground truth
        return None, None, None, None, None


def initialize_model_and_tokenizer(model_name, to_quantize=False):
    '''
    Args:
        * model_name (str): Model name ("gemma" or "codegemma").
        * to_quantize (bool): Use a quantized version of the model (e.g. 4bits)
    
    To access to the model on HuggingFace, you need to log in and review the 
    conditions and access the model's content.
    '''
    if model_name == "gemma":
        model_id = "google/gemma-1.1-2b-it"
        tokenizer = GemmaTokenizerFast.from_pretrained(model_id)
        # Native weights exported in bfloat16 precision
        model = GemmaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, 
        ).to(DEVICE)
    elif model_name == "codegemma":
        model_id = "google/codegemma-7b-it"
        tokenizer = GemmaTokenizer.from_pretrained(model_id)
        if to_quantize:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4", # 4-bit quantization
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                quantization_config=nf4_config
            ).to(DEVICE)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16
            ).to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return tokenizer, model


def main():
    '''
    Main function for prompting experiments.
    '''
    args = get_args()
    shot = args.shot
    ptype = args.ptype
    model_name = args.model
    to_quantize = args.quantization
    experiment_name = args.experiment_name

    set_random_seeds(args.seed)

    data_folder = 'data'
    train_x, train_y, dev_x, dev_y, test_x = load_prompting_data(data_folder)

    # Load schema (optional)
    schema_text = ""
    schema_path = os.path.join(data_folder, "flight_database.schema")
    if os.path.exists(schema_path):
        schema_text = read_schema(schema_path)
        print(f"Loaded schema from {schema_path}")

    # Model and tokenizer
    print(f"Loading {model_name} model...")
    tokenizer, model = initialize_model_and_tokenizer(model_name, to_quantize)
    print(f"Model loaded on {DEVICE}")

    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    for eval_split in ["dev", "test"]:
        print(f"\n{'='*70}")
        print(f"Evaluating on {eval_split} set with {shot}-shot prompting")
        print(f"{'='*70}")
        
        eval_x, eval_y = (dev_x, dev_y) if eval_split == "dev" else (test_x, None)

        # Run k-shot prompting
        raw_outputs, extracted_queries = exp_kshot(
            tokenizer, model, eval_x, shot, train_x, train_y, schema_text
        )

        # Set up paths
        gt_sql_path = os.path.join(f'data/{eval_split}.sql')
        gt_record_path = os.path.join(f'records/ground_truth_{eval_split}.pkl')
        model_sql_path = os.path.join(f'results/{model_name}_{experiment_name}_{eval_split}.sql')
        model_record_path = os.path.join(f'records/{model_name}_{experiment_name}_{eval_split}.pkl')
        
        # Evaluate
        sql_em, record_em, record_f1, model_error_msgs, error_rate = eval_outputs(
            extracted_queries, eval_y,
            gt_sql_path, model_sql_path,
            gt_record_path, model_record_path
        )
        
        if eval_y is not None:  # Only print metrics if we have ground truth
            print(f"\n{eval_split} set results:")
            print(f"  Record F1: {record_f1:.4f}")
            print(f"  Record EM: {record_em:.4f}")
            print(f"  SQL EM: {sql_em:.4f}")
            print(f"  Error rate: {error_rate*100:.2f}%")
            print(f"✓ Saved results to {model_sql_path}")

            # Save logs
            log_path = f"logs/{model_name}_{experiment_name}_{eval_split}.txt"
            save_logs(log_path, sql_em, record_em, record_f1, model_error_msgs)
            print(f"✓ Saved logs to {log_path}")
        else:
            print(f"✓ Saved test predictions to {model_sql_path}")

    print(f"\n{'='*70}")
    print("All done!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
