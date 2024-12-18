import datasets
from transformers import AutoTokenizer
import re
from typing import Optional, List
from tqdm.auto import tqdm

from reasoning_model import ReasoningModelForCausalLM

def get_reasoning_model_responses(
    input_text: str,
    model: ReasoningModelForCausalLM,
    tokenizer: AutoTokenizer,
    system_prompt: str = "You are a helpful and harmless assistant. You should think step-by-step.",
    iterations_per_step: int = 5,
    max_iterations: int = 15,
    mini_step_size: int = 32,
    expand_threshold: int = 0,
    step_separator_ids: Optional[List[int]] = None,
) -> List[str]:
    """Get model responses for a batch of input texts"""
    # 入力の用意
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # モデルによる推論
    final_tokens, final_node = model.generate(
        **model_inputs,
        iterations_per_step=iterations_per_step,    # 1推論ステップの探索に何回シミュレーションを行うか。長いほど精度が高まる可能性はあるが、推論時間が伸びる。
        max_iterations=max_iterations,              # 推論ステップの上限: 0.5Bモデルの場合、そこまで長いステップの推論はできないため10~15くらいが妥当。
        mini_step_size=mini_step_size,              # mini-step: 32tokens。Step as Action戦略を採用する場合、ここを512など大きな数字にする。（実行時間が伸びるため非推奨）
        expand_threshold=expand_threshold,          # ノードを拡張するために必要なそのノードの訪問回数の閾値。デフォルトの0で、拡張には1回以上の訪問が求められる。基本デフォルトで良い。
        step_separator_ids=step_separator_ids,      # Reasoning Action StrategyでStep as Actionを採用するときの区切りとなるトークンのIDリスト。NoneでモデルConfigの値を利用するため、変更は非推奨。Step as Action不採用時には[]を設定する。
    )

    # 結果をテキスト化
    final_text = tokenizer.decode(final_tokens, skip_special_tokens=True)

    return final_text

def extract_last_number(text: str) -> Optional[str]:
    """Extract the last number from text"""
    numbers = re.findall(r'\d+', text)
    return numbers[-1] if numbers else None

def process_in_batches(items: List, batch_size: int, desc: str = "Processing"):
    """Helper function to process items in batches with tqdm"""
    return [
        item
        for i in tqdm(range(0, len(items), batch_size), desc=desc)
        for item in items[i:i + batch_size]
    ]

def evaluate_model_on_mgsm(
    model_name: Optional[str] = None,
    model: Optional[ReasoningModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    repo_id: Optional[str] = None,
    iterations_per_step: int = 5,
    max_iterations: int = 15,
    mini_step_size: int = 32,
    expand_threshold: int = 0,
    step_separator_ids: Optional[List[int]] = None,
    system_prompt: str = "You are a helpful and harmless assistant. You should think step-by-step.",
) -> tuple[datasets.Dataset, dict]:
    """
    Evaluate a model on the MGSM-250 dataset.

    Args:
        model_name: Name of the model on HuggingFace Hub. Required if model and tokenizer are not provided.
        model: Pre-initialized ReasoningModelForCausalLM instance. Required if model_name is not provided.
        tokenizer: Pre-initialized AutoTokenizer instance. Required if model_name is not provided.
        repo_id: Repository ID to push results to HuggingFace Hub. If None or empty, results won't be pushed.
        iterations_per_step: Number of simulations per inference step. Higher values may improve accuracy but increase inference time.
        max_iterations: Maximum number of inference steps. For 0.5B models, 10-15 is recommended.
        mini_step_size: Size of mini-steps (32 tokens). Set larger (e.g., 512) for Step as Action strategy (not recommended).
        expand_threshold: Threshold for node expansion visits. Default 0 requires at least one visit.
        step_separator_ids: Token IDs for step separation in Step as Action strategy. None uses model config, [] disables.
        system_prompt: System prompt for the model.

    Returns:
        tuple of (Dataset with results, dict with metrics)
            - Dataset includes original data, model responses, predictions, and correctness
            - Metrics dict contains correct_count, total_count, and accuracy
    """
    # Load dataset
    print("Loading dataset...")
    dataset = datasets.load_dataset("HachiML/mgsm_250", split="test")

    # Initialize model and tokenizer if not provided
    if model is None or tokenizer is None:
        if model_name is None:
            raise ValueError("Either model_name or both model and tokenizer must be provided")
        
        print("Loading model and tokenizer...")
        model = ReasoningModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get model responses in batches
    print("Generating model responses...")
    inputs = dataset['input']
    responses = []

    for input in tqdm(inputs):
        batch_responses = get_reasoning_model_responses(
            input_text=input,
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            iterations_per_step=iterations_per_step,
            max_iterations=max_iterations,
            mini_step_size=mini_step_size,
            expand_threshold=expand_threshold,
            step_separator_ids=step_separator_ids,
        )
        responses.append(batch_responses)

    print("Extracting predictions...")
    predictions = [extract_last_number(response) for response in tqdm(responses, desc="Extracting numbers")]

    # Add predictions to dataset
    print("Adding predictions to dataset...")
    dataset = dataset.add_column("response", responses)
    dataset = dataset.add_column("pred", predictions)

    # Add correct column
    dataset = dataset.add_column(
        "correct",
        [str(pred) == str(output) for pred, output in zip(predictions, dataset['output'])]
    )

    # Calculate metrics
    correct_count = sum(dataset['correct'])
    total_count = len(dataset)
    accuracy = correct_count / total_count

    metrics = {
        "correct_count": correct_count,
        "total_count": total_count,
        "accuracy": float(accuracy)
    }

    print(f"Correct answers: {correct_count} out of {total_count}")
    print(f"Accuracy: {accuracy:.2%}")

    if repo_id is not None and repo_id.strip():
        print("Pushing results to HuggingFace Hub...")
        try: 
            dataset.push_to_hub(repo_id)
            print(f"Results pushed to: {repo_id}")
        except Exception as e:
            print(f"Failed to push results to HuggingFace Hub: {str(e)}")

    return dataset, metrics