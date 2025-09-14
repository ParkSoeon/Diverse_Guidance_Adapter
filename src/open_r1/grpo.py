# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field
import wandb

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state
    # "set_peft_model_state" is for loading LoRA weights for each Adapter
from transformers.trainer_utils import get_last_checkpoint

# from open_r1.configs import GRPOConfig
from diversity_grpo_config import GRPOConfig 

from open_r1.rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
)
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
# from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from custom_diversity_grpo_trainer import CustomGuidanceGRPOTrainer # 

from sentence_transformers import SentenceTransformer
import torch.nn.functional as F


logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count", "interactive_bleu", "smi", "negative_bleu", "one_minus_bleu"], ##
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )

    ###
    # Multi-adapter configuration
    num_guidance_adapters: int = field(
        default=2,
        metadata={"help": "Number of guidance adapters for diversity"}
    )
    num_candidates_main: int = field(
        default=6,
        metadata={"help": "Number of candidates from main adapter (accuracy focused)"}
    )
    num_candidates_per_guidance: int = field(
        default=2,
        metadata={"help": "Number of candidates per guidance adapter (diversity focused)"}
    )
    # LoRA hyperparameters
    lora_rank: int = field(
        default=128,
        metadata={"help": "LoRA rank"},
    )
    _lora_alpha: int = field(
        default=64,
        metadata={"help": "LoRA alpha"},
    )
    _lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"},
    )
    ###

### 
def setup_multi_adapter_model(model, script_args):
    """ Setup Models with Multiple LoRA Adapters(For Diversity) """

    # Base LoRA Config
    base_lora_config = LoraConfig(
        r=script_args.lora_rank,
        lora_alpha=script_args._lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=script_args._lora_dropout,
    )

    # Apply Main Adapter (Default and Base)
    model = get_peft_model(model, base_lora_config)
    model = prepare_model_for_kbit_training(model)

    # Add Multi Guidance Adapters
    for i in range(script_args.num_guidance_adapters):
        adapter_name = f"diversity_guidance_adapter_{i}"

        # Newly create LoRA Config for multi Guidance Adapter
        guidance_lora_config = LoraConfig(
            r = script_args.lora_rank,
            lora_alpha = script_args._lora_alpha,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            task_type = "CAUSAL_LM",
            lora_dropout = script_args._lora_dropout,
        )

        # Add new adapter to the model
        model.add_adapter(adapter_name, config=guidance_lora_config)
        logger.info(f"Added adapter: {adapter_name}")

    model.print_trainable_parameters()

    # Set back to Default Adapter (Main)
    model.set_adapter("main")

    logger.info(f"Model setup with {script_args.num_guidance_adapters} guidance adapters.")
    logger.info(f"Model setup complete with {script_args.num_guidance_adapters + 1}")
    model.print_trainable_parameters()

    return model

def set_run_name(training_args, script_args, model_args):
    if training_args.run_name is None:
        base_name = model_args.model_name_or_path.split("/")[-1]
        training_args.run_name = (
            f"{base_name}-lr{script_args.learning_rate}"
            f"-mcl{script_args.max_completion_length}"
            f"-epoch{script_args.num_train_epochs}"
        )
    return training_args.run_name
###

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    wandb.init(project="acl2026", name=training_args.run_name)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    training_args.run_name = set_run_name(training_args, script_args, model_args)
    logger.info(f"Run name: {training_args.run_name}")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # Load tokenizer
    ################
    # tokenizer = get_tokenizer(model_args, training_args)

    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": code_reward,
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # Format into conversation
    def make_conversation(example):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    # for i in range(2):
    #     adapter_name = f"diversity_guidance_adapter_{i}"

    # model = get_peft_model(model, peft_config)
    # model = prepare_model_for_kbit_training(model)
    # model.print_trainable_parameters()

    """
        Main -> AdapterA -> Adapter B(Or simulateously) -> Main
        It means that firstly, in the main adapter, the model generates outputs that only consider the primary reward (e.g., accuracy). -> Generate outputs.(num=6)
        Then, the model switches to Adapter A, which is trained to optimize for diversity(e.g., Interactive BLEU) -> Generate outputs.(num=2)
        Adapter B is also used to optimize for diversity, but it does so in a different manner or with a different focus. -> Generate outputs.(num=2)
        Finally, the model switches back to the main adapter to generate the final outputs that balance both accuracy and diversity. -> Update? the main adapter will also train with the 10 losses from before.....
        Itteratively, the model switches between these adapters during training, allowing it to learn how to balance the trade-off between accuracy and diversity in its outputs.
    """
    # Setup Multi-Adapter Model
    model = setup_multi_adapter_model(model, script_args)
    # Get Reward Functions
    base_reward = get_base_reward_functions(script_args)
    diversity_reward = get_diversity_reward_functions()

    # Combine Reward Functions based on Configuration
    all_reward_funcs = []
    for func_name in script_args.reward_funcs:
        if func_name in base_reward:
            all_reward_funcs.append(base_reward[func_name])
        elif func_name in diversity_reward:
            all_reward_funcs.append(diversity_reward[func_name])
        else:
            assert False, f"Reward function {func_name} not recognized."

    # Sepup Training Arguments for Multi-Adapter
    training_args.guidance_adapter_names = [
        f"diversity_guidance_adapter_{i}" for i in range(script_args.num_guidance_adapters)
    ]
    training_args.num_guidance_adapters = script_args.num_guidance_adapters
    training_args.num_candidates_main = script_args.num_candidates_main
    training_args.num_candidates_per_guidance = script_args.num_candidates_per_guidance

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = CustomGuidanceGRPOTrainer(
        model=model,
        reward_funcs=all_reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        # callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    # logger.info("*** Train ***")
    logger.info("*** Starting Multi-Adapter GRPO Training ***")

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)  # Saves the tokenizer too for easy upload

    # Save each adapter separately!!!!
    for i in range(script_args.num_guidance_adapters):
        adapter_name = f"diversity_guidance_adapter_{i}"
        adapter_dir = os.path.join(training_args.output_dir, f"adapter_{adapter_name}")
        model.save_pretrained(adapter_dir, adapter_name=[adapter_name])
        logger.info(f"Saved adapter {adapter_name} to {adapter_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True 
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        # metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("*** Training Finished ***")

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
