import torch
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union, Sized
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from datasets import Dataset, IterableDataset
import warnings
import torch.nn.functional as F
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers.utils import is_peft_available
from torch import nn
from trl.import_utils import is_rich_available, is_vllm_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    get_high_entropy_mask,
    nanmin, nanmax
)
import wandb

if is_peft_available():
    from peft import PeftConfig, get_peft_model

RewardFunc = Union[str, PretrainedModel, Callable[]] ##

class CustomGuidanceGRPOTrainer(GRPOTrainer):  
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):

        # Initialize the parent GRPOTrainer class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

        self.diversity_adapters = getattr(self.args, "guidance_adapter_name", [])

        # Check if num of Candidates is consistent with num_generations and diversity guidance adapters
        total_candidates = (
            self.args.num_candidates_main + self.args.num_candidates_per_guidance * self.args.num_diversity_adapters
        )
        assert total_candidates == self.num_generations, (
            f"Total candidates ({total_candidates}) does not match num_generations ({self.num_generations}). "
            "Please check your configuration."
        )

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_loss:
            # Compute the loss using the liger grpo loss
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, inputs)
        else:
            return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        # Extract Metadata from Commons
        common_data = inputs["common"]
        guidance_data = inputs["guidance"]

        prompt_ids = common_data["prompt_ids"]
        prompt_mask = common_data["prompt_mask"]
        completion_ids = common_data["completion_ids"]
        completion_mask = common_data["completion_mask"]

        # Compute the per-token log probabilities for the model
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
        )

        common_loss_data = {
            "per_token_logps": per_token_logps,
            "entropies": entropies,
            "completion_mask": completion_mask,
            "old_per_token_logps": common_data.get("old_per_token_logps"),
            "ref_per_token_logps": common_data.get("ref_per_token_logps"),
        }

        total_loss = 0.0

        # Main Adapter Loss
        main_loss = self._compute_adapter_loss(per_token_logps, entropies, common_data, guidance_data["main"]["advantages"])
        total_loss += main_loss

        # Guidance Adapters Losses
        for adapter_name in self.diversity_adapters:
            adapter_advantages = guidance_data["guidance"][adapter_name]["advantages"]
            guidance_loss = self._compute_adapter_loss(model, per_token_logps, entropies, common_data, adapter_advantages)
            total_loss += guidance_loss
        
        return total_loss

    def _compute_adapter_loss(self, per_token_logps, entropies, common_data, advantages): # Not Sure
        completion_mask = common_data["completion_mask"]
        old_per_token_logps = common_data["old_per_token_logps"]
        ref_per_token_logps = common_data["ref_per_token_logps"]

        if self.top_entropy_quantile < 1.0:
            entropy_mask = get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        if self.beta != 0.0 and ref_per_token_logps is not None:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach() # Fallback if not provided

        log_ratio = per_token_logps - old_per_token_logps

        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )
        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        return loss

    def get_logits(self, model, input_ids):
        logits = model(input_ids).logits
        return logits

    def _get_per_token_logps(self, model, input_ids, logits_to_keep):
        """
        Calculate per-token log probabilities.
        """
        with torch.no_grad():
            logits = model(input_ids).logits

            # Extract Completion Logits (last_logits_to_keep tokens)
            completion_logits = logits[:, -logits_to_keep - 1 : -1, :] # Shifting for next-token prediction
            completion_targets = input_ids[:, -logits_to_keep:]

            log_probs = F.log_softmax(completion_logits, dim=-1)
            per_token_logps = torch.gather(
                log_probs, dim=-1, index=completion_targets.unsqueeze(-1)
            ).squeeze(-1)

        return per_token_logps

    def _prepare_inputs( # DO NOT CHANGE!!
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions( # Fixed
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        # Configuration for the Generation
        gen_length = self.args.max_completion_length
        temperature = float(self.args.temperature or 0.0)
        top_p = float(self.args.top_p or 0.9)
        top_k = int(self.args.top_k or 50)
        repeat_penalty = float(self.args.repeat_penalty or 1.0)
        num_main = int(self.args.num_candidates_main)
        num_per_guidance = int(self.args.num_candidates_per_guidance)

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        solution_inputs = self.processing_class(
            text = solutions,
            return_tensors="pt",
            padding="max_length", # But need to write new Dynamic Padding in future
            max_length=answer_length,
            padding_side="right",
            add_special_tokens=False,
        )
        solution_inputs = Trainer._prepare_inputs(self, solution_inputs)
        solution_ids, solution_mask = solution_inputs["input_ids"], solution_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size # always set same as per_device_train_batch_size
            all_generations = {"main": [], "guidance": {}}

            for adapter_name in self.diversity_adapters:
                all_generations["guidance"][adapter_name] = []

            # Process in batches
            for i in range(0, prompt_ids.size(0), generation_batch_size):
                end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                batch_prompt_ids = prompt_ids[i:end_idx].to(device)
                batch_prompt_mask = prompt_mask[i:end_idx]
                batch_size = batch_prompt_ids.size(0)

                # Comments below is not writtem by me. But don't delete it.
                # WARNING: Attention masks are not currently used during generation.
                # This works fine as we set num_generations == per_device_train_batch_size (no padding tokens created) in our config, but may cause
                # unintended attention to padding tokens when num_generations is smaller.
                # As currently we find Llada's modeling file does not handle attention mask. We will address this in future update soon.
                ### 

                # ==== 1. Main Adapter Generations(based on ROUGE) ====
                # Ensure that Main Adpater is enabled
                try: 
                    unwrapped_model.enable_adapter("main")
                except:
                    pass
                
                main_outputs = unwrapped_model.generate(
                    input_ids=batch_prompt_ids,
                    max_new_tokens=gen_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repeat_penalty,
                    num_return_sequences=num_main,
                    return_dict_in_generate=False, # -> Get Tensor
                    eos_token_id=self.processing_class.eos_token_id, # 
                    pad_token_id=self.processing_class.eos_token_id, #
                )
                # main_outputs shape: (batch_size * num_main, seq_len)
                all_generations["main"].append(main_outputs)

                # ==== 2. Diversity Guidance Generations (based on Diversity Guidance) ====
                for adapter_name in self.diversity_adapters:
                    try:
                        unwrapped_model.enable_adapter(adapter_name)
                    except:
                        pass
                
                    guidance_outputs = unwrapped_model.generate(
                        input_ids=batch_prompt_ids,
                        max_new_tokens=gen_length,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repeat_penalty,
                        num_return_sequences=num_per_guidance,
                        return_dict_in_generate=False, # -> Get Tensor
                        eos_token_id=self.processing_class.eos_token_id, #
                        pad_token_id=self.processing_class.eos_token_id, #
                    )
                    all_generations["guidance"][adapter_name].append(guidance_outputs)
                        
                    # # Ensure that Diversity Guidance Adapter is enabled
                    # # ==== 2.a Diversity Guidance 1 ====
                    # try:
                    #     unwrapped_model.enable_adapter("diversity_guidance_1")
                    # except:
                    #     pass

                    # guidance1_outputs = unwrapped_model.generate(
                    #     input_ids=batch_prompt_ids,
                    #     max_new_tokens=gen_length,
                    #     do_sample=True,
                    #     temperature=temperature,
                    #     top_p=top_p,
                    #     top_k=top_k,
                    #     repetition_penalty=repeat_penalty,
                    #     num_return_sequences=per_adapter,
                    #     return_dict_in_generate=False, # -> Get Tensor
                    #     eos_token_id=self.processing_class.eos_token_id, #
                    #     pad_token_id=self.processing_class.eos_token_id, #
                    # )
                    # prompt_completion_ids_all.append(guidance1_outputs)

                    # # ==== 2.b Diversity Guidance 2 ====
                    # try:
                    #     unwrapped_model.enable_adapter("diversity_guidance_2")
                    # except:
                    #     pass

                    # guidance2_outputs = unwrapped_model.generate(
                    #     input_ids=batch_prompt_ids,
                    #     max_new_tokens=gen_length,
                    #     do_sample=True,
                    #     temperature=temperature,
                    #     top_p=top_p,
                    #     top_k=top_k,
                    #     repetition_penalty=repeat_penalty,
                    #     num_return_sequences=per_adapter,
                    #     return_dict_in_generate=False, # -> Get Tensor
                    #     eos_token_id=self.processing_class.eos_token_id, #
                    #     pad_token_id=self.processing_class.eos_token_id, #
                    # )
                    # prompt_completion_ids_all.append(guidance2_outputs)

                # Restore the main adapter after generation(SWITCH back to main adapter)
                try:
                    unwrapped_model.set_adapter("main")
                except:
                    pass
            
                # Free GPU cache per batch(Optional...)
                del batch_prompt_ids
                torch.cuda.empty_cache()

        # Back half; with Answers
        for i in range(num_main, main_outputs.size(0), generation_batch_size):
            end_idx = min(i + generation_batch_size, main_outputs.size(0))
            batch_prompt_ids = prompt_ids[i:end_idx]
            batch_solution_ids = solution_ids[i:end_idx]
            batch_prompt_mask = prompt_mask[i:end_idx]
            batch_solution_mask = solution_mask[i:end_idx]

            batch_completion_ids = self.generate(
                model = unwrapped_model,
                prompt_ids = batch_prompt_ids,
                gen_length = gen_length,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                repeat_penalty = repeat_penalty,
                num_return_sequences = num_main,
                answer = batch_solution_ids,
            )
            prompt_completion_ids_all.append(batch_completion_ids)

            del batch_prompt_ids, batch_solution_ids
            torch.cuda.empty_cache()

        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        all_old_per_token_logps = None
        all_ref_per_token_logps = None
        
        with torch.no_grad():
            if self.num_iterations > 1:
                # repeat prompt completion ids self.num_iterations times
                prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0).expand(
                    self.num_iterations, -1, -1
                )
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids_expanded, logits_to_keep
                )
                all_old_per_token_logps = old_per_token_logps
            else:
                old_per_token_logps = None

            if float(self.beta) == 0.0:
                ref_per_token_logps = None
            else:
                # Compute ref per-token log probabilities(logps) with adapters disabled
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids_expanded, logits_to_keep
                    )
                    all_ref_per_token_logps = ref_per_token_logps

        # Decode Completions to text for Computing Rewards(reward functions)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # ==== 3. Compute Rewards ====
        # Compute rewards for each completion with each reward function
        total_generations = len(prompts) * (num_main + num_per_guidance * len(self.diversity_adapters))
        rewards_per_func = torch.zeros(total_generations, len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            # Module instead of PretrainedModel for compat with compiled models
            if isinstance(reward_func, nn.Module): 
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):

                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                # Convert None values to NaN
                output_reward_func = [
                    reward if reward is not None else torch.nan for reward in output_reward_func
                ]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather Across Processes
        rewards_per_func = gather(rewards_per_func)
        # Weighted Sum across reward functions (Ensure that reward_weights is on device)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        rewards_from_half = rewards[: num_main]
        rewards_back_half = rewards[num_main :]

        # Split Rewards per type of Generation Adapter
        adapter_rewards = {"main": None, "guidance": {}}
        start_idx = 0

        # Main Adapter Rewards
        end_idx = start_idx + len(prompts) * num_main
        adapter_rewards["main"] = rewards[start_idx:end_idx]
        start_idx = end_idx

        # Guidance Adapters Rewards
        for adapter_name in self.diversity_adapters:
            end_idx = start_idx + len(prompts) * num_per_guidance
            adapter_rewards["guidance"][adapter_name] = rewards[start_idx:end_idx]
            start_idx = end_idx
        
        adapter_advantages = {"main": None, "guidance": {}}

        # Compute Advantages per Adapter
        # Main
        main_rewards_grouped = adapter_rewards["main"].view(-1, num_main)
        main_mean_rewards = main_rewards_grouped.mean(dim=1)
        adapter_advantages["main"] = (
            adapter_rewards["main"] - main_mean_rewards.repeat_interleave(num_main, dim=0)
        )
        # Guidance
        for adapter_name in self.diversity_adapters:
            guidance_rewards_grouped = adapter_rewards["guidance"][adapter_name].view(-1, num_per_guidance)
            guidance_mean_rewards = guidance_rewards_grouped.mean(dim=1)
            adapter_advantages["guidance"][adapter_name] = (
                adapter_rewards["guidance"][adapter_name] - guidance_mean_rewards.repeat_interleave(num_per_guidance, dim=0)
            )

        # Overall Mean and Std (for logging only)
        total_generations_per_prompt = num_main + (num_per_guidance * len(self.diversity_adapters))
        all_grouped_rewards = rewards.view(-1, total_generations_per_prompt)
        std_grouped_rewards = all_grouped_rewards.std(dim=1)

        # Handle Distributed slice for Advantages as Original
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        # Metrics Logging        
        rewards_front_half = rewards[: len(rewards) // 2]
        rewards_back_half = rewards[len(rewards) // 2 :]

        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        zero_std_count = (std_grouped_rewards < 1e-6).sum().item()
        total_prompts = std_grouped_rewards.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        # per-reward Log
        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  
                # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_front_half"].append(rewards_front_half.mean().item())
        self._metrics[mode]["reward_back_half"].append(rewards_back_half.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["reward_front_half_std"].append(rewards_front_half.std().item())
        self._metrics[mode]["reward_back_half_std"].append(rewards_back_half.std().item())
        self._metrics[mode]["total_generations_per_prompt"].append(total_generations_per_prompt)
        self._metrics[mode]["main_num_candidates"].append(num_main)
        self._metrics[mode]["per_guidance_num_candidates"].append(num_per_guidance)

        # Log Completions if needed
        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)

            # Gather advantages and rewards
            main_advantages = adapter_advantages["main"]
            advantages_to_log = gather_object(advantages.tolist())
            rewards_to_log = rewards.tolist()
            
            # Split into front and back halves
            prompts_front_half = prompts_to_log[: len(prompts_to_log) // 2]
            prompts_back_half = prompts_to_log[len(prompts_to_log) // 2 :]
            completions_front_half = completions_to_log[: len(completions_to_log) // 2]
            completions_back_half = completions_to_log[len(completions_to_log) // 2 :]
            rewards_front_half = rewards_to_log[: len(rewards) // 2]
            rewards_back_half = rewards_to_log[len(rewards) // 2 :]
            advantages_front_half = advantages_to_log[: len(rewards) // 2]
            advantages_back_half = advantages_to_log[len(rewards) // 2 :]
            
            result = {
                "steps": self.state.global_step,
                "prompts_front_half": prompts_front_half,
                "completions_front_half": completions_front_half,
                "rewards_front_half": rewards_front_half,
                "advantages_front_half": advantages_front_half,
                "prompts_back_half": prompts_back_half,
                "completions_back_half": completions_back_half,
                "rewards_back_half": rewards_back_half,
                "advantages_back_half": advantages_back_half,
            }
            
            print(f"Results at step {self.state.global_step}:\n")
            print(f"Prompt:\n{Text(prompts_text[0])}\n")
            
            print_prompt_completions_sample(
                prompts_front_half,
                completions_front_half,
                rewards_front_half,
                advantages_front_half,
                step=self.state.global_step,
                is_front_half=True
                )
            
            print_prompt_completions_sample(
                prompts_back_half,
                completions_back_half,
                rewards_back_half,
                advantages_back_half,
                step=self.state.global_step,
                is_front_half=False
            )
            
            append_jsonl(f"{self.args.output_dir}/results.jsonl", result)                

        return {
            "common": {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "old_per_token_logps": all_old_per_token_logps,
                "ref_per_token_logps": all_ref_per_token_logps,
            },
            "guidance": {
                "main": {"advantages": adapter_advantages["main"]},
                "guidance": {
                    adapter_name: {"advantages": adapter_advantages["guidance"][adapter_name],}
                    for adapter_name in self.diversity_adapters
                },
            }
        }

if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

def print_prompt_completions_sample(prompts: list[str], completions: list[str], rewards: list[int], 
                                    advantages: list[float],
                                    step: int, is_front_half: bool) -> None:
    
    if not is_rich_available():
        raise ImportError("This feature requires `rich` to be installed. Please install it first: `pip install rich`")

    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)
    
    # Add columns
    # table.add_column("Prompt", style="bright_blue")
    table.add_column("Completion", style="bright_green")
    table.add_column("Reward", style="bold cyan", justify="right")
    table.add_column("Advatages", style="bold magenta", justify="right")

    for prompt, completion, reward, advantage in zip(prompts, completions, rewards, advantages):
        table.add_row(Text(completion), f"{reward:.2f}", f"{advantage:.2f}")  # Formatting reward to 2 decimal places
        table.add_section()  # Adds a separator between rows
        
    title = "Step {} - Front Half".format(step) if is_front_half else "Step {} - Back Half".format(step)
    panel = Panel(table, expand=False, title=title, border_style="bold white") 
    console.print(panel)
    
import os, json
from pathlib import Path

def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = _to_jsonable(obj)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        
def _to_jsonable(x):
    import torch, numpy as np
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x
