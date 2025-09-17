import numpy as np
import torch
from collections import Counter
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Dict, Any, Optional

# Initializing Smoothing for BLEU
smoothing = SmoothingFunction().method1

# Cache for SentenceTransformer model
_cached_model = None

def get_sentence_transformer():
    global _cached_model
    if _cached_model is None:
        _cached_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _cached_model

def extract_content(response: Any) -> str:
    """ Extract Content from Various Completion Formats """
    if isinstance(response, dict) and "content" in response:
        return response["content"]
    elif isinstance(response, list) and len(response) > 0:
        if isinstance(response[0], dict) and "content" in response[0]:
            return response[0]["content"]
    return str(response)

def extract_xml_answer(text: str) -> str:
    """ Extract answer from XML-like tags """
    if "<answer>" in text and "</answer>" in text:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    return text.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def tokenize_text(text: str) -> list[str]:
    try:
        return nltk.word_tokenize(text.lower())
    except:
        return text.lower().split()

# Compute BLEU score
def compute_bleu(reference: str, hypothesis: str, weights=(0.25, 0.25, 0.25, 0.25)) -> float:
    try:
        ref_tokens = tokenize_text(reference)
        hyp_tokens = tokenize_text(hypothesis)

        if len(hyp_tokens) == 0:
            return 0.0

        bleu = sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            weights = weights,
            smoothing_function=smoothing,
        )
        return bleu
    except:
        return 0.0


# ======= Main Diversity Reward Functions =======
# MUST BE IN Functions(BLEU Regarding things; negative(-) bleu, bleu in denominator, 1-BLEU, SMI, etc)
# a) Negative BLEU Reward
def interactive_negative_bleu_reward_func(prompts: str, completions: str, references: str=None, **kwargs) -> list[float]:
    """ 
        Calculate negative BLEU score between completions and references.
        It means that higher BLEU score results in lower reward.
    """
    if isinstance(completions[0], list):
        responses = [completion[0]["content"] for completion in completions]
    else:
        response = completions

    if references is None: # Just for.... Exception Handling
        # Use the first response as reference for all, if references are not provided
        references = [responses[0] * len(responses)]

    rewards = []
    for response, reference in zip(responses, references):
        try:
            cleaned_response = extract_xml_answer(response)
            cleaned_reference = extract_hash_answer(reference)
        except:
            cleaned_response = response
            cleaned_reference = reference

        bleu_score = compute_bleu(cleaned_reference, cleaned_response)
        rewards.append(-bleu_score)

    return rewards
    
# b) BLEU in Denominator Reward
def interactive_bleu_in_denominator_reward_func(prompts: str, completions: str, references: str=None, **kwargs) -> list[float]:
    """
        Calculate reward as 1 / (1 + BLEU score between completions and references).
        It means that higher BLEU score results in lower reward.
    """
    epsilon = 1e-6  # To avoid division by zero !! IMPORTANT

    if isinstance(completions[0], list):
        responses = [completion[0]["content"] for completion in completions]
    else:
        response = completions

    if references is None: # Just for.... Exception Handling
        # Use the first response as reference for all, if references are not provided
        references = [responses[0] * len(responses)]

    rewards = []
    for response, reference in zip(responses, references):
        try:
            cleaned_response = extract_xml_answer(response)
            cleaned_reference = extract_hash_answer(reference)
        except:
            cleaned_response = response
            cleaned_reference = reference

        bleu_score = compute_bleu(cleaned_reference, cleaned_response)
        reward = 1 / (1 + bleu_score + epsilon)  # Adding epsilon to avoid division by zero
        rewards.append(reward)

    return rewards

# c) 1 - BLEU Reward    
def interactive_one_minus_bleu_reward_func(prompts: str, completions: str, references: str=None, **kwargs) -> list[float]:
    """
        Calculate reward as 1 - BLEU score between completions and references.
        It means that higher BLEU score results in lower reward.
    """
    if isinstance(completions[0], list):
        responses = [completion[0]["content"] for completion in completions]
    else:
        response = completions

    if references is None: 
        # Use the first response as reference for all, if references are not provided
        references = [responses[0] * len(responses)]

    rewards = []
    for response, reference in zip(responses, references):
        try:
            cleaned_response = extract_xml_answer(response)
            cleaned_reference = extract_hash_answer(reference)
        except:
            cleaned_response = response
            cleaned_reference = reference

        bleu_score = compute_bleu(cleaned_reference, cleaned_response)
        reward = 1.0 - bleu_score
        rewards.append(reward)

    return rewards

# d) SMI Reward
def pre_smi_reward_func(prompts: str, completions: str, **kwargs) -> list[float]:
    """
        Calculate diversity gain based on Self-BLEU.
    """
    if isinstance(completions[0], list):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions
    
    if len(responses) < 2:
        return [0.0] * len(responses)
    
    # Extract clean responses
    clean_responses = []
    for response in responses:
        try:
            clean_responses.append(extract_xml_answer(response))
        except:
            clean_responses.append(response)
    
    rewards = []
    for i, target_response in enumerate(clean_responses):
        # Calculate Self-BLEU with all other responses as references
        other_responses = clean_responses[:i] + clean_responses[i+1:]
        
        bleu_scores = []
        for other_response in other_responses:
            bleu = compute_bleu(other_response, target_response)
            bleu_scores.append(bleu)
        
        # AVG Self-BLEU
        avg_self_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        
        # SMI: 1 - average_self_bleu (Lower self-bleu = Higher diversity)
        smi = 1.0 - avg_self_bleu
        rewards.append(smi)
    
    return rewards

def smi_reward_func(prompts: str, completions: str, references: str=None, **kwargs) -> list[float]:
    """
        Calculate reward based on Self-BLEU Mutual Information (SMI).
        SMI = BLEU(input, GT) - BLEU(input, pred)
        Reward = SMI
    """
    if isinstance(completions[0], list):
        responses = [completion[0]["content"] for completion in completions]
    else:
        response = completions

    scalar_rewards = []
    for i, response in enumerate(responses):
        try:
            cleaned_response = extract_xml_answer(response)
            reference = str(references[i]) if references else ""

            scalar_reward = 2.0 if extracted.strip() == reference.strip() else 0.0
            scalar_rewards.append(scalar_reward)
        except:
            scalar_rewards.append(0.0)
            
    diversity_gains = pre_smi_reward_func(prompts, completions, **kwargs)

    # Combined reward
    for scalar, diversity in zip(scalar_rewards, diversity_gains):
        combined_reward = alpha * scalar + beta * diversity
        rewards.append(combined_reward)

    return rewards

# e) BLEU Difference Reward
def interactive_bleu_difference_reward_func(prompt: str, completions: str, reference: str=None, **kwargs) -> list[float]:
    """
        Calculate reward as the difference between BLEU(input, GT) and BLEU(input, pred).
        Reward = BLEU(input, GT) - BLEU(input, pred)
        Higher reward for responses that are more similar to the reference than to the input prompt.
    """
    if isinstance(completions[0], list):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    if isinstance(prompts[0], list):
        inputs = [prompt[0]["content"] for prompt in prompts]
    else:
        inputs = prompts

    rewards = []
    for i, (input_text, response, reference) in enumerate(zip(inputs, responses, reference)):
        try:
            cleaned_response = extract_xml_answer(response)
            cleaned_reference = extract_hash_answer(reference)
        except:
            cleaned_response = response
            cleaned_reference = reference

        # BLEU(input, GT)
        bleu_input_ref = compute_bleu(cleaned_reference, input_text)
        # BLEU(input, pred)
        bleu_input_resp = compute_bleu(cleaned_response, input_text)

        reward = bleu_input_ref - bleu_input_resp
        rewards.append(reward)

    return rewards

# f) Embedding Similarity Reward
def smi_semantic_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
        Calculate diversity gain based on semantic similarity using SentenceTransformer embeddings.
        Uses a submodular function to compute marginal contributions.
        Reward = Marginal contribution to semantic diversity.
    """
    if isinstance(completions[0], list):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions
    
    if len(responses) < 2:
        return [0.0] * len(responses)
    
    try:
        # SentenceTransformer
        if not hasattr(smi_semantic_reward_func, 'model'): ## Cache the model to avoid reloading
            smi_semantic_reward_func.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        model = smi_semantic_reward_func.model
        
        # Clean responses
        clean_responses = []
        for response in responses:
            try:
                clean_responses.append(extract_xml_answer(response))
            except:
                clean_responses.append(response)
        
        # Get embeddings
        embeddings = model.encode(clean_responses)
        
        def semantic_diversity_function(indices_set):
            """Submodular function for semantic diversity."""
            if len(indices_set) <= 1:
                return 0.0
            
            subset_embeddings = embeddings[list(indices_set)]
            similarity_matrix = cosine_similarity(subset_embeddings)
            
            n = len(indices_set)
            total_similarity = np.sum(similarity_matrix) - n
            avg_similarity = total_similarity / (n * (n - 1)) if n > 1 else 0.0
            
            # Diversity score with submodular property
            diversity_score = (1.0 - avg_similarity) * np.sqrt(len(indices_set))
            return max(0.0, diversity_score)
        
        # Calculate marginal contributions for each sample
        smi_rewards = []
        for i in range(len(responses)):
            # S: set of all indices except i
            other_indices = set(range(len(responses))) - {i}
            
            # f(S)
            f_without = semantic_diversity_function(other_indices)
            
            # f(S ∪ {i})
            f_with = semantic_diversity_function(other_indices | {i})
            
            # Marginal gain
            marginal_gain = f_with - f_without
            smi_rewards.append(marginal_gain)
        
        return smi_rewards
        
    except Exception as e:
        print(f"Semantic SMI calculation failed: {e}")
        return [0.0] * len(responses)

##########################################
# Optional...
# Rewards for diverse length completions
# def length_diversity_reward_func(prompts: str, completions: str, **kwargs) -> list[float]:
    
