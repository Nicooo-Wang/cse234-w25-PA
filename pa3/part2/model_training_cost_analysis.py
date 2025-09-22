import argparse
import json
import math

def model_training_cost_analysis_llama(model_config_path):
    with open(model_config_path, "r") as f:
        data = json.load(f)

    vocab_size = data["vocab_size"]
    hidden_size = data["hidden_size"]
    intermediate_size = data["intermediate_size"]
    num_hidden_layers = data["num_hidden_layers"]
    num_attention_heads = data["num_attention_heads"]
    max_sequence_length = data["max_sequence_length"]

    # params
    params_emb = vocab_size * hidden_size
    params_layer_norm = hidden_size
    params_self_atten = 4 * hidden_size * hidden_size
    params_ffn = 3 * hidden_size * intermediate_size
    params_linear_trans = vocab_size * hidden_size
    total_params = (
        num_hidden_layers * (params_layer_norm * 2 + params_self_atten + params_ffn)
        + params_emb
        + params_linear_trans
    )

    # flops
    flops_layer_TF = (
        6 * max_sequence_length * hidden_size**2
        + 4 * max_sequence_length**2 * hidden_size
        + 3 * max_sequence_length**2 * num_attention_heads
        + 2 * max_sequence_length * hidden_size**2
        + 6 * max_sequence_length * hidden_size * intermediate_size
    ) / 1e12

    # mem
    params_per_layer = params_layer_norm * 2 + params_self_atten + params_ffn
    peak_memory_GB = 0
    peak_memory_GB += 2 * params_per_layer  # weights
    peak_memory_GB += 12 * params_per_layer  # weights
    peak_memory_GB += 2 * hidden_size * max_sequence_length # activations
    peak_memory_GB /= 1e9

    return total_params, flops_layer_TF, peak_memory_GB

def model_training_cost_analysis_deepseek(model_config_path):
    #TODO you code here.
    

    return total_params, flops_layer_TF, peak_memory_GB

def get_optimal_N_D_from_cost(cost_budget):
    """
    cost_budget:  a monetary training budget (in dollars)
    Returns:
        N: Optimal total model parameters (in absolute numbers)
        D: Optimal number of training tokens (in absolute numbers)
        training_budget_flops: Effective total training FLOPs (in FLOPs)
        best_gpu: name of the selected GPU (one of 'A100', 'V100', 'T4')
    """
    #TODO you code here

    return N, D, training_budget_flops, best_gpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training cost analysis')
    parser.add_argument('--model_config', type=str, help='Path to model config file')
    parser.add_argument('--training_budget', type=float, default=None, help='Training budget')
    args = parser.parse_args()

    if args.model_config:
        if 'deepseek' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_deepseek(args.model_config)
        elif 'llama' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_llama(args.model_config)
        else:
            print('Unknown LLM Type!')
            exit()
        print(f"Number of parameters: {num_parameters}")
        print(f"Number of TFLOPs: {num_flops}")
        print(f"Peak memory cost: {memory_cost} GBs")

    if args.training_budget:    
        N, D, training_budget_flops, best_gpu = get_optimal_N_D_from_cost(args.training_budget)
        print(f"best_gpu: {best_gpu}")
        print(f"training_budget_flops: {training_budget_flops}")
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")

    
