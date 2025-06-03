import os
import json
import torch
import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG") else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if os.environ.get("DEBUG"):
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug mode enabled")

def structured_pruning(model, amount=0.1, exclude_layers=None):
    """
    Apply structured pruning to a model by zeroing out neurons with smallest L2 norm.
    
    Args:
        model: PyTorch model
        amount: Fraction of neurons to prune (0.0 to 1.0)
        exclude_layers: List of layer names to exclude from pruning
    
    Returns:
        model: Pruned model
    """
    logger.info("Applying structured pruning...")
    
    if exclude_layers is None:
        exclude_layers = []
    
    total_neurons = 0
    pruned_neurons = 0
    
    # Iterate through named modules
    for name, module in model.named_modules():
        # Only prune linear layers
        if isinstance(module, torch.nn.Linear):
            # Skip excluded layers
            if any(excluded in name for excluded in exclude_layers):
                logger.info(f"Skipping pruning for excluded layer: {name}")
                continue
            
            # Get output dimension (number of neurons)
            out_features = module.out_features
            total_neurons += out_features
            
            # Use reduced pruning amount for attention layers
            if "attention" in name:
                reduced_amount = amount * 0.2  # 20% of the original pruning amount
                logger.info(f"Using reduced pruning amount ({reduced_amount:.3f}) for attention layer: {name}")
                current_amount = reduced_amount
            else:
                current_amount = amount
            
            # Calculate number of neurons to prune
            num_to_prune = int(out_features * current_amount)
            
            if num_to_prune == 0:
                logger.info(f"No neurons to prune in layer {name}")
                continue
                
            logger.info(f"Layer {name}: pruning {num_to_prune} out of {out_features} neurons")
            
            # Calculate L2 norm of each neuron (output dimension)
            weight = module.weight.data
            norm_per_neuron = torch.norm(weight, p=2, dim=1)
            
            # Find the smallest norms
            _, indices = torch.topk(norm_per_neuron, k=num_to_prune, largest=False)
            
            # Zero out the neurons with smallest norm
            for idx in indices:
                module.weight.data[idx, :] = 0
                if module.bias is not None:
                    module.bias.data[idx] = 0
            
            pruned_neurons += num_to_prune
    
    logger.info(f"Pruned {pruned_neurons} out of {total_neurons} neurons ({pruned_neurons/total_neurons*100:.2f}%)")
    return model

def main():
    # Load model info from JSON file
    model_info_path = "model_info.json"
    logger.info(f"Loading model info from {model_info_path}")
    
    try:
        with open(model_info_path, "r") as f:
            model_info = json.load(f)
        logger.info(f"Model info loaded: {json.dumps(model_info, indent=2)}")
    except FileNotFoundError:
        logger.error(f"Model info file not found: {model_info_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in model info file: {model_info_path}")
        return
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory created: {output_dir}")
    
    # Process each model in the model info
    for model_key, model_data in model_info.items():
        model_name = model_data.get("model_name")
        task = model_data.get("task")
        
        if not model_name or not task:
            logger.warning(f"Missing model_name or task for {model_key}, skipping")
            continue
        
        logger.info(f"Processing model: {model_name} for task: {task}")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            continue
        
        # Load model based on task
        logger.info(f"Loading model: {model_name}")
        try:
            if task == "token-classification":
                logger.info("Loading token classification model...")
                model = AutoModelForTokenClassification.from_pretrained(model_name)
            else:
                logger.error(f"Unsupported task: {task}")
                continue
            
            logger.info("Model loaded successfully")
            logger.debug(f"Model structure: {model}")
            logger.debug(f"Model config: {model.config}\n")
            
            # Get model size
            param_size = sum(p.numel() for p in model.parameters())
            logger.info(f"Model size: {param_size} parameters")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            continue
        
        # Apply structured pruning with very minimal amount
        pruning_amount = 0.03  # 3% pruning
        logger.info(f"Applying structured pruning with amount {pruning_amount}")
        
        # Exclude classifier layer from pruning
        exclude_layers = ["classifier"]
        model = structured_pruning(model, amount=pruning_amount, exclude_layers=exclude_layers)
        logger.info("Pruning applied successfully")
        
        # Save pruned model
        output_model_dir = os.path.join(output_dir, f"{model_key}_pruned_minimal")
        logger.info(f"Saving pruned model to {output_model_dir}")
        
        try:
            model.save_pretrained(output_model_dir)
            tokenizer.save_pretrained(output_model_dir)
            
            # Verify saved files
            saved_files = os.listdir(output_model_dir)
            logger.info(f"Saved files: {saved_files}")
            logger.info("Pruned model saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save pruned model: {e}")

if __name__ == "__main__":
    main()
