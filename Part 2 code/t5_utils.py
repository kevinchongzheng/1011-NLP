import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config, T5TokenizerFast
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
# import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    '''
    Setup Weights & Biases for experiment tracking.
    Implement this if you wish to use wandb.
    '''
    # import wandb
    # wandb.init(
    #     project="text-to-sql",
    #     name=args.experiment_name,
    #     config=vars(args)
    # )
    pass


def initialize_model(args):
    '''
    Initialize T5 model - either fine-tune pretrained or train from scratch.
    
    Args:
        args: Must have args.finetune (bool)
    
    Returns:
        T5ForConditionalGeneration model
    '''
    if args.finetune:
        # Load pretrained T5-small and fine-tune all parameters
        model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
        print("Loaded pretrained T5-small model for fine-tuning")
    else:
        # Initialize from scratch using T5-small config
        config = T5Config.from_pretrained("google-t5/t5-small")
        model = T5ForConditionalGeneration(config)
        print("Initialized T5-small model from scratch")
    
    # Attach tokenizer to model for convenience
    model.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    
    model.to(DEVICE)
    return model


def mkdir(dirpath):
    '''Create directory if it doesn't exist.'''
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass


def save_model(checkpoint_dir, model, best):
    '''
    Save model checkpoint.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        model: Model to save
        best: If True, save as 'model_best.pt', else 'model_last.pt'
    '''
    mkdir(checkpoint_dir)
    filename = "model_best.pt" if best else "model_last.pt"
    path = os.path.join(checkpoint_dir, filename)
    
    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    print(f"Saved {'best' if best else 'last'} model to {path}")


def load_model_from_checkpoint(args, best):
    '''
    Load model from checkpoint.
    
    Args:
        args: Must have args.checkpoint_dir and args.finetune
        best: If True, load 'model_best.pt', else 'model_last.pt'
    
    Returns:
        Loaded model
    '''
    # Initialize model with same architecture
    model = initialize_model(args)
    
    filename = "model_best.pt" if best else "model_last.pt"
    path = os.path.join(args.checkpoint_dir, filename)
    
    if not os.path.exists(path):
        print(f"Warning: Checkpoint {path} not found. Returning freshly initialized model.")
        return model
    
    # Load state dict
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.to(DEVICE)
    
    print(f"Loaded {'best' if best else 'last'} model from {path}")
    return model


def initialize_optimizer_and_scheduler(args, model, epoch_length):
    '''
    Initialize optimizer and learning rate scheduler.
    
    Args:
        args: Training arguments
        model: Model to optimize
        epoch_length: Number of batches per epoch
    
    Returns:
        optimizer, scheduler
    '''
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler


def initialize_optimizer(args, model):
    '''
    Initialize AdamW optimizer with weight decay.
    
    Applies weight decay to all parameters except biases and LayerNorm parameters.
    '''
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer_type} not implemented")

    return optimizer

        
def initialize_scheduler(args, optimizer, epoch_length):
    '''
    Initialize learning rate scheduler.
    
    Args:
        args: Must have scheduler_type, max_n_epochs, num_warmup_epochs
        optimizer: Optimizer to schedule
        epoch_length: Number of batches per epoch
    
    Returns:
        Scheduler or None
    '''
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError(f"Scheduler {args.scheduler_type} not implemented")


def get_parameter_names(model, forbidden_layer_types):
    '''
    Get all parameter names in model, excluding forbidden layer types.
    
    Used to determine which parameters should have weight decay applied.
    '''
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result
