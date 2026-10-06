
import argparse
import math
from tqdm import tqdm
import torch
import os
import util
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, SequentialLR
import models.model_provider as model_provider


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
parser = argparse.ArgumentParser(
    prog="Train",
    description="Training args",
)
parser.add_argument("--model", "--m", required=True, help="Model Type")
parser.add_argument("--ckpt", "--checkpoint", "--c", required=False, help="Checkpoint to load from, from scratch if unspecified")
parser.add_argument("--dataset", "--data", "--d", required=True, help="Dataset to use")
parser.add_argument("--seed", "--s", required=True, help="Random seed")
parser.add_argument("--epoch", "--e", required=True, help="Epoches")
parser.add_argument("--batch_size", "--b", required=True, help="Batch size")
parser.add_argument("--learn_rate", "--lr", required=True, help="Learn rate")
parser.add_argument("--learn_rate_decay", "--lrd", required=True, help="Learn rate decay")
parser.add_argument("--weight_decay", "--wd", required=True, help="Weight decay")
parser.add_argument("--warmup_epoch_ratio", "--wm", required=True, help="Warmup epoch ratio")
parser.add_argument("--eval_epoch_ratio", "--ev", required=True, help="Eval epoch ratio")
args, _ = parser.parse_known_args()

MODEL_NAME = args.model  # "baseline", "bst", "belief_encoder"
LOAD_CHECKPOINT_PATH = args.ckpt # None for from scratch
DATASET_PATH = args.dataset
RANDOM_SEED = args.seed

EPOCHS = int(args.epoch)
BATCH_SIZE = int(args.batch_size)
LEARNING_RATE = int(args.learn_rate)
LR_DECAY_GAMMA = float(args.learn_rate_decay)  # exponential decay factor after warmup
WEIGHT_DECAY = float(args.weight_decay)
WARMUP_EPOCH_RATIO = 0 if LOAD_CHECKPOINT_PATH else float(args.warmup_epoch_ratio)
EVAL_EPOCH_RATIO = float(args.eval_epoch_ratio)

# =============================================================================



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)


def main():
    # model
    model = model_provider.get_model(MODEL_NAME)
    if LOAD_CHECKPOINT_PATH:
        util.load_checkpoint(LOAD_CHECKPOINT_PATH, model)
    model.to(device)

    # dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}.")
    data_dict = torch.load(DATASET_PATH)

    train_dataset, eval_dataset, test_dataset = util.load_dataset(DATASET_PATH, random_seed=RANDOM_SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch
    # convert epoch-level gamma to a per-step gamma
    gamma_per_step = LR_DECAY_GAMMA ** (1 / steps_per_epoch)
    print(f"Total training steps: {total_steps}, Steps per epoch: {steps_per_epoch}, Gamma per step: {gamma_per_step:.12f}")

    WARMUP_STEPS = steps_per_epoch*WARMUP_EPOCH_RATIO
    warmup_scheduler = LinearLR(optimizer, start_factor=0, end_factor=1.0, 
                                total_iters=WARMUP_STEPS)
    decay_scheduler = ExponentialLR(optimizer, gamma=gamma_per_step)

    scheduler = SequentialLR(optimizer, 
                             schedulers=[warmup_scheduler, decay_scheduler], 
                             milestones=[WARMUP_STEPS])


    print(f"--- Starting {MODEL_NAME.upper()} Training ---")
    EVAL_INTERVAL = int(math.ceil(steps_per_epoch * EVAL_EPOCH_RATIO))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        with tqdm(total=total_steps) as pbar:
            for step, batch in enumerate(train_loader, 1):
                batch = batch[0] # extract the "x" value
                batch = batch.to(device)

                optimizer.zero_grad()
                loss = model.compute_loss(batch)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                pbar.step()
                pbar.set_description(f"Epoch {epoch}/{EPOCHS} | Step {step}/{total_steps} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.12f}")

                if (step + 1) % EVAL_INTERVAL == 0:
                    print(f"Evaluating Epoch {epoch} at step {step}...")
                    # TODO
                    
        ckpt_path = f"{MODEL_NAME}_ep{epoch}.pt"
        util.save_checkpoint(model.state_dict(), ckpt_path)

    print("Testing Model Performance...")
    # TODO
    

if __name__ == "__main__":
    main()