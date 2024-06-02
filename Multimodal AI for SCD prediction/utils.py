#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses, train_survlosses, val_survlosses):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))

    # Plot training and validation reconstruction losses
    ax1.plot(range(1, len(train_losses) + 1), train_losses, label='Training Recon Loss', marker='o')
    ax1.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Recon Loss', marker='o')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot training and validation survival losses
    ax2.plot(range(1, len(train_survlosses) + 1), train_survlosses, label='Training Surv Loss', marker='o')
    ax2.plot(range(1, len(val_survlosses) + 1), val_survlosses, label='Validation Surv Loss', marker='o')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()  # Ensure proper spacing between subplots
    plt.show()


def plot_losses_unsuper(train_losses, val_losses):
    # Create a figure with a single subplot
    fig, ax = plt.subplots(figsize=(5, 6))

    # Plot training and validation reconstruction losses on the same subplot (ax)
    ax.plot(range(1, len(train_losses) + 1), train_losses, label='Training Recon Loss', marker='o')
    ax.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Recon Loss', marker='o')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    plt.tight_layout()  # Ensure proper spacing
    plt.show()

import torch
def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def save_checkpoint(epoch, model, optimizer, loss, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, save_path)

import os
def save_checkpoint_intraining(epoch, vae, optimizer, best_loss, checkpoint_folder, checkpoint_interval, checkpoint_counter, active=False):
    if active==True:
        if (epoch + 1) % checkpoint_interval == 0:
            # Save the model checkpoint
            checkpoint_counter += 1
            checkpoint_filename = f'checkpoint_{checkpoint_counter:03d}.pth'
            checkpoint_path = os.path.join(checkpoint_folder, checkpoint_filename)
            save_checkpoint(epoch, vae, optimizer, best_loss, checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch+1}')

def load_checkpoint_if_exists(vae, optimizer, checkpoint_path, load=True):
    if load and os.path.exists(checkpoint_path):
        vae, optimizer, start_epoch, best_loss = load_checkpoint(vae, optimizer, checkpoint_path)
        print(f"Checkpoint found! {start_epoch+1}")
    else:
        print("Checkpoint not found or loading is disabled.")

def append_losses(train_losses, val_losses, train_survlosses, val_survlosses, avg_loss, avg_val_loss, avg_survival_loss, avg_val_survival_loss):
    train_losses.append(avg_loss)
    val_losses.append(avg_val_loss)
    train_survlosses.append(avg_survival_loss)
    val_survlosses.append(avg_val_survival_loss)


def append_losses_unsuper(train_losses, val_losses, avg_loss, avg_val_loss):
    train_losses.append(avg_loss)
    val_losses.append(avg_val_loss)


def update_total_losses(total_loss, total_survival_loss, total_hybrid_loss, total_kl_loss, loss, survival_loss, hybrid_loss, kl_loss):
    total_loss += loss.item()
    total_survival_loss += survival_loss.item()
    total_hybrid_loss += hybrid_loss.item()
    total_kl_loss += kl_loss.item()
    return total_loss, total_survival_loss, total_hybrid_loss, total_kl_loss


def update_total_losses_unsuper(total_loss, total_kl_loss, loss, kl_loss):
    total_loss += loss.item()
    total_kl_loss += kl_loss.item()
    return total_loss, total_kl_loss


def calculate_average_losses_unsuper(total_loss,total_kl_loss, train_dataloader):
    avg_loss = total_loss / len(train_dataloader)
    avg_kl_loss = total_kl_loss / len(train_dataloader)

    return avg_loss, avg_survival_loss

def calculate_average_losses_unsuper(total_loss, total_kl_loss, train_dataloader):
    avg_loss = total_loss / len(train_dataloader)
    avg_kl_loss = total_kl_loss / len(train_dataloader)
    return avg_loss, avg_kl_loss


def early_stopping(avg_val_survival_loss, best_loss, no_improvement_counter, early_stopping_patience, epoch, early_stopping_activate=True):
    if early_stopping_activate:
        if avg_val_survival_loss < best_loss:
            best_loss = avg_val_survival_loss
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
            if no_improvement_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
                early_stopping_activate = False  # Disable early stopping here
                return best_loss, no_improvement_counter, early_stopping_activate

    return best_loss, no_improvement_counter, early_stopping_activate



