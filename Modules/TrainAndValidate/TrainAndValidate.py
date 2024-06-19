"""
FILENAME: TrainAndValidate.py
DESCRIPTION: function defintion for training and validation 
@author: Jian Zhong
"""

import torch


# train encoder and decoder for one epoch
def train_one_epoch(
    encoder_model, 
    decoder_model,
    train_loader,
    data_loss_func,
    optimizer,
    code_loss_rate = 0,
    code_loss_func = None,
    device = None,
):
    tot_loss = 0.0
    avg_loss = 0.0
    tot_nof_batch = 0

    encoder_model.train(True)
    decoder_model.train(True)

    for i_batch, data in enumerate(train_loader):        
        inputs, targets = data

        if device:
            inputs = inputs.to(device)
            targets = targets.to(device)

        optimizer.zero_grad()
        
        cur_codes = encoder_model(inputs) # encode input data into codes in latent space
        cur_preds = decoder_model(cur_codes) # reconstruct input image
        
        data_loss = data_loss_func(cur_preds, targets) 
        code_loss = 0
        if code_loss_func:
            code_loss = code_loss_func(cur_codes)
        loss = data_loss + code_loss_rate * code_loss
        
        loss.backward()
        optimizer.step()
        
        tot_loss += loss.item()
        tot_nof_batch += 1

        if i_batch % 100 == 0:
            print(f"batch {i_batch} loss: {tot_loss/tot_nof_batch: >8f}")

    avg_loss = tot_loss/tot_nof_batch

    print(f"Train: Avg loss: {avg_loss:>8f}")

    return avg_loss


# validate encoder and decoder for one epoch
def validate_one_epoch(
    encoder_model,
    decoder_model,
    validate_loader,
    loss_func,
    device = True,
):
    tot_loss = 0.0
    avg_loss = 0.0
    tot_nof_batch = len(validate_loader)
    tot_samples = len(validate_loader.dataset)

    encoder_model.eval()
    decoder_model.eval()
    with torch.no_grad():
        for i_batch, data in enumerate(validate_loader):
            inputs, targets = data

            if device:
                inputs = inputs.to(device)
                targets = targets.to(device)

            cur_codes = encoder_model(inputs) # encode input data into codes in latent space
            cur_preds = decoder_model(cur_codes) # reconstruct input image
        
            loss = loss_func(cur_preds, targets)
            tot_loss += loss.item()

    avg_loss = tot_loss/tot_nof_batch

    print(f"Validate: Avg loss: {avg_loss: > 8f}")

    return avg_loss
