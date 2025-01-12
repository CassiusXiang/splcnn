from __future__ import division
from __future__ import print_function

import time
import itertools
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN

USE_CUDA = torch.cuda.is_available()

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if USE_CUDA:
    torch.cuda.manual_seed(seed)

adj, features, labels, idx_train, idx_val, idx_test = load_data()

if USE_CUDA:
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

param_grid = {
    'lr': [0.01, 0.005, 0.001],
    'weight_decay': [5e-4, 1e-3, 5e-5],
    'hidden': [16, 32, 64],
    'dropout': [0.3, 0.5, 0.7],
    'epochs': [1000],
    'early_stopping': [100]
}

keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

results = []

def train_model(params):
    model = GCN(nfeat=features.shape[1],
                nhid=params['hidden'],
                nclass=labels.max().item() + 1,
                dropout=params['dropout'])
    if USE_CUDA:
        model.cuda()
    optimizer = optim.Adam(model.parameters(),
                           lr=params['lr'], weight_decay=params['weight_decay'])
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_count = 0  
    best_model_state = None

    for epoch in range(params['epochs']):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

        print(f"Params: lr={params['lr']}, weight_decay={params['weight_decay']}, hidden={params['hidden']}, dropout={params['dropout']}")
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))
        
        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            best_val_acc = acc_val.item()
            patience_count = 0
            best_model_state = model.state_dict()
        else:
            patience_count += 1

        if patience_count >= params['early_stopping']:
            print("Early stopping triggered at epoch: {}".format(epoch+1))
            break

    return best_val_loss, best_val_acc, best_model_state
for idx, params in enumerate(param_combinations):
    print(f"\n=== Training combination {idx+1}/{len(param_combinations)} ===")
    val_loss, val_acc, best_state = train_model(params)
    results.append({
        'params': params,
        'val_loss': val_loss,
        'val_acc': val_acc
    })

best_result = max(results, key=lambda x: x['val_acc'])
best_params = best_result['params']
print("\n=== Best Hyperparameters ===")
print(f"Learning Rate: {best_params['lr']}")
print(f"Weight Decay: {best_params['weight_decay']}")
print(f"Hidden Units: {best_params['hidden']}")
print(f"Dropout: {best_params['dropout']}")
print(f"Validation Accuracy: {best_result['val_acc']:.4f}")

print("\n=== Training Final Model with Best Hyperparameters ===")
final_model = GCN(nfeat=features.shape[1],
                 nhid=best_params['hidden'],
                 nclass=labels.max().item() + 1,
                 dropout=best_params['dropout'])
if USE_CUDA:
    final_model.cuda()
optimizer = optim.Adam(final_model.parameters(),
                       lr=best_params['lr'], weight_decay=best_params['weight_decay'])

best_val_loss = float('inf')
best_val_acc = 0.0
patience_count = 0
best_model_state = None

for epoch in range(best_params['epochs']):
    t = time.time()
    final_model.train()
    optimizer.zero_grad()
    output = final_model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    final_model.eval()
    with torch.no_grad():
        output = final_model(features, adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    
    if loss_val.item() < best_val_loss:
        best_val_loss = loss_val.item()
        best_val_acc = acc_val.item()
        patience_count = 0
        best_model_state = final_model.state_dict()
    else:
        patience_count += 1

    if patience_count >= best_params['early_stopping']:
        print("Early stopping triggered at epoch: {}".format(epoch+1))
        break

print("Training Finished!")

final_model.load_state_dict(best_model_state)

def test(model):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    
test(final_model)
