from __future__ import division
from __future__ import print_function

import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN

# 设置是否使用CUDA
USE_CUDA = torch.cuda.is_available()

# 设置随机种子
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if USE_CUDA:
    torch.cuda.manual_seed(seed)

# 加载数据
adj, features, labels, idx_train, idx_val, idx_test = load_data()

if USE_CUDA:
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# 定义超参数网格
param_grid = {
    'lr': [0.01, 0.001],            # 学习率
    'weight_decay': [5e-4, 1e-3],   # 权重衰减
    'hidden': [16, 32],             # 隐藏单元数
    'dropout': [0.3, 0.5],          # Dropout率
    'epochs': [1000]                # 最大训练轮数
}

# 生成所有超参数组合
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# 记录每组参数的验证集损失和准确率，以及损失曲线
results = []

# 定义训练函数
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
    best_model_state = None

    # 记录损失和准确度曲线
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(params['epochs']):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

        # 记录损失和准确度
        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())
        train_accs.append(acc_train.item())
        val_accs.append(acc_val.item())

        # 打印训练和验证信息
        print(f"Params: lr={params['lr']}, weight_decay={params['weight_decay']}, hidden={params['hidden']}, dropout={params['dropout']}")
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        # 检查是否有提升
        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            best_val_acc = acc_val.item()
            best_model_state = model.state_dict()

    return {
        'params': params,
        'val_loss': best_val_loss,
        'val_acc': best_val_acc,
        'best_state': best_model_state,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

# 超参数调优
for idx, params in enumerate(param_combinations):
    print(f"\n=== Training combination {idx+1}/{len(param_combinations)} ===")
    result = train_model(params)
    results.append(result)

# 选择验证集上表现最好的超参数组合
best_result = max(results, key=lambda x: x['val_acc'])
best_params = best_result['params']
print("\n=== Best Hyperparameters ===")
print(f"Learning Rate: {best_params['lr']}")
print(f"Weight Decay: {best_params['weight_decay']}")
print(f"Hidden Units: {best_params['hidden']}")
print(f"Dropout: {best_params['dropout']}")
print(f"Validation Accuracy: {best_result['val_acc']:.4f}")

# 使用最佳超参数重新训练模型，并保存最佳模型参数
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
best_model_state = None

# 记录损失和准确度曲线
final_train_losses = []
final_val_losses = []
final_train_accs = []
final_val_accs = []

for epoch in range(best_params['epochs']):
    t = time.time()
    final_model.train()
    optimizer.zero_grad()
    output = final_model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # 验证
    final_model.eval()
    with torch.no_grad():
        output = final_model(features, adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

    # 记录损失和准确度
    final_train_losses.append(loss_train.item())
    final_val_losses.append(loss_val.item())
    final_train_accs.append(acc_train.item())
    final_val_accs.append(acc_val.item())

    # 打印训练和验证信息
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    # 检查是否有提升
    if loss_val.item() < best_val_loss:
        best_val_loss = loss_val.item()
        best_val_acc = acc_val.item()
        best_model_state = final_model.state_dict()

print("Training Finished!")

# 加载最佳模型参数
final_model.load_state_dict(best_model_state)

# 测试函数
def test(model):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

# 测试最终模型
test(final_model)

# 绘制损失曲线和准确度曲线
import matplotlib.pyplot as plt

# 绘制验证损失曲线
plt.figure(figsize=(12, 8))
for idx, result in enumerate(results):
    params = result['params']
    val_losses = result['val_losses']
    label = f"lr={params['lr']}, wd={params['weight_decay']}, hid={params['hidden']}, drop={params['dropout']}"
    plt.plot(val_losses, label=label)

plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Curves for Different Hyperparameter Combinations')
plt.legend(fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# 如果组合数量较多，可以选择仅绘制验证损失最低的前N个组合
# 例如，绘制验证准确率前5的组合
top_n = 5
sorted_results = sorted(results, key=lambda x: x['val_acc'], reverse=True)[:top_n]

plt.figure(figsize=(12, 8))
for idx, result in enumerate(sorted_results):
    params = result['params']
    val_losses = result['val_losses']
    label = f"lr={params['lr']}, wd={params['weight_decay']}, hid={params['hidden']}, drop={params['dropout']}"
    plt.plot(val_losses, label=label)

plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title(f'Top {top_n} Validation Loss Curves')
plt.legend(fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# 绘制验证准确度曲线
plt.figure(figsize=(12, 8))
for idx, result in enumerate(results):
    params = result['params']
    val_accs = result['val_accs']
    label = f"lr={params['lr']}, wd={params['weight_decay']}, hid={params['hidden']}, drop={params['dropout']}"
    plt.plot(val_accs, label=label)

plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Curves for Different Hyperparameter Combinations')
plt.legend(fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# 绘制Top N的验证准确度曲线
plt.figure(figsize=(12, 8))
for idx, result in enumerate(sorted_results):
    params = result['params']
    val_accs = result['val_accs']
    label = f"lr={params['lr']}, wd={params['weight_decay']}, hid={params['hidden']}, drop={params['dropout']}"
    plt.plot(val_accs, label=label)

plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title(f'Top {top_n} Validation Accuracy Curves')
plt.legend(fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()
