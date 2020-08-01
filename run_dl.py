
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from DL.model import Generator
from DL.train import train
from DL.convert import convert_npy_to_torch
from DL.evaluate import evaluate
from tqdm import tqdm
from DL.tools import shuffle_dataset, plot_history


num = 0
test = 0
# hyperparameters
test_rate = 1/8
batch_size = 50
lr = 0.001
num_epochs = 5

#load_dir = './mae_nospade2'
log_dir = "results/"

data_dir = "data/bar_nx_32_ny_32/gen_500_pa_640_vf0.4"
load_dir = 0
model = Generator()

if(load_dir != 0):
    test = 1
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

cuda = torch.cuda.is_available()

if cuda:
    print('cuda available!')
    if torch.cuda.device_count() > 1:
        if(load_dir):
            PATH = os.path.join(load_dir, 'GoodG.pth')
            model.load_state_dict(torch.load(PATH))
        print('load for multi!')
        print('\nYou can use {} GPU'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model).cuda()
    else:
        print('\n you use only one GPU')
        model.cuda()
        if (load_dir):
            PATH = os.path.join(load_dir, 'GoodG.pth')
            model.load_state_dict(torch.load(PATH))
            print('load!')


# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

# loss
MAE = nn.L1Loss()

# dataload
structures, E_data, G_data = convert_npy_to_torch(data_dir)
structures, E_data, G_data = structures.float(), E_data.float(), G_data.float()
print("データの数: ", structures.shape[0])

train_structures, eval_structures,\
    test_structures = shuffle_dataset(structures)
train_E_data, eval_E_data,\
    test_E_data = shuffle_dataset(E_data)
train_G_data, eval_G_data,\
    test_G_data = shuffle_dataset(G_data)
train_loader = torch.utils.data.TensorDataset(
    train_structures, train_E_data, train_G_data)
eval_loader = torch.utils.data.TensorDataset(
    eval_structures, eval_E_data, eval_G_data)
test_loader = torch.utils.data.TensorDataset(
    test_structures, test_E_data, test_G_data)

train_loader = DataLoader(
    train_loader, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(
    eval_loader, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(
    test_loader, batch_size=batch_size, shuffle=True)

history = {}
history['epoch'] = []
history['train_loss'] = []
history['eval_loss'] = []
history['test_loss'] = []

best_loss = 1000
for epoch in tqdm(range(num_epochs)):
    start = time.time()
    train_loss = train(model, MAE, optimizer, train_loader, batch_size, cuda)
    eval_loss = evaluate(model, MAE, optimizer,
                         eval_loader, batch_size, cuda)
    test_loss = evaluate(model, MAE, optimizer,
                         test_loader, batch_size, cuda)
    if eval_loss < best_loss:
        best_loss = eval_loss
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(),
                       os.path.join(log_dir, 'Good.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(log_dir, 'Good.pth'))
    history['epoch'].append(epoch+1)
    history['train_loss'].append(train_loss)
    history['eval_loss'].append(eval_loss)
    history['test_loss'].append(test_loss)
    # 学習履歴を保存
    with open(os.path.join(log_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(),
                   os.path.join(log_dir, 'last.pth'))
    else:
        torch.save(model.state_dict(), os.path.join(log_dir, 'last.pth'))
plot_history(history, os.path.join(log_dir, 'learning_curve.png'))
