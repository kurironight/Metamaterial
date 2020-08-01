
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from DL.model import Generator
from DL.train import train
from DL.convert import convert_npy_to_torch
num = 0
test = 0
# hyperparameters
test_rate = 1/3
batch_size = 1
lr = 0.001
num_epochs = 10

#load_dir = './mae_nospade2'
log_dir = "results"
data_dir = "data/bar_nx_32_ny_32/gen_15_pa_5"
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
if cuda:
    structures, E_data, G_data = structures.cuda(), E_data.cuda(), G_data.cuda()
train_structures, test_structures,\
    train_E_data, test_E_data, \
    train_G_data, test_G_data = train_test_split(
        structures, E_data, G_data, test_size=test_rate, random_state=0)
train_dataloader = torch.utils.data.TensorDataset(
    train_structures, train_E_data, train_G_data)
input_loader = DataLoader(
    train_dataloader, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.TensorDataset(
    test_structures, test_E_data, test_G_data)
test_loader = DataLoader(
    test_loader, batch_size=batch_size, shuffle=True)


history = {}
history['train_loss'] = []
history['test_loss'] = []

best_loss = 1000
for epoch in range(num_epochs):
    start = time.time()
    G_loss = train(model, MAE, optimizer, input_loader, batch_size)
    elapsed_time = time.time()-start
    print(str(elapsed_time)+"sec")
    with torch.no_grad():
        test_loss = 0
    for tbatch_idx, [tinput_images, treal_images, tvol, tempty] in enumerate(test_loader):
        # 一番最後、バッチサイズに満たない場合は無視する
        if (tinput_images.size()[0] != batch_size or treal_images.size()[0] != batch_size or tvol.size()[0] != batch_size or tempty.size()[0] != batch_size):
            break
        tz = tinput_images  # 潜在ベクトル
        # 条件入力ベクトルを入れたい
        if cuda:
            tz, tvol, tempty = tz.cuda(), tvol.cuda(), tempty.cuda()
        start = time.time()
        #test_result=generate(epoch + 1, model, tz,tvol,tempty, log_dir)
        test_result = model(tz, tvol, tempty).data.cpu()
        loss = MAE(treal_images, test_result).cpu()
        test_loss = test_loss+loss.data
    test_loss = (test_loss / len(test_loader))
    if test_loss < best_loss:
        best_loss = test_loss
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(),
                       os.path.join(log_dir, 'GoodG.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(log_dir, 'GoodG.pth'))
    print('epoch %d, G_loss: %.4f test_loss: %.4f' %
          (epoch + 1, G_loss, test_loss))
    history['G_loss'].append(G_loss)
    history['test_loss'].append(test_loss)
    # 学習履歴を保存
    with open(os.path.join(log_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(),
                   os.path.join(log_dir, 'lastG.pth'))
    else:
        torch.save(model.state_dict(), os.path.join(log_dir, 'lastG.pth'))
