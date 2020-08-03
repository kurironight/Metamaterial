
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import time
import DL.model
from DL.train import train
from DL.evaluate import evaluate
from tqdm import tqdm
from DL.tools import split_data_train_eval_test
from DL.show_result import plot_history

test = 0
# hyperparameters
batch_size = 64
lr = 0.001
num_epochs = 1000
model = DL.model.FirstModelBatch()

log_dir = "data/results/{}_epoch{}_lr{}".format(model.name, num_epochs, lr)

data_dir = "data/bar_nx_32_ny_32/gen_500_pa_640_vf0.4"
load_dir = 0


if(load_dir != 0):
    test = 1
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

with open(os.path.join(log_dir, "model.txt"), mode='w') as f:
    f.writelines("model_name:{}\n".format(model.name))
    f.writelines("data_path:{}\n".format(data_dir))
    f.writelines("DL_model:{}\n".format(model()))

cuda = torch.cuda.is_available()

if cuda:
    print('cuda available!')
    if torch.cuda.device_count() > 1:
        print('\nYou can use {} GPU'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model).cuda()
    else:
        print('\n you use only one GPU')
        model.cuda()


# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

# loss
MAE = nn.L1Loss()

# dataload
train_loader, eval_loader, test_loader = split_data_train_eval_test(
    data_dir, batch_size=batch_size)

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
    with open(os.path.join(log_dir, "progress.txt"), mode='a') as f:
        f.writelines('epoch %d, train_loss: %.4f eval_loss: %.4f test_loss: %.4f\n' %
                     (epoch + 1, train_loss, eval_loss, test_loss))
min_test_loss, min_epoch = plot_history(
    history, os.path.join(log_dir, 'learning_curve.png'))
with open(os.path.join(log_dir, "represent_value.txt"), mode='w') as f:
    f.writelines('epoch %d,  minimum of testloss: %.4f\n' %
                 (min_epoch, min_test_loss))
