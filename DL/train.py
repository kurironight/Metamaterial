import torch

# for the training


def train(model, loss_func, optimizer, input_loader, batch_size):
    # 訓練モードへ

    model.train()

    running_loss = 0
    for ibatch_idx, [structure, E, G] in enumerate(input_loader):
        # 一番最後、バッチサイズに満たない場合は無視する
        if (structure.size()[0] != batch_size):
            break
        optimizer.zero_grad()
        output_structure = model(E, G)

        loss = loss_func(output_structure, structure)
        loss.backward()
        optimizer.step()
        running_loss += loss.data

    running_loss /= len(input_loader)

    return running_loss
