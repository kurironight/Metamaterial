import torch

# for the training


def train(model, MAE, optimizer, input_loader, batch_size, cuda):
    # 訓練モードへ

    model.train()
    y_real = torch.ones(batch_size, 1)
    y_fake = torch.zeros(batch_size, 1)
    if cuda:
        y_real = y_real.cuda()
        y_fake = y_fake.cuda()

    G_running_loss = 0
    for ibatch_idx, [input_images, real_images, vol, empty] in enumerate(input_loader):
        # 一番最後、バッチサイズに満たない場合は無視する
        if (input_images.size()[0] != batch_size or real_images.size()[0] != batch_size or vol.size()[0] != batch_size or empty.size()[0] != batch_size):
            break

        z = input_images

        if cuda:
            real_images, z, vol, empty = real_images.cuda(), z.cuda(),
            vol.cuda(), empty.cuda()
        optimizer.zero_grad()
        fake_images = model(z, vol, empty)

        G_loss = MAE(fake_images, real_images)  # GlossにMAEを加えた
        G_loss.backward()
        optimizer.step()
        G_running_loss += G_loss.data

    G_running_loss /= len(input_loader)

    return G_running_loss
