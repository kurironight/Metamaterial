import torch


def evaluate(model, loss_func, optimizer, input_loader, batch_size, cuda):
    model.eval()
    evaluate_loss = 0
    with torch.no_grad():
        for _, [structure, E, G] in enumerate(input_loader):
            # 一番最後、バッチサイズに満たない場合は無視する
            if (E.size()[0] != batch_size):
                break
            # 条件入力ベクトルを入れたい
            if cuda:
                structure, E, G = structure.cuda(), E.cuda(), G.cuda()
            output_structure = model(E, G)
            loss = loss_func(output_structure, structure)
            evaluate_loss += loss.data
    evaluate_loss /= len(input_loader)

    return evaluate_loss
