import itertools
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

import model

dataset_file = ""  # 入力ファイル(.npy)
param_dir = ""  # パラメータの保存先
eval_interval = 50  # 評価間隔
batch_size = 1  # バッチサイズ
num_workers = 1  # データローダーのワーカー数
learning_rate = 0.00001  # 学習率
seed = 0  # シード値
random_split_seed = 0  # データセットをランダムに分割するときのシード値

torch.manual_seed(seed)  # https://pytorch.org/docs/stable/notes/randomness.html
param_dir = Path(param_dir)
param_dir.mkdir(parents=True, exist_ok=True)

dataset = torch.from_numpy(np.load(dataset_file))  # データ数 x 自由度 x フレーム数
dataset_mean = torch.mean(dataset, dim=(0, 2), keepdim=True)  # 各自由度の平均
dataset_std = torch.std(dataset)  # 全体の標準偏差
dataset = (dataset - dataset_mean) / dataset_std  # 標準化

dataset = torch.utils.data.TensorDataset(dataset)
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9),
                                                                       len(dataset) - int(len(dataset) * 0.9)],
                                                             torch.Generator().manual_seed(random_split_seed))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                              drop_last=True,
                              shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                              drop_last=True,
                              shuffle=True) #drop_lastは256以上ある場合はデータを捨てる　重複なく取るような処理

torch.backends.cudnn.benchmark = True  # 学習の高速化


def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.zeros_(m.bias.data)


auto_encoder = model.AutoEncoder().to("cuda")

auto_encoder.apply(weights_init)

# data = next(iter(train_dataloader)).to("cuda")
# writer.add_graph(auto_encoder, data)

# scaler = torch.cuda.amp.GradScaler()
optim = torch.optim.Adam(auto_encoder.parameters(), lr=learning_rate)

loss_f = nn.MSELoss()
iter_ = 0

for epoch in itertools.count():
    for train_data in train_dataloader:
        train_data = train_data[0]

        # ノイズの追加
        mask = torch.rand_like(train_data) > 0.1
        train_input = train_data * mask
        # train_data = train_data.to('cuda', non_blocking=True)
        train_data = train_data.to('cuda')

        train_input = train_input.to("cuda")

        optim.zero_grad()
        # with torch.cuda.amp.autocast():
        train_out = auto_encoder(train_input)
        train_err = loss_f(train_data, train_out)

        train_err_ = train_err.item()
        # writer.add_scalars('loss', {'train': train_err_}, iter_)

        if iter_ % eval_interval == 0:

            valid_err_ = 0
            valid_iters = 0
            for valid_data in valid_dataloader:
                valid_data = valid_data[0]
                # ノイズの付加
                mask = torch.rand_like(valid_data) > 0.1
                valid_input = valid_data * mask
                valid_data = valid_data.to('cuda', non_blocking=True)

                valid_input = valid_input.to("cuda", non_blocking=True)
                with torch.no_grad():
                    # with torch.cuda.amp.autocast():
                    valid_output = auto_encoder(valid_input)
                    valid_err = loss_f(valid_data, valid_output)
                valid_iters += 1
                valid_err_ += valid_err.item()

            valid_err_ /= valid_iters

            print("epoch:", epoch, "iter:", iter_, "train_loss:", f"{train_err_:.2f}", "valid_loss",
                  f"{valid_err_:.2f}")
            # writer.add_scalars('loss', {'valid': valid_err_}, iter_)

            torch.save({'epoch': epoch,
                        'iters': iter_,
                        'time': time.time(),
                        'train_loss': train_err_,
                        'valid_loss': valid_err_,
                        'network': auto_encoder.state_dict(),
                        'optim': optim.state_dict()},
                       param_dir / f'{iter_:06d}.pth')

        # caler.scale(train_err).backward()
        train_err.backward()

        # scaler.step(optim)
        optim.step()

        # scaler.update()

        iter_ += 1