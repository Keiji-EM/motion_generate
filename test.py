# This program tests whether learning is taking place.
from pathlib import Path

import numpy as np
import torch

import bvh
import model

torch.backends.cudnn.benchmark = True  # 学習の高速化

dataset_file = ""  # データセットファイル(.npy)
input_BVHs = [""] # 入力するBVHファイルのリスト
param_file = ""  # パラメータファイル
output_dir = ""  # 出力ディレクトリ
n_frames = 216  # フレーム数 8の倍数が良い

joint_windows = []
output_files = []
for bvh_file in input_BVHs:
    bvh_np = bvh.bvh_to_numpy(bvh_file, n_frames)
    if bvh_np is not None:
        bvh_f = []
        for i in range(bvh_np.shape[1] // 6):
            bvh_f.append(bvh_np[:,6 * i + 3:6 * i + 6])
        bvh_stack = np.hstack(bvh_f)
        joint_windows.append(bvh_stack)

dataset = np.load(dataset_file)  # データ数 x 自由度 x フレーム数
dataset_mean = np.mean(dataset, axis=(0, 2), keepdims=True)  # 各自由度の平均
dataset_std = np.std(dataset)  # 全体の標準偏差

test_dataset = np.stack(joint_windows).transpose((0, 2, 1))  # データ数 x 自由度 x フレーム数
test_dataset = (test_dataset - dataset_mean) / dataset_std  # 標準化
test_dataset = torch.from_numpy(test_dataset).to("cuda")

auto_encoder = model.AutoEncoder().to("cuda")
auto_encoder.load_state_dict(torch.load(param_file)["network"])

with torch.no_grad():
    output = auto_encoder(test_dataset).cpu()

output_motions = output.numpy() * dataset_std + dataset_mean
output_motions = output_motions.transpose((0, 2, 1))  # データ数 x フレーム数 x 自由度
print(output_motions.shape)

output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
for input_BVH, output_motion in zip(map(Path, input_BVHs), output_motions):
    np.savetxt(output_dir / input_BVH.with_suffix(".csv").name, output_motion, fmt="%f", delimiter=",")