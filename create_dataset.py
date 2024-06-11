import re
from pathlib import Path
import bvh

import numpy as np

bvh_dataset_dir = ""  # BVHファイルのディレクトリ
output_dataset_file = ""  # 出力ファイル
n_frames = 216  # フレーム数 8の倍数が良い

i = []
joint_windows = []
for bvh_file in Path(bvh_dataset_dir).rglob("*.bvh"):
    bvh_np = bvh.bvh_to_numpy(bvh_file, n_frames)
    print(bvh_np.shape)
    if bvh_np is not None:
        bvh_f = []
        for i in range(bvh_np.shape[1] // 6):
            bvh_f.append(bvh_np[:,6 * i + 3:6 * i + 6])
        bvh_stack = np.hstack(bvh_f)
        joint_windows.append(bvh_stack)

dataset = np.stack(joint_windows).transpose((0, 2, 1))  # データ数 x 自由度 x フレーム数
print(dataset.shape)
#print(dataset)
# 標準化
# dataset_mean = np.mean(dataset, axis=(0, 2), keepdims=True)  # 各自由度の平均
# dataset_std = np.std(dataset)  # 全体の標準偏差
# dataset = (dataset - dataset_mean) / dataset_std  # 標準化

np.save(output_dataset_file, dataset)