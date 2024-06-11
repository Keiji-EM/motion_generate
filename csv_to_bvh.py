import numpy as np

csv_file = "" # 入力CSVファイル
bvh_file = "" # 入力BVHファイル
out_file = "" # 出力BVHファイル

structure_str = ""

with open(bvh_file) as file:
    while True:
        line = next(file)
        structure_str += line
        if line.startswith("Frame Time"):
            break

    bvh_np = np.loadtxt(file, dtype=np.float32)

csv_np = np.loadtxt(csv_file, delimiter=",", dtype=np.float32)

for i in range(csv_np.shape[1] // 3):
    bvh_np[:, 6 * i + 3 : 6 * i + 6] = csv_np[:, 3 * i : 3 * i + 3]

np.savetxt(out_file, bvh_np, fmt="%f", header=structure_str[:-1], comments="")