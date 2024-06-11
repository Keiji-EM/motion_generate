import re

import numpy as np


# bvh_fileは読み込むbvhファイルのパス
# n_framesは読み込むフレーム数
def bvh_to_numpy(bvh_file, n_frames):
    with open(bvh_file) as file:

        # フレーム数を取得
        while (match := re.fullmatch(r"^Frames:\s+(?P<frames>\d+)\s*", next(file))) is None:
            pass

        # フレーム数が足りない場合はNone
        if int(match["frames"]) < n_frames:
            return None

        return np.loadtxt(file, skiprows=1, max_rows=n_frames, dtype=np.float32)
