# Motion Generate - AIによるゴルフスイング補正モーション生成

## 概要

初級者のゴルフスイングに多く見られる**スライス現象**を補正するため、
**モーションキャプチャデータ**と**機械学習（畳み込みオートエンコーダ）**を用いて、
スイング動作の特徴を抽出・学習し、理想的なスイングモーションの生成と比較分析を行います。


https://github.com/user-attachments/assets/4929af2e-655b-48f0-bd01-4d56aede173d


また、**動的時間伸縮法（Dynamic Time Warping：DTW）**を用いて、
生成モーションと実際のスイングデータの顕似度を定量・定性的に評価し、補正指標を提示します。

![dtw_readme_ready](https://github.com/user-attachments/assets/04d560cb-de36-473c-ac97-ac46a6738ffa)


---

## 特徴

- **モーションデータの次元圧縮と特徴抽出**  
  番み込みオートエンコーダ（CAE）により、スイングモーションの本質的な特徴を学習。

- **理想的なスイングモーションの生成**  
  学習済みCAEに実準データを入力することで、補正用モーションを生成。

- **スイング比較と補正指標の提示**  
  動的時間伸縮法（DTW）を用いて、生成モーションとスライス発生時モーションの顕似度を比較。

---

## ファイル構成

| ファイル/フォルダ | 説明 |
| :----------------- | :--- |
| `train_autoencoder.py` | CAEの学習スクリプト |
| `generate_motion.py` | スイングモーションの生成 |
| `evaluate_motion.py` | DTWによるモーション比較・評価 |
| `models/` | CAEモデル定義ファイル群 |
| `data/` | 入力モーションデータ保管用 |
| `results/` | 生成結果出力用 |

---

## 使用ライブラリ

- Python 3.11
- PyTorch 2.0.1
- NumPy
- Matplotlib

---

## 研究背景

ゴルフスイングにおいて、ボールを真っ直ぐ飛ばすことは必須ですが、
初心者にとっては**スライス**に悩まされることが多いです。

本研究では，
- モーションセンサデータを用いた動作解析
- 畳み込みオートエンコーダによる特徴抽出
- 動的時間伸縮法による比較分析
を通じて、初心者スイング動作の補正を支援する方法を提案しました。

---

## 参考文献

- Daniel Holden, Jun Saito, Thomas Joyce, Taku Komura, "Learning Motion Manifolds with Convolutional Autoencoders", SIGGRAPH Asia 2015
- Daniel Holden, Jun Saito, Taku Komura, "A Deep Learning Framework for Character Motion Synthesis and Editing", ACM Transactions on Graphics
- Ryohei Osawa, Takaaki Ishikawa, Hiroshi Watanabe, "Pitching Motion Matching based on Pose Similarity using Dynamic Time Warping", IEEE GCCE 2020

---

## 著者

東京工科大学 バイオ・情報メディア研究科 コンピュータサイエンス専攻：エムペラド ケイジ ノエル  
指導教員：生野 壮一郎 教授

