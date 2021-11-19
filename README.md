# Occlusion-Robust-Face-Recognition-Based-on-Mask-Learning-with-Pairwise-Differential-Siamese-Network

paper link: [https://arxiv.org/pdf/1908.06290.pdf]

利用 mnist 手寫數字辨識實現本篇論文，本 project 只做 PDSN 的部分，並沒有做整體架構的 training。

## load_data.py 
原 paper 會準備各式各樣的 mask 蓋在特定區域以及其四周，這裡只是先做蓋在 first block (一張影像切割成 5*5 patch 的左上角區域 )
而原 paper 會根據不同的 patch 做不同的 Z_different 這裡只嘗試做第一塊

## model.py 
參考原paper的論文，並稍微調整一些參數設計

## training.py 
主要訓練PDSN的部分，並不是"整體架構"
