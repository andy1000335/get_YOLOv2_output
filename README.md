# YOLOv2

## YOLOv2 架構
![](image/yolov2Architecture.jpg)  
- YOLOv2 downscaling 5 次，因此輸出 cell 數為 13x13 (416/2^5=13)，每一個 cell 大小為 32x32
### 輸入
- input size：416x416x3
### 輸出
每一個 bounding box 包含 (x, y, w, h, confidence, classes) x AnchorNum
- x, y：bounding box 的中心座標
- w, h：bounding box 的寬與長
- confidence：bounding box 信心指數
- classes：各個類別的準確率
- AnchorNum：anchor 數量，YOLOv2 為 5  

因此 output 為 13 x 13 x [(5+classes)x5]

## Bounding Box
<img src="image/detail.jpg" width="600">  
此處假設共有 20 個 class，因此輸出為 13x13x125  
每一個 cell 可以取得 5 個不同大小的 bounding box，若其 confidence 乘上 class 的準確率(最大的那一個)大於 threshold 即為我們要的 bounding box    

註：YOLOv2 的 Bounding Box 參數是在縮小 1/32 後的 feature map 上對應的大小，因此最後必須承上 32 取得實際的大小

## 輸入 Resize
YOLOv2 輸入大小為 416x416，因此圖片輸入時必須先 resize 為 416x416  
為防止圖片變形，必須將長寬等比例縮放，並將不足的部分填入(128, 128, 128)  
<img src="image/resize.jpg" width="300">  

- 以```min(w/img_w, h/img_h)```比例將原圖縮放，剩下的區域(灰色)填入 (128, 128, 128)

## Non-Maximum Suppression (NMS)
到目前為止得到的 bounding box 不一定都是我們想要的，例如同一個物件可能被同時框兩次以上  
NMS 主要是計算兩個 bounding box 間的 IoU，若 IoU 大於某個 threshold 代表兩個 bounding box 交疊  
1. 建立一空集合 output = {}
2. 將 bounding box 依信心指數排序
3. 信心指數最高的 bounding box 即為我們要的，將其加入 output 集合內
4. 將其他的 bounding box 與剛剛選出來的 bounding box 計算 IoU，若大於某個 threshold 則將其信心指數設為 0
5. 剩下的 bounding box 再取出信心指數最高的加入 output 集合內並重複步驟 4
6. 持續進行以上步驟直至沒有信心指數大於 0 的 bounding box  

最後 output 集合內的 bounding box 即為我們要的輸出

![](image/NMS.jpg)

## Demo
### input
![](image/testImage.jpg)
### output
![](image/demo.png)
