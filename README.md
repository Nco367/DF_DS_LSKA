## 方法

* 网络模型整体结构代码位于
```	
./lib/network.py
```


## 数据集
* 1、LineMOD数据集
* [LineMOD](http://campar.in.tum.de/Main/StefanHinterstoisser): Download the [preprocessed LineMOD dataset](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7) (including the testing results outputted by the trained vanilla SegNet used for evaluation).

* 2、发黑氧化工业零件数据集
* 链接：[https://pan.baidu.com/s/1KW2PGaKfC8BidK62rP6tHw?pwd=8888](https://pan.baidu.com/s/1wAqbQHWoXMgoyabb5WfdhA?pwd=8888) 提取码: 8888 


## 训练
* LineMOD Dataset:
	After you have downloaded and unzipped the Linemod_preprocessed.zip, please run:
```	
./experiments/scripts/train_linemod1.sh
```
* 发黑氧化工业零件数据集:
	After you have downloaded, please run:
```	
./experiments/scripts/train_linemod3.sh
```

## 评估

### Evaluation on LineMOD Dataset
Just run:
```
./experiments/scripts/eval_linemod1.sh
```

### 评估发黑氧化工业零件数据集
Just run:
```
python ./tools/eval_linemod3.py
```
