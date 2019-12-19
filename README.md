### caption zoo

---

**Environment**:

python3.7

pytorch1.0



**Before using this code, please install pycocotools**

```pip install pycocotools```

```pip install git+https://github.com/flauted/coco-caption.git@python23```



**Data process**

dataset :Please download the coco dataset by coco official website.

 in the project root run these:

``` mkdir -p data/coco ```

```cd data/coco```

Then put your coco dataset in data/coco

```mv train2014 val2014/*```

```mv train2014 coco_image2014```

split cocodataset in Karpathy type:

```python karpathy_split.py```



**Run**

eg: run nic model

```cd train/nic```

``` python nic.py --version=baseline```



**eval**

```cd train/nic```

```python nic.py --version=baseline```

**Implemented models**:

training information in train/model_name/readme.md

eg : train/nic/readme.md

* show and tell (NIC) 2015CVPR  [[paper]](https://arxiv.org/abs/1411.4555v2) 
* Show,attend and tell 2015ICLR [[paper]](https://arxiv.org/abs/1502.03044)





