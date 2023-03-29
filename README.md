---

---

# visualize evaluation for YOLOX

[原readme](README_origin.md)

[知乎](https://zhuanlan.zhihu.com/p/499759736)|[bilibili](https://www.bilibili.com/read/cv16169153)

## 1

画混淆矩阵，各种曲线，支持coco和voc数据集

效果预览：

<img src="http://m.qpic.cn/psc?/V53B8TyR2Noekm3rNUZH48QLmk39wAow/ruAMsa53pVQWN7FLK88i5j8Il3SvYwJAF.zgHrRw4riV2JHSNEQWoLdRQKdrgjBSnbjW8WF3y.WBcHxpMBgdsnypSW1FFgWCAKm7R5gUx8M!/b&bo=oAU4BAAAAAADB7s" alt="confusion_matrix" style="zoom: 25%;" />

<img src="http://m.qpic.cn/psc?/V53B8TyR2Noekm3rNUZH48QLmk39wAow/ruAMsa53pVQWN7FLK88i5j8Il3SvYwJAF.zgHrRw4rjv.a6m2PeCqOmkiJDrCl*ylh3KV.dqdR.h21qwHMLDgn2sngn3G5*GByjjY*HUDH0!/b&bo=VAY4BAAAAAADRww!&rf=viewer_4" alt="PR_curve" style="zoom: 25%;" />

提高配置文件中的`plot_sample_rate`可以使曲线更平滑，但从采样率到pr-pair的映射显然不是线性的所以效果有限。

```
python  tools/eval.py -f  yolox_s.py -c yolox_s.pth -b 16 -d 1
```

## 2

正则匹配log保存map，loss到csv

```
log2csv.py
```

