# 图像分类训练脚手架（开箱即用）

这个仓库已经帮你把**模型训练的完整骨架**搭好，你只需要：

1. 安装依赖
2. 把图片数据放进指定目录
3. 执行训练命令

> 默认任务：图像分类（多类别）

---

## 1) 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2) 放入数据（你只需要做这一步）

按下面目录组织你的图片（类别名就是文件夹名）：

```text
data/
  train/
    cat/
      001.jpg
      002.jpg
    dog/
      a.png
  val/
    cat/
    dog/
  test/
    cat/
    dog/
```

- `train/`：训练集
- `val/`：验证集（用于挑选最佳模型）
- `test/`：测试集（最终评估）

可先检查数据是否识别正常：

```bash
python scripts/check_data.py
```

---

## 3) 一键训练

```bash
PYTHONPATH=src python src/train.py --config configs/train.yaml
```

训练完成后会在 `checkpoints/` 输出：

- `best.pt`：最佳权重
- `history.json`：每个 epoch 的训练/验证指标
- `metrics.json`：最终结果（best val + test）

---

## 可调配置

配置文件：`configs/train.yaml`

你最常改的参数：

- `data.image_size`：输入分辨率
- `data.batch_size`：批大小
- `model.name`：`resnet18/resnet34/resnet50`
- `model.pretrained`：是否使用 ImageNet 预训练
- `train.epochs`：训练轮数
- `train.lr`：学习率
- `train.mixed_precision`：是否开启混合精度（GPU 推荐）

---

## 说明

- 当前脚手架默认是**监督学习的图像分类**。
- 如果你后续要做检测/分割/多标签，我可以在此基础上给你扩展同风格模板。
