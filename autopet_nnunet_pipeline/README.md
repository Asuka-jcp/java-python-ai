# autopet_nnunet_pipeline

一个完整的 AutoPET PET/CT 到 **nnU-Net v2** 的 3D 肿瘤病灶分割训练项目。

## 项目目标

本项目用于训练 **PET/CT 肿瘤病灶分割模型**，并输出疑似病灶的二值 mask。

- 输入通道 0：`CTres.nii.gz`
- 输入通道 1：`SUV.nii.gz`
- 标签：`SEG.nii.gz`
- 模型：`nnU-Net v2`
- 数据集：`Dataset501_AutoPET`（ID `501`）

> 重要说明：
> - 该项目训练的是 **病灶分割模型**，输出是疑似肿瘤病灶 mask。
> - **不能直接判断肿瘤良恶性**。
> - **不能判断肺癌病理类型**（如腺癌、鳞癌、小细胞癌等）。

---

## 目录结构

```text
autopet_nnunet_pipeline/
├── configs/
│   └── config.yaml
├── scripts/
│   ├── inspect_autopet_dataset.py
│   ├── check_nifti_geometry.py
│   ├── prepare_nnunet_dataset.py
│   ├── run_preprocess.sh
│   ├── run_train.sh
│   ├── run_predict.sh
│   └── evaluate_segmentation.py
├── src/
│   ├── nifti_utils.py
│   └── metrics.py
├── tests/
│   ├── test_metrics.py
│   └── test_prepare_dataset.py
└── requirements.txt
```

---

## 步骤 1：安装环境

建议使用 Python 3.10+。

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## 步骤 2：修改 `configs/config.yaml`

编辑以下关键路径：

- `raw_data_root`: 原始 AutoPET 数据根目录
- `nnunet_raw`: nnU-Net 原始数据目录
- `nnunet_preprocessed`: 预处理目录
- `nnunet_results`: 训练结果目录

默认文件名和通道定义：

- `ct_filename: CTres.nii.gz`
- `suv_filename: SUV.nii.gz`
- `seg_filename: SEG.nii.gz`
- `channel_0_name: CT`
- `channel_1_name: SUV`
- `label_name: lesion`

---

## 步骤 3：扫描数据

递归扫描 `raw_data_root`，找到同时包含 CTres/SUV/SEG 的病例目录，输出清单：

```bash
python scripts/inspect_autopet_dataset.py \
  --config configs/config.yaml \
  --output manifest.csv \
  --missing-output missing_files.csv
```

输出：
- `manifest.csv`
- `missing_files.csv`（可选）

---

## 步骤 4：检查 NIfTI 几何一致性

检查 CTres、SUV、SEG 三者的：
- shape
- spacing
- affine
- SEG 是否为二值（0/1）

```bash
python scripts/check_nifti_geometry.py \
  --manifest manifest.csv \
  --output geometry_report.csv
```

---

## 步骤 5：转换为 nnU-Net v2 格式

将数据复制为 nnU-Net 规范命名：
- `imagesTr/AutoPET_XXXX_0000.nii.gz`（CT）
- `imagesTr/AutoPET_XXXX_0001.nii.gz`（SUV）
- `labelsTr/AutoPET_XXXX.nii.gz`（SEG）

并自动生成 `dataset.json`。

```bash
python scripts/prepare_nnunet_dataset.py \
  --config configs/config.yaml \
  --manifest manifest.csv
```

---

## 步骤 6：运行 nnU-Net 预处理

```bash
bash scripts/run_preprocess.sh configs/config.yaml
```

等价核心命令：
```bash
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
```

---

## 步骤 7：训练模型

默认训练 `fold=0`、`config=3d_fullres`：

```bash
bash scripts/run_train.sh
```

指定参数：

```bash
bash scripts/run_train.sh 0 3d_fullres 501
```

---

## 步骤 8：推理

```bash
bash scripts/run_predict.sh <input_dir> <output_dir> [fold] [config] [dataset_id]
```

示例：

```bash
bash scripts/run_predict.sh ./infer_images ./infer_preds 0 3d_fullres 501
```

核心命令：
```bash
nnUNetv2_predict -i <input_dir> -o <output_dir> -d 501 -c 3d_fullres -f 0
```

---

## 步骤 9：评估分割结果

支持逐病例计算：
- Dice
- False Positive Volume
- False Negative Volume

```bash
python scripts/evaluate_segmentation.py \
  --prediction_dir <prediction_dir> \
  --label_dir <label_dir> \
  --output_dir <metrics_output_dir>
```

输出：
- `per_case_metrics.csv`
- `summary_metrics.json`

---

## 跨平台说明（Windows / Linux / WSL2）

- Python 脚本全部使用 `pathlib` 处理路径。
- 不依赖硬编码绝对路径。
- Shell 脚本适合 Linux/WSL2；Windows 建议在 WSL2 或 Git Bash 下运行。
