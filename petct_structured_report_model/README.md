# PET/CT 多解剖区域结构化报告模型（Baseline）

本项目用于训练 **PET/CT 多区域结构化报告模型**：输入 CT（`*_0000.nii.gz`）+ PET/SUV（`*_0001.nii.gz`），输出每个解剖区域的 `pet_status` 与 `ct_status` 分类结果，并可选回归 `suv`。

## 重要边界说明（必须阅读）

- 本项目使用的 JSON 标注是 **区域级结构化标签**，不是 voxel-level segmentation mask。
- 因此本项目**不能直接训练 nnU-Net 分割模型**。
- 本项目也**不能直接判断肿瘤良恶性或病理类型**，因为未提供病理标签。
- 若要训练分割模型，必须补充 voxel-level mask 标注（如 NIfTI mask）。
- 若要训练良恶性分类，必须补充病理或随访终点标签。

## 目录结构

```text
petct_structured_report_model/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml
├── scripts/
│   ├── inspect_dataset.py
│   ├── build_manifest.py
│   ├── flatten_json_labels.py
│   ├── train_region_classifier.py
│   ├── evaluate_region_classifier.py
│   └── predict_structured_report.py
├── src/
│   ├── dataset.py
│   ├── nifti_utils.py
│   ├── label_utils.py
│   ├── model.py
│   ├── metrics.py
│   └── visualization.py
└── tests/
    ├── test_label_parser.py
    ├── test_manifest.py
    └── test_metrics.py
```

## 快速开始

```bash
pip install -r requirements.txt

python scripts/inspect_dataset.py --data-root /path/to/data --output outputs/manifest.csv
python scripts/flatten_json_labels.py --manifest outputs/manifest.csv --output outputs/labels_flat.csv

python scripts/train_region_classifier.py \
  --config configs/config.yaml \
  --manifest outputs/manifest.csv \
  --labels outputs/labels_flat.csv \
  --output-dir outputs

python scripts/evaluate_region_classifier.py \
  --config configs/config.yaml \
  --manifest outputs/manifest.csv \
  --labels outputs/labels_flat.csv \
  --model outputs/best_model.pth \
  --output-dir outputs/eval

python scripts/predict_structured_report.py \
  --config configs/config.yaml \
  --model outputs/best_model.pth \
  --ct /path/to/CASE_0000.nii.gz \
  --pet /path/to/CASE_0001.nii.gz \
  --output outputs/CASE_pred.json
```

## 数据要求

每个病例建议有三件文件：
- `CASE_0000.nii.gz`：CT
- `CASE_0001.nii.gz`：PET/SUV
- `CASE.json`：结构化标注

`inspect_dataset.py` 会自动生成 `manifest.csv`：
- `case_id, ct_path, pet_path, json_path, has_ct, has_pet, has_json`

`flatten_json_labels.py` 会展开 JSON 至 `labels_flat.csv`：
- `case_id, region, pet_status, suv, ct_status, ct_description`

## Baseline 模型设计

- 2 通道 3D 输入（CT + PET）
- 3D ResNet encoder
- region embedding
- 多任务头：
  - 每 region 的 `pet_status` 分类
  - 每 region 的 `ct_status` 分类
  - 可选 `suv` 回归头

## 测试

```bash
pytest -q
```

## 跨平台说明

- 路径处理使用 `pathlib`，兼容 Windows / Linux / WSL2。
- 命令行参数使用 `argparse`，无硬编码数据路径。
