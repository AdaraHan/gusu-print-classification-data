# 姑苏版画分类数据表

`gusu_classification.sqlite` 是 892 张分类图像和原五折 OOF 结果的查询副本。逐图 CSV 是保存分数原始精度的权威文件；数据库把常用关联做成外键和视图。图像本体保存在 `images/`，数据库中的 `image_path` 为相对于本目录的路径。没有原始文件名与公开编号的映射。

| 表或视图 | 粒度 | 用途 |
|---|---|---|
| `images` | 每张图一行，共 892 行 | 公开编号、脱敏图像路径、分类标签、原验证折、SHA-256；176 条私人藏家记录的 `collection` 为“私人藏家” |
| `predictions` | 模型×图像，共 2676 行 | 三模型逐图类别预测与概率 |
| `formal_candidates` | 候选图像一行，共 568 行 | 形式分析候选组与审核状态；`include_in_analysis=0`，不代表最终分析集 |
| `prediction_audit` | 模型×图像 | 联结真值、原折、预测与是否正确 |
| `misclassifications` | 误判一行 | 46／71／86 条误判的数据库视图 |

示例查询：

```sql
SELECT class, COUNT(*) FROM images GROUP BY class;
SELECT validation_fold, COUNT(*) FROM images GROUP BY validation_fold ORDER BY validation_fold;
SELECT model, COUNT(*) FROM misclassifications GROUP BY model;
SELECT sample_id, image_path, true_class, predicted_class, prob_gusu
FROM prediction_audit WHERE model='effnetv2_s' AND is_correct=0;
SELECT COUNT(*), SUM(include_in_analysis) FROM formal_candidates;
```

使用 Python 标准库即可打开：

```python
import sqlite3
con = sqlite3.connect("gusu_classification.sqlite")
print(con.execute("PRAGMA integrity_check").fetchone())
print(con.execute("SELECT model, COUNT(*) FROM misclassifications GROUP BY model").fetchall())
```

`features/dinov2_v4_public.npz` 保存原 `X` 浮点矩阵和原 `y` 编码，另外保存按原特征行顺序排列的公开 `sample_id`、`image_path`。只替换了含原文件名的路径数组；`X` 和 `y` 逐元素不变，可用 `numpy.load(..., allow_pickle=False)` 读取。其五折归属以数据库 `images.validation_fold` 或 `manifest.csv` 为准。
