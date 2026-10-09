# 姑苏版画分类公开数据集

本仓库提供姑苏版画二分类研究的图像、标签、原五折预测、SQLite 数据库、代码和数据说明。文件清单及校验值见 `RELEASE_FILE_LIST.txt`，数据质量和使用边界见 `DATASET_NOTES.md`。

本数据集含 892 张分类图像（姑苏 459、非姑苏 433）及三种模型的逐图折外预测、误判表。`manifest.csv` 保存原验证折号；图像字节未改变。属于私人藏家身份的来源标记统一写为“私人藏家”，馆藏字段在对应的 176 条记录中亦如此填写。图像可见文字的小字与水印仍有人工复核空间。

`gusu_classification.sqlite` 是 CSV 的查询数据库，包含 892 张图像、原五折、2676 条三模型逐图预测及 568 条形式分析**候选**记录；数据库不把候选图像当作最终形式分析集。`features/dinov2_v4_public.npz` 保存与原始值相同的 892×768 特征矩阵及公开编号。用法见 `DATABASE_SCHEMA.md`。

568 张形式分析候选图像是这 892 张中的子集；`formal_candidate_subset.csv` 只记录成员编号、候选分组和待审核状态，不另行复制图像，也不将候选状态表述为人工核定。3031 张原始收集图像及历史 896 张遴选图像未纳入此目录。

`code/verify_oof.py` 可在不重新训练的情况下计算逐图关联、五折指标和合并 OOF 指标。`code/source/` 保存生成这些数据所用版本的历史训练与导出脚本；`code/rerun_classifiers.py`、`code/rerun_effnet_manifest.py` 可用公开编号和保存的原折运行新实验。EfficientNet 入口已通过只读预检。形式分析候选特征与图表见 `formal_candidate_evidence/`，相关代码在 `code/formal_candidate/`。

`figures/` 中两张误判拼图按公开编号排版，未裁剪作品图像；生成代码为 `code/make_public_contact_sheets.py`。

历史数据规模、筛选与去重的现存记录见 `selection_and_dedup_evidence.json`。
