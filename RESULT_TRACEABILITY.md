# 论文结果—输入数据—处理代码—输出文件

论文核对基线是研究团队指定的 2026-09-30 Word 稿。下面仅说明当前文件可证明的关系；不把后来的候选实验冒充已冻结的最终形式分析。对应原始论文文件及含身份信息的私有审计记录未收入本目录。

| 论文结果或图表 | 输入数据 | 处理代码、配置 | 输出文件 | 可验证程度 |
|---|---|---|---|---|
| 3031→896→892 数据流程、数据缩略图 | 原始 3031 张集合的计数；历史 896 张统计；本目录 892 张 | `code/source/Step1_run_dataset_stats.py`、`code/source/STEP7_make_all_contact_sheet.py` | `selection_and_dedup_evidence.json`、`manifest.csv`、`images/` | 3031、892 数量可核；历史 896 目录缺失，逐图遴选无法复原。3031 张原图未打包。 |
| 表1—3：训练、预处理及分类器参数 | `images/`、`features/dinov2_v4_public.npz` | `code/source/` 中历史脚本；`code/config/effnet_historical_run_public.json` | 原五折、三模型 OOF 表 | 参数与运行记录吻合。历史脚本按原文件名排序，公开图像改名后重跑须使用下述 manifest 适配代码。 |
| 表4：三模型五折指标 | 892 张原折逐图预测及概率 | `code/verify_oof.py` | `predictions/*_oof_predictions.csv` | 可直接重算。LogReg 本版 AUC 为 0.9612±0.0091；另有历史分数版为 0.9614±0.0091，未混入本目录。 |
| 表5—6：总体 OOF 指标、混淆矩阵及误判分布 | 同上 | `code/verify_oof.py` | `predictions/*_misclassified.csv`，以及数据库 `misclassifications` 视图 | 可直接重算；误判数 46／71／86。 |
| 18 张漏报、28 张误报插图 | EfficientNet OOF 表和图像 | `code/make_public_contact_sheets.py` 按公开编号重新排版 | `figures/effnet_false_negatives_public.png`、`figures/effnet_false_positives_public.png` | 新图逐图数量与原论文一致，标签只含公开编号、折号和类别；原论文拼图不收入本目录。 |
| 表7与色彩图 | 论文给出的六个色彩均值和标准差 | `code/formal_candidate/generate_color_summary_chart.py` 仅把汇总数绘图 | `formal_candidate_evidence/figures/color_reported_summary_not_recomputed.png` | 缺少逐图 HSL 特征和最终样本依据，不能独立重算。此图是均值±标准差柱状图，不是颜色直方图。 |
| 表8：Sobel/Canny；边缘质控图 | `formal_candidate_subset.csv` 的 568 张候选图 | `code/formal_candidate/run_candidate_line_analysis.py`、`add_direction_concentration_to_table9.py`、`edge_detection_pipeline.py` 及参数文件 | `formal_candidate_evidence/line_*`、`paper_table8_candidate.csv`、质控图 | 候选样本指标可核算；正式作品/版次清单未冻结。 |
| 表9：GLCM 四属性 | 同一 568 张候选图 | `code/formal_candidate/run_candidate_glcm_analysis.py`、`glcm_parameters.json` | `formal_candidate_evidence/glcm_metrics_per_image.csv`、`paper_table9_candidate.csv` | 与论文四舍五入值一致；仍为候选分析。 |
| Gram 距离和分布图 | 同一 568 张候选图及卷积特征 | `code/formal_candidate/run_candidate_gram_analysis.py`、`gram_parameters.json`、随附 SqueezeNet 权重 | `formal_candidate_evidence/gram_embeddings_public.npz`、逐对距离、组汇总和图 | 候选样本可核算；图像对不独立，正式分组未冻结。 |
| 470 张检出、854 个消失点候选及示例图 | 同一 568 张候选图 | `code/formal_candidate/run_candidate_vanishing_point_analysis.py`、参数文件 | `formal_candidate_evidence/vanishing_point_*` 和示例图 | 自动候选数可核算；尚无已确认的真实消失点。 |
| 表10：形状指标 | 同一 568 张候选图 | `code/formal_candidate/run_candidate_shape_analysis.py`、`shape_parameters.json` | `formal_candidate_evidence/shape_metrics_per_image.csv`、`paper_table10_candidate.csv` | 与论文四舍五入值一致；仍为候选分析。 |

## 能否重新运行

- **可直接核算**：表4—6、原五折、每张图的预测与误判；候选实验的表8—10、Gram 距离与自动消失点数量可从随附逐图或逐对特征核算。
- **可作为新实验重新运行**：DINOv2 特征分类器可用 `code/rerun_classifiers.py` 和固定原折重跑；EfficientNet 可用 `code/rerun_effnet_manifest.py` 把历史训练脚本接到公开编号和原折。重新训练得到的输出应另存，不得替代已保存的论文 OOF 分数。候选形式分析的历史代码和参数均列于 `code/formal_candidate/`，其默认路径仍反映原项目结构，需要按公开清单适配后才可在此目录执行。
- **仍有缺口**：3031→896 的完整逐图筛选记录、历史 896 图像目录；表7色彩逐图数据和方法链；形式美学最终人工确认的作品/版次、组别及去重清单；历史训练环境锁定；两版 LogReg 概率差异的根因。
