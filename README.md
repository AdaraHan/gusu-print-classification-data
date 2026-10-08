# 姑苏版画分类：编审核验数据与代码（本地待发布）

**状态：本地待发布。** 公开图像和结构化数据已完成自动脱敏及关联核验；上传前须使用 `RELEASE_FILE_LIST.txt` 白名单，并按 `PUBLICATION_CHECK.md` 复核。

本候选含 892 张分类图像（姑苏 459、非姑苏 433）及三种模型的逐图折外预测、误判表。`manifest.csv` 保存原验证折号；图像字节未改变。已确认属于私人藏家身份的来源标记统一写为“私人藏家”，馆藏字段在对应的 176 条记录中亦如此填写。其他来源身份、图像可见文字和论文图注仍待复核。

`gusu_classification.sqlite` 是 CSV 的查询数据库，包含 892 张图像、原五折、2676 条三模型逐图预测及 568 条形式分析**候选**记录；数据库不把候选图像当作最终形式分析集。`features/dinov2_v4_public.npz` 保存与原始值相同的 892×768 特征矩阵及公开编号。用法见 `DATABASE_SCHEMA.md`。

568 张形式分析候选图像是这 892 张中的子集；`formal_candidate_subset.csv` 只记录成员编号、候选分组和待审核状态，不另行复制图像，也不将候选状态表述为人工核定。3031 张原始收集图像及历史 896 张遴选图像未纳入此目录。

`code/verify_oof.py` 可在不重新训练的情况下复核逐图引用、原验证折、五折指标和合并 OOF 指标。`code/source/` 保存版本对应的历史训练与导出脚本；`code/rerun_classifiers.py`、`code/rerun_effnet_manifest.py` 使用公开编号和保存的原折作新实验入口，**不会替换论文原结果**。EfficientNet 入口已通过只读预检，未重新训练。候选形式分析特征与图表见 `formal_candidate_evidence/`，其历史代码在 `code/formal_candidate/`。

`figures/` 中两张误判拼图按公开编号重新排版，未裁剪作品图像，避免沿用论文原拼图的来源文件名标签；生成代码为 `code/make_public_contact_sheets.py`。

论文各结果的证据链和缺口见 `RESULT_TRACEABILITY.md`。历史 896 张的筛选、去重证据见 `selection_and_dedup_evidence.json`；3031 张原始收集图像未整批纳入。论文原 DOCX 含内部批注、个人元数据及未脱敏误判拼图标签，不在本目录内。原名映射、含姓名审计明细、日志和旧压缩包均不在公开候选中。
