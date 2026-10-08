# 上传前核验记录

本目录是供编审核验的本地数据包。所有文件由 `RELEASE_FILE_LIST.txt` 明确列出，不从项目根目录直接打包。

- 892 张分类图像与内部源文件逐张 SHA-256 一致；公开文件名只替换已确认的私人藏家身份片段，`manifest.csv` 中 176 条 `collection=私人藏家`。
- 892 张标签、原五折和三模型 2676 条预测关联可从 SQLite 与逐图 CSV 交叉核对；原分数未因脱敏而改变。
- 568 张形式分析图像全部属于上述 892 张，仍是候选；没有已冻结的最终形式分析集。
- 所有 892 张图像做过本地 OCR 定位；176 张私人藏家来源图像又以更高分辨率复查。未检出已确认藏家的完整姓名或简称。OCR 不能证明所有题签和水印均无姓名，作品图像没有裁剪、涂抹、重编码或清理元数据。
- 论文八张嵌入图均已检查。论文原件含 11 处误判图来源标签、内部批注及个人邮箱元数据，未收入本包；`figures/` 已依据原 18／28 张误判样本重排安全标签。论文图注将均值±标准差柱状图称作“颜色直方图”，建议在论文公开副本中更正。
- 论文表7色彩逐图特征、历史 896 图像完整清单、形式美学最终作品集和历史完整环境锁定仍缺失，已在 `RESULT_TRACEABILITY.md` 标明。

本地验证：

```bash
python code/verify_oof.py
python code/rerun_effnet_manifest.py --preflight
python -c 'import sqlite3; c=sqlite3.connect("gusu_classification.sqlite"); print(c.execute("PRAGMA integrity_check").fetchone())'
```

执行重新训练须另选新输出路径，并把它作为新实验处理。图像文字的 OCR 阴性结果有识别限度；正式公开前建议研究团队查看带题签、水印和个人来源标签的高清原图及拟发布论文副本。
