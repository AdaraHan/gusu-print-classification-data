#!/usr/bin/env python3
"""Visualize the six mean±SD values printed in the current manuscript Table 8.

This script does not read image pixels and does not produce a hue/pixel
histogram. It only creates a grouped comparison figure from existing summary
numbers supplied in the manuscript screenshot.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


OUT = Path(__file__).resolve().parent
FONT_REGULAR = FontProperties(fname="/System/Library/Fonts/Supplemental/Songti.ttc", size=12)
FONT_BOLD = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc", size=12)

groups = ["姑苏版画", "桃花坞版画", "杨柳青年画"]
saturation_mean = np.array([0.263, 0.205, 0.306])
saturation_sd = np.array([0.121, 0.039, 0.102])
lightness_mean = np.array([0.565, 0.587, 0.585])
lightness_sd = np.array([0.096, 0.032, 0.077])

colors = ["#3977A8", "#C9952E", "#B85C74"]
hatches = ["///", "\\\\", "..."]

fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.9), sharey=True)
fig.patch.set_facecolor("white")
x = np.arange(len(groups))

for ax, means, errors, panel_title in [
    (axes[0], saturation_mean, saturation_sd, "饱和度"),
    (axes[1], lightness_mean, lightness_sd, "明度"),
]:
    bars = ax.bar(
        x,
        means,
        yerr=errors,
        width=0.62,
        color=colors,
        edgecolor="#28323C",
        linewidth=0.9,
        capsize=5,
        error_kw={"elinewidth": 1.2, "ecolor": "#28323C", "capthick": 1.2},
        zorder=3,
    )
    for bar, hatch, mean, sd in zip(bars, hatches, means, errors):
        bar.set_hatch(hatch)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + sd + 0.025,
            f"{mean:.3f} ± {sd:.3f}",
            ha="center",
            va="bottom",
            color="#202832",
            fontproperties=FONT_REGULAR,
            fontsize=10.5,
        )
    ax.set_title(panel_title, fontproperties=FONT_BOLD, fontsize=15, pad=12, color="#202832")
    ax.set_xticks(x, groups, fontproperties=FONT_REGULAR, fontsize=11)
    ax.set_ylim(0, 0.75)
    ax.set_yticks(np.arange(0, 0.76, 0.1))
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#66717E")
    ax.tick_params(axis="y", colors="#4D5966")
    ax.tick_params(axis="x", length=0, pad=8)

axes[0].set_ylabel("归一化指标值", fontproperties=FONT_REGULAR, fontsize=12, color="#202832")

fig.suptitle(
    "三类版画图像的饱和度与明度（均值 ± 标准差）",
    fontproperties=FONT_BOLD,
    fontsize=18,
    y=0.98,
    color="#17212B",
)
fig.text(
    0.5,
    0.925,
    "数据来自论文当前表8；仅可视化既有汇总值，非像素级颜色直方图",
    ha="center",
    fontproperties=FONT_REGULAR,
    fontsize=11,
    color="#596574",
)
fig.text(
    0.5,
    0.025,
    "注：误差线表示表中标准差；正式复现实验结果应在样本manifest冻结后重新计算并替换。",
    ha="center",
    fontproperties=FONT_REGULAR,
    fontsize=10,
    color="#66717E",
)
fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.16, wspace=0.12)

fig.savefig(OUT / "table8_saturation_lightness_mean_sd.png", dpi=300, facecolor="white")
fig.savefig(OUT / "table8_saturation_lightness_mean_sd.svg", facecolor="white")
plt.close(fig)
