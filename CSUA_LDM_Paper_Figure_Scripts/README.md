# 论文绘图脚本

本目录集中存放 CSUA_LDM 论文实验结果的制图与可视化分析脚本。脚本可从项目根目录或本目录直接运行，生成结果仍保存至项目根目录下原有的论文图像文件夹，不会写入本目录。

| 脚本 | 用途 | 默认输出目录 |
| --- | --- | --- |
| `CSUA_LDM_Generate_Qualitative_Comparison.py` | 生成 CSUA_LDM 与其他超分模型的空间和光谱定性对比图 | `与其他模型对比的定性展示` |
| `CSUA_LDM_Generate_Ablation_Qualitative_Comparison.py` | 生成主版本及各消融版本的空间和光谱定性对比图 | `消融试验的定性展示` |
| `CSUA_LDM_Analyze_Uncertainty_Error_Correlation.py` | 生成单步速度不确定性及完整采样后像素不确定性与实际误差的相关性统计和示例图 | `CSUA_LDM_Uncertainty_Error_Correlation_Results` |

以上脚本仅负责论文制图或支撑图像分析。训练、推理、定量评估与参数搜索脚本继续保留在原有目录中。
