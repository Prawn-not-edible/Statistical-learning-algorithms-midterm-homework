# 特征重构与文献依据

本次预处理将特征工程分为三层：缺失指示、异常值温和截断、医学先验交互特征。前两层来自本数据集的缺失率与分布探索；第三层把文献中反复出现的复合临床信号显式化，便于树模型和后续特征重要性分析捕捉。

## 第一层：缺失指示

新增特征：`TnI峰值_missing`、`CK-MB峰值_missing`、`MYO峰值_missing`、`CK同工酶(质量)_missing`、`*钠_missing`、`*钙_missing`、`胆碱脂酶_missing`、`阴离子间隙_missing`、`*尿酸_missing`、`*肌酐(酶法)_missing`。

叙事：检测行为本身携带标签信息。EDA 中 `TnI峰值`、`CK-MB峰值`、`MYO峰值` 在 Class 2 的缺失率显著低于 Class 0，说明疑似心肌梗死患者更可能被反复检测心肌损伤标志物；生化/肾功能检查的缺失模式也在 Class 3 和 Class 5 中表现出分层差异。

## 第二层：长尾指标与异常值处理

对 IQR 异常值比例较高的连续指标，使用训练集 99 分位作为上限进行 winsorize，再把同一上限应用到测试集。该策略保留原始变量的临床量级，同时降低极端异常值对树模型分裂的影响。

我们曾测试额外单变量 `log1p` 特征和临床分级特征，但 LightGBM OOF 消融显示它们没有稳定提升，因此最终不作为默认派生列。需要 `log1p` 压缩的地方保留在医学先验交互公式内部，例如 `cardiac_triple_hit` 和 `cardiorenal_index`。

## 第三层：医学先验交互特征

**cardiorenal_index**：`log1p(入院BNP) / (EGFR + 1)`。BNP 反映心脏压力负荷，eGFR 反映肾功能，二者组合用于捕捉心肾综合征式的复合风险。Tolomeo 等在 *European Journal of Heart Failure* 2024 年研究中证明 BUN/肌酐比值具有独立心衰预后价值；MIMIC-III 心衰研究也指出 BUN/肌酐比值比单独评估尿素氮或肌酐更有预测价值。因此这里构造更贴近本数据字段的 BNP/eGFR 交互指数。需要注意，这是基于心肾交互机制的特征外推，而不是文献中的原公式复现。

**crp_albumin_ratio**：`log1p(超敏感C-反应蛋白) / (白蛋白 + 1)`。CRP/白蛋白比值同时反映炎症和营养状态。Oh 等在 *Frontiers in Cardiovascular Medicine* 2024 年 OPCAB 队列研究中报告 CAR 对 1 年死亡的 AUC 为 0.767，且高 CAR 与更高死亡风险相关。该特征对应本数据中 Class 5 相关的炎症与白蛋白信号。

**bun_creatinine_ratio**：`尿素 / (肌酐 + 0.01)`。Zhu 等在 *Frontiers in Cardiovascular Medicine* 2025 年基于 MIMIC-III 的心衰研究中发现 BUN/creatinine ratio 与心衰死亡率呈非线性 U 型关系，并在较高区间转为风险因子。我们的数据字段为 `*尿素` 和 `*肌酐(酶法)`，单位尺度可能不同于文献中的 BUN/Cr，因此该特征用于模型的相对排序，不直接套用文献阈值。

**cardiac_triple_hit**：`log1p(心肌肌钙蛋白I) + log1p(CK同工酶(质量)) + log1p(MYO峰值)`。急诊疑似 ACS 的 764 例研究显示，肌红蛋白和 cTnI 的联合检测在 9 小时窗口内预测 30 天死亡或心肌梗死的敏感性达到 94%；解释性 EBM 心梗预测研究也显示 troponin 和 CK-MB 是排名靠前的关键预测因子。这里采用加和而非乘法，表达“多个心肌损伤信号累积升高”的临床逻辑。

**metabolic_burden**：`糖尿病年限 * log1p(葡萄糖)`。糖尿病病程反映慢性暴露，即时葡萄糖反映当前代谢状态。糖尿病和血糖水平与冠心病/血管事件风险相关，因此该交互项作为累积代谢损伤的近似指标，主要用于补充 Class 4 较弱的可分性。

**troponin_hr**：`心肌肌钙蛋白I * 心率`。肌钙蛋白 I 反映心肌损伤，心率反映急性应激和血流动力学负担。二者相乘用于捕捉“心肌损伤程度较高且伴随心率应激”的联合状态。该特征来自本项目的 OOF 消融实验：在保留医学先验交互特征的 balanced LightGBM 中，加入 `troponin_hr` 后 macro-F1 和少数类召回优于其他额外交互候选。

**hospitalization_severity**：`max(住院日, 0) * max(压疮评分, 0)`。住院日反映治疗复杂度和病情持续时间，压疮评分反映卧床、活动能力和护理风险。Braden 量表相关研究显示其活动/压疮风险信息具有住院后预后含义。实现中只对派生特征裁剪为非负，不改动原始列。

**cardiac_panel_count / renal_panel_count**：分别统计心肌损伤和肾功能面板中实际完成检测的项目数。这两个变量不是传统生理比值，而是“检查行为强度”特征，与第一层缺失指示互补。

## 报告可用表述

> **cardiorenal_index**：基于 Tolomeo 等（*European Journal of Heart Failure*, 2024）关于 BUN/肌酐比值是心衰独立预后因子的研究，结合本数据中的 BNP 与 eGFR 字段，构造心脏压力与肾功能下降的交互指数。

> **crp_albumin_ratio**：参考 Oh 等（*Frontiers in Cardiovascular Medicine*, 2024）关于 CRP/白蛋白比值对冠脉搭桥患者死亡风险的预测价值，构造炎症-营养复合指标。

> **cardiac_triple_hit**：参考急诊 ACS 研究中肌红蛋白、cTnI、CK-MB 联合检测的预后价值，以及解释性心梗预测模型中 troponin 和 CK-MB 的高重要性，构造心肌损伤三联评分。

> **bun_creatinine_ratio**：参考 MIMIC-III 心衰研究中 BUN/creatinine ratio 与死亡率的非线性关系，构造尿素/肌酐相对比值作为肾功能与容量状态的复合特征。

> **troponin_hr**：结合心肌肌钙蛋白 I 的心肌损伤信号和心率的急性应激信号，构造心肌损伤-血流动力学负担交互项；OOF 消融实验显示该特征可提升少数类召回。

## 参考文献

1. Tolomeo P, Butt JH, Kondo T, et al. Independent prognostic importance of blood urea nitrogen to creatinine ratio in heart failure. *European Journal of Heart Failure*. 2024;26:245-256. DOI: [10.1002/ejhf.3114](https://doi.org/10.1002/ejhf.3114), PubMed: [38124454](https://pubmed.ncbi.nlm.nih.gov/38124454/).
2. Zhu C, Wu L, Xu Y, et al. Predicting mortality in heart failure: BUN/creatinine ratio in MIMIC-III. *Frontiers in Cardiovascular Medicine*. 2025;12:1510317. DOI: [10.3389/fcvm.2025.1510317](https://doi.org/10.3389/fcvm.2025.1510317).
3. Oh AR, Kwon JH, Park J, et al. Preoperative C-reactive protein/albumin ratio and mortality of off-pump coronary artery bypass graft. *Frontiers in Cardiovascular Medicine*. 2024;11:1354816. DOI: [10.3389/fcvm.2024.1354816](https://doi.org/10.3389/fcvm.2024.1354816).
4. The prognostic significance of serial myoglobin, troponin I, and creatine kinase-MB measurements in patients evaluated in the emergency department for acute coronary syndrome. PubMed: [12944886](https://pubmed.ncbi.nlm.nih.gov/12944886/).
5. Interpretable Prediction of Myocardial Infarction Using Explainable Boosting Machines: A Biomarker-Based Machine Learning Approach. *Diagnostics*. 2025;15(17):2219. DOI: [10.3390/diagnostics15172219](https://doi.org/10.3390/diagnostics15172219).
6. Diabetes mellitus, fasting blood glucose concentration, and risk of vascular disease: a collaborative meta-analysis of 102 prospective studies. *The Lancet*. 2010. PMC: [PMC2904878](https://pmc.ncbi.nlm.nih.gov/articles/PMC2904878/).
7. Prognostic Value of Braden Activity Subscale for Mobility Status in Hospitalized Older Adults. PMC: [PMC5551676](https://pmc.ncbi.nlm.nih.gov/articles/PMC5551676/).
