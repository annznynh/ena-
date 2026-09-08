import streamlit as st

from zhiji.ui import setup

setup("方法与限制")
st.markdown("""
本原型使用 ASSISTments 2012–2013 年公开答题日志。数据年代、平台和学习情境与当前学校可能存在差异。

**模型预测**学生下一次作答正确的概率。答错风险为 1 减去正确概率；概率不是确定结论。模型按学生划分训练、验证和测试集，历史特征只读取预测时刻以前的数据。

**近期事实**描述已经发生的答错、提示和重复尝试。**模型贡献**描述哪些特征推动估计上升或下降。两者都不证明教学原因。

**ENA 群体网络**使用较早 70% 的交互构建网络，用较晚 30% 的表现分组。群体平均共现不能用于给个体贴标签。四个情感字段是其他模型预测的置信度，不是真实情绪测量。

教师应结合课堂观察、题目难度、学生反馈和实际学习条件复核，再决定讲解、补充练习或继续观察。系统不自动通知学生或家长，也不自动作出教学决定。

本页面展示的是匿名测试样本，而非真实班级；没有证据证明本原型的预警能改善学习结果。尚未开展真实教师可用性访谈。

[ASSISTments 官方数据说明](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect) · [数据副本入口](https://www.kaggle.com/datasets/nicolaswattiez/skillbuilder-data-2009-2010)
""")
