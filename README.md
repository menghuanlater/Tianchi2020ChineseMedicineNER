# Tianchi2020ChineseMedicineNER
2020阿里云天池大数据竞赛-中医药命名实体识别挑战赛

* `初赛`: 0.7680, `排名`：35/1682(*single model*)
* `复赛`: 0.7128, `排名`: 20/1682(*single model*)

**核心思路**: 利用Machine Reading Comprehension的思路来解决NER问题(思路来源: ACL2020 A Unified MRC Framework for Named Entity Recognition)

>> **数据处理**: preprocess.py, 构造(Query, Answer, Context)三元组, 对于较长文本, 采用滑动窗口法处理(等于将长文本拆分成多个短文本, 为了尽可能保持上下文连续性, 后面的每个短文本都会有一部分其前序文本的片段, 具体看构造流程)

>> **模型训练**: RoBERTa + Finetune(MRC任务利用BERT解决的最基本的方法), 与参考的那篇论文相比, 我们模型去除了span loss, 因为加了span loss模型都无法训练. 同时我们也测试了focal loss, 但似乎效果并没有提升

>> **个人感悟**：玄学比赛(qaq), 复赛我的小伙伴用了很多方法, 但效果不增反降, 最佳的成绩居然还是我们初赛的baseline模型, 炼丹真奇妙
