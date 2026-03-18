## 作业1

### BERT分类 vs NER 对比

```
                  文本分类                    实体识别(NER)
输出层           [CLS] → 全连接              每个token → 全连接
输出维度         (batch, num_classes)        (batch, seq_len, num_labels)
Loss            交叉熵(句子级)               交叉熵(token级均值)
特殊处理         无                          ignore_index=-100
对应类           BertForSequenceClassification   BertForTokenClassification
```

### 多任务Loss不平衡解决方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 静态加权 | 简单直观 | 需要手动调参 | 有先验知识时 |
| 动态不确定性权重 | 自动学习，无需调参 | 实现复杂，引入额外参数 | 任务差异较大时 |
| GradNorm | 梯度层面平衡 | 计算开销大 | 对训练稳定性要求高 |
| 分阶段训练 | 每阶段专注，稳定 | 训练时间长 | 任务差异极大时 |
| 损失归一化 | 简单有效，最实用 | 依赖初始loss稳定性 | **通用首选** ✓ |

---