# MinerU vs pdfplumber 文档解析对比

## 一、MinerU 简介

### 1.1 项目背景
MinerU（原名 PDF-Extract-Kit / Magic-PDF）是由上海人工智能实验室（OpenDataLab）开源的多模态文档解析工具，专门针对 PDF 文档的高质量内容提取而设计。

- **GitHub**: https://github.com/opendatalab/MinerU
- **官方网站**: https://mineru.net
- **论文**: MinerU 相关技术基于 PDF-Extract-Kit，核心思路是将文档解析拆分为版面分析、公式检测、表格识别、OCR 等多个子任务，分别用专用模型处理。

### 1.2 核心能力

| 能力 | 说明 |
|------|------|
| **版面分析** | 基于深度学习模型（如 YOLO 系列），识别文本区域、图片区域、表格区域、标题、页眉页脚等 |
| **公式提取** | 检测并提取行内公式和独立公式，输出为 LaTeX 格式 |
| **表格识别** | 识别复杂表格（合并单元格、跨页表格等），输出为 Markdown/HTML |
| **OCR** | 对扫描版 PDF 进行光学字符识别，支持中英文混排 |
| **多栏排版** | 自动识别并还原双栏、三栏等复杂排版结构 |
| **图片提取** | 从 PDF 中提取嵌入式图片，保留原始分辨率 |
| **阅读顺序还原** | 基于版面分析结果，还原文档的逻辑阅读顺序 |

### 1.3 输出格式
- **Markdown**（主要输出，含图片引用、表格、公式 LaTeX）
- **中间 JSON**（版面分析结果，包含每个元素的类型、位置、内容）
- **图片文件夹**（提取的图片文件）

### 1.4 安装与使用

```bash
# 安装
pip install mineru

# 命令行使用
mineru -p input.pdf -o ./output -b vlm-http-client -u http://127.0.0.1:30000

# 参数说明
# -p : 输入 PDF 文件路径
# -o : 输出目录
# -b : 后端模式（vlm-http-client 表示使用 VLM HTTP 客户端进行公式/表格识别）
# -u : VLM 服务地址
```

### 1.5 技术原理

MinerU 的解析流程是一个多阶段流水线：

```
PDF → 页面光栅化 → 版面检测(YOLO) → 区域分类 → 
├─ 文本区域 → OCR / 直接提取
├─ 公式区域 → 公式检测 + LaTeX 识别
├─ 表格区域 → 表格结构识别 → Markdown 表格
├─ 图片区域 → 图片裁剪保存
└─ 标题/列表 → 结构化标记

最终拼接为 Markdown，保留阅读顺序
```

---

## 二、pdfplumber 简介

### 2.1 项目背景
pdfplumber 是一个纯 Python 的 PDF 文本和布局提取库，基于 `pdfminer.six` 构建，提供了更友好的 API。

- **GitHub**: https://github.com/jsvine/pdfplumber
- **定位**: 轻量级、纯文本导向的 PDF 解析

### 2.2 核心能力

| 能力 | 说明 |
|------|------|
| **文本提取** | 逐字符、逐行、逐段落提取文本，保留字符级位置信息 |
| **表格提取** | 基于线条检测提取表格，输出为列表/CSV |
| **页面布局** | 可获取每个字符的 x/y 坐标、字体大小、颜色等 |
| **元数据** | 提取 PDF 元信息（作者、创建时间等） |

### 2.3 不支持的能力

| 能力 | 支持情况 |
|------|----------|
| 公式识别 | ❌ 不支持，只能提取公式文本（如果 PDF 中是文本型公式） |
| 图片提取 | ❌ 无法提取嵌入图片（只能知道图片位置） |
| 扫描版 OCR | ❌ 完全不支持 |
| 复杂版面理解 | ❌ 只能获取字符坐标，需自行编写逻辑还原文档结构 |
| 多栏自动识别 | ❌ 需要手动处理 |
| 输出格式 | 仅文本/表格数据，无 Markdown/结构化输出 |

### 2.4 安装与使用

```bash
# 安装
pip install pdfplumber
```

```python
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    for page in pdf.pages:
        # 提取全文
        text = page.extract_text()
        
        # 提取表格
        tables = page.extract_tables()
        
        # 逐字符提取（含坐标）
        chars = page.chars
        for char in chars:
            print(char["text"], char["x0"], char["top"], char["size"])
```

---

## 三、核心差异对比

### 3.1 功能对比

| 维度 | MinerU | pdfplumber |
|------|--------|------------|
| **文档类型支持** | 扫描版 + 文本型 PDF | 仅文本型 PDF |
| **版面理解** | 深度学习版面分析，自动分类区域 | 仅提供字符坐标，需自行处理 |
| **公式识别** | ✅ 检测 + LaTeX 输出 | ❌ 仅提取文本 |
| **表格识别** | ✅ 复杂表格 → Markdown | ⚠️ 简单表格（基于线条检测） |
| **图片提取** | ✅ 自动裁剪并保存 | ❌ 无法提取图片内容 |
| **OCR** | ✅ 内置 OCR 支持扫描版 | ❌ 不支持 |
| **多栏排版** | ✅ 自动识别并还原 | ❌ 需手动处理 |
| **阅读顺序** | ✅ 基于版面逻辑还原 | ⚠️ 按 PDF 内部流顺序 |
| **输出格式** | Markdown + JSON + 图片 | 纯文本 + 表格数据 |
| **安装依赖** | 需要 GPU / 模型文件 | 纯 Python，零依赖 |
| **速度** | 较慢（1页约 5-30秒） | 快（1页 < 1秒） |
| **硬件要求** | 推荐 GPU，内存 8GB+ | 无特殊要求 |

### 3.2 适用场景

| 场景 | 推荐工具 | 原因 |
|------|----------|------|
| 学术论文解析（含公式/图表） | MinerU | 能识别公式为 LaTeX，提取图表 |
| 扫描版合同/发票解析 | MinerU | 唯一选择（需要 OCR） |
| 简单文本提取 | pdfplumber | 速度快，轻量 |
| 表格数据提取（简单表格） | pdfplumber | 足够用，无需 GPU |
| RAG 知识库构建 | MinerU | Markdown 输出可直接用于 chunk 和向量化 |
| 批量快速处理纯文本文档 | pdfplumber | 无 GPU 依赖，速度极快 |
| 需要图片内容的多模态 RAG | MinerU | 提取图片 + 文本 |

---

## 四、实际效果差异示例

假设解析一份包含以下内容的 PDF 页面：

```
┌─────────────────────────────────────────┐
│  标题: 深度学习中的注意力机制             │
│                                         │
│  [左栏]                    [右栏]        │
│  注意力机制的核心思想...     公式:        │
│                             Attention    │
│                             = softmax(...)│
│                                         │
│  ┌─────────────────────────┐            │
│  │      图片: 注意力热力图    │            │
│  │      [热力图内容]         │            │
│  └─────────────────────────┘            │
│                                         │
│  表1: 不同模型在 ImageNet 上的表现        │
│  ┌──────┬────────┬────────┐             │
│  │ 模型  │ Top-1  │ Top-5  │             │
│  ├──────┼────────┼────────┤             │
│  │ ResNet│ 76.2%  │ 93.1%  │             │
│  │ ViT   │ 84.5%  │ 97.2%  │             │
│  └──────┴────────┴────────┘             │
└─────────────────────────────────────────┘
```

### MinerU 输出（Markdown）

```markdown
# 深度学习中的注意力机制

注意力机制的核心思想是通过...

$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

![注意力热力图](images/attention_heatmap.png)

表1: 不同模型在 ImageNet 上的表现

| 模型 | Top-1 | Top-5 |
|------|-------|-------|
| ResNet | 76.2% | 93.1% |
| ViT | 84.5% | 97.2% |
```

### pdfplumber 输出（纯文本）

```
深度学习中的注意力机制

注意力机制的核心思想...   Attention
                          = softmax(...)
                          
表1: 不同模型在 ImageNet 上的表现
模型  Top-1  Top-5
ResNet  76.2%  93.1%
ViT     84.5%  97.2%
```

**关键差异**:
1. MinerU 能正确还原公式为 LaTeX，pdfplumber 只能提取零散字符
2. MinerU 能提取图片并保存，pdfplumber 完全丢失图片信息
3. MinerU 自动识别双栏并转换为线性 Markdown，pdfplumber 按坐标顺序拼接
4. MinerU 表格输出为标准 Markdown 表格，pdfplumber 表格需后处理

---

## 五、总结

| 一句话总结 | |
|-----------|--|
| **MinerU** | 面向多模态 RAG 的专业文档解析工具，"看得懂" PDF 的内容结构，输出结构化 Markdown |
| **pdfplumber** | 轻量级文本提取工具，"看得见" PDF 的字符和坐标，但不理解内容含义 |

**选择建议**:
- 构建多模态 RAG 知识库（如 week15 项目） → **MinerU**（必需）
- 简单文本提取 / 数据爬取 → **pdfplumber**（足够）
- 两者可互补使用：pdfplumber 做快速预处理，MinerU 做深度解析

---

## 六、参考资料

- MinerU GitHub: https://github.com/opendatalab/MinerU
- MinerU 官方文档: https://mineru.net
- pdfplumber GitHub: https://github.com/jsvine/pdfplumber
- PDF-Extract-Kit 论文: https://arxiv.org/abs/2409.18839
