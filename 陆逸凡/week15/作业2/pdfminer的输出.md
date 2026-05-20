MinerU2.5: A Decoupled Vision-Language Model for Efficient
High-Resolution Document Parsing

Junbo Niu1,2∗, Zheng Liu1,2∗, Zhuangcheng Gu1∗, Bin Wang1∗‡, Linke Ouyang1∗
Zhiyuan Zhao1∗, Tao Chu1∗, Tianyao He1∗, Fan Wu1∗, Qintong Zhang1,2∗, Zhenjiang Jin1∗
Guang Liang1, Rui Zhang1, Wenzheng Zhang1,2, Yuan Qu1, Zhifei Ren1, Yuefeng Sun1
Yuanhong Zheng1, Dongsheng Ma1, Zirui Tang1,3, Boyu Niu1,3, Ziyang Miao1, Hejun Dong1
Siyi Qian1,2, Junyuan Zhang1, Jingzhou Chen1,2, Fangdong Wang1, Xiaomeng Zhao1, Liqun Wei1
Wei Li1, Shasha Wang1, Ruiliang Xu1, Yuanyuan Cao1, Lu Chen1, Qianqian Wu1, Huaiyu Gu1
Lindong Lu1, Keming Wang1, Dechen Lin1, Guanlin Shen1, Xuanhe Zhou1,3, Linfeng Zhang3
Yuhang Zang1, Xiaoyi Dong1, Jiaqi Wang1, Bo Zhang1, Lei Bai1, Pei Chu1, Weijia Li1, Jiang Wu1
Lijun Wu1, Zhenxiang Li1, Guangyu Wang1, Zhongying Tu1, Chao Xu1, Kai Chen1
Yu Qiao1, Bowen Zhou1, Dahua Lin1 (cid:0), Wentao Zhang1,2 (cid:0), Conghui He1 (cid:0)

1Shanghai Artificial Intelligence Laboratory, 2Peking University, 3Shanghai Jiao Tong University

We introduce MinerU2.5, a 1.2B-parameter document parsing vision-language model that
achieves state-of-the-art recognition accuracy while maintaining exceptional computational
efficiency. Our approach employs a coarse-to-fine, two-stage parsing strategy that decouples
global layout analysis from local content recognition. In the first stage, the model performs
efficient layout analysis on downsampled images to identify structural elements, circumvent-
ing the computational overhead of processing high-resolution inputs. In the second stage,
guided by the global layout, it performs targeted content recognition on native-resolution
crops extracted from the original image, preserving fine-grained details in dense text, complex
formulas, and tables. To support this strategy, we developed a comprehensive data engine
that generates diverse, large-scale training corpora for both pretraining and fine-tuning. Ulti-
mately, MinerU2.5 demonstrates strong document parsing ability, achieving state-of-the-art
performance on multiple benchmarks, surpassing both general-purpose and domain-specific
models across various recognition tasks, while maintaining significantly lower computational
overhead.

* Equal contribution (cid:0) Corresponding author
Correspondence: Conghui He, heconghui@pjlab.org.cn
Code: https://github.com/opendatalab/MinerU
Model: https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B
Date: September 30, 2025

‡ Project leader

5
2
0
2

p
e
S
9
2

]

V
C
.
s
c
[

2
v
6
8
1
2
2
.
9
0
5
2
:
v
i
X
r
a

1

 
 
 
 
 
 
MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Contents

1 Introduction

2 Related Work

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2.1 Traditional Pipelines .
2.2 General-Purpose Vision Language Models
. . . . . . . . . . . . . . . . . . . . . . . . . .
2.3 Domain-Specific Vision Language Models . . . . . . . . . . . . . . . . . . . . . . . . . . .

.

.

.

4

5
5
6
6

3 MinerU2.5

.

.

.

.

3.1 Model Architecture .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
3.2 Two-Stage Parsing Strategy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
3.3 Training Recipe .
.
Stage 0-Modality Alignment
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
Stage 1-Document Parsing Pre-training . . . . . . . . . . . . . . . . . . . . . . . .
Stage 2-Document Parsing Fine-tuning . . . . . . . . . . . . . . . . . . . . . . . .

6
6
7
8
8
9
9
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10

3.3.1
3.3.2
3.3.3
3.3.4 Data Augmentation Strategies

3.4 Model Deployment .

.

.

.

.

.

.

.

.

.

.

4 Data Engine

.

.
.

.
.

.
.

4.1 Overall Workflow .

11
.
.
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
4.1.1 Data Curation .
4.1.2 Pre-training Dataset Preparation . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
Fine-tuning Dataset Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
4.1.3
4.2 Task Reformulation and Enhancement . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
4.2.1 Layout Analysis .
Formula Recognition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
4.2.2
4.2.3 Table Recognition .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
Iterative Mining via Inference Consistency . . . . . . . . . . . . . . . . . . . . . . . . . . 17

4.3

.

.

.

5 Evaluation

5.1 Full-Document Parsing Task .

19
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
5.1.1 Evaluation Details and Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
5.1.2 Evaluation Results .
5.2 Element-Specific Parsing Task . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
Formula Recognition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

5.2.1 Layout Analysis .
.
5.2.2 Table Recognition .
5.2.3

.
.

6 Conclusion

25

.

.

.

.

.

.

A Qualitative examples
.
A.1 Overview .

30
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
.
.
.
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
A.1.1 Among PDF types
.
A.1.2 Among Table types .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
A.1.3 Among Formula types . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
A.2 Compare to Previous Versions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50

.
A.2.1 Table .
.
A.2.2 Formula .
.
A.2.3 Layout&OCR .
A.3 Compare with Others .
.
.

A.3.1 Table .
.
A.3.2 Formula .

.
.
.
.
.
.

.
.
.
.
.
.

.
.
.
.
.
.

.
.

.
.

.
.

.
.

.

.

2

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

A.3.3 Layout&OCR .

.

.

.

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53

B Prompt Details

.
B.1 Layout Detection .
B.2 Text Recognition .
.
.
B.3 Formula Recognition .
.
B.4 Table Recognition .

.
.

.

56
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56
.
.
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 57
.

.
.
.
.

.
.
.
.

.
.
.
.

3

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 1: Performance Highlights of MinerU2.5 on OmniDocBench. MinerU2.5 consistently out-
performs both general-purpose VLMs (e.g., Gemini-2.5 Pro, Qwen2.5-VL-72B, GPT-4o) and domain-
specific models (e.g., MonkeyOCR, dots.ocr, PP-StructureV3), establishing new performance records in
text recognition, formula recognition, table recognition, and reading order prediction. Detailed results
are presented in Table 5.

1 Introduction

Document parsing [57] serves as a fundamental task in multimodal understanding, underpinning
a variety of downstream applications such as information extraction [18, 43], Retrieval-Augmented
Generation (RAG) [19, 56, 58] and intelligent document analysis [2, 4, 40]. In contrast to natural images,
document images are characterized by significantly higher resolutions, denser content, and more
complex structural layouts [20, 51, 52]. These inherent properties introduce a unique set of challenges.
Firstly, the high resolution and fine-grained layout structures necessitate models capable of processing
images at their native resolution. Secondly, the text-dense and often lengthy nature of documents
imposes stringent requirements on the parameter efficiency and robustness of the models. Thirdly, the
success of OCR is contingent not only on precise text recognition but also heavily on reliable layout
analysis and efficient inference.

Contemporary approaches to document parsing can be broadly categorized into two paradigms:
pipeline-based approaches [8, 24, 32, 46] and end-to-end approaches based on VLMs [1, 3, 7, 37, 52].
The former employs a modular design, decomposing the task into discrete stages such as layout
detection, reading order prediction, and recognition of text lines, formulas, and tables. Each stage
is handled by a specialized model. While this approach offers interpretability, it suffers from a
cumbersome workflow and the potential for error propagation across modules. The latter paradigm
exhibits superior semantic modeling capabilities, yet it is still widely constrained by the hallucination
problem in long-document processing and suffers from severe efficiency bottlenecks when dealing
with high-resolution inputs. A critical factor limiting the performance and efficiency of VLM-based

4

MinerU2.5dots.ocrGemini-2.5 ProMonkeyOCR-pro-3BQwen2.5-VL-72BMonkeyOCR-pro-1.2BPP-StructureV3Nanonets-OCR-sMinerU2-VLMMistralOCRMinerU2-pipelinePOINTS-ReaderGPT-4oOCRFluxDolphinInternVL3.5-241BMarker1009590858010090807060Text BlockFormulaTableReading OrderOverall PerformanceElement-wise Performance1-Edit1-EditTEDSCDM100959085801009590858010095908580MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

parsing is token redundancy, arising from large blank or low-information regions within the document
image.

In response to the aforementioned challenges, we introduce a new document parsing framework,
MinerU2.5. The key innovation is a decoupled architecture that separates global layout analysis from
local content recognition via an efficient coarse-to-fine, two-stage inference mechanism. In the first stage,
the model conducts fast and holistic layout analysis on downsampled document images, capturing the
global structural organization with minimal computational cost. In the second stage, guided by the
detected layout, it crops key regions from the original high-resolution input and performs fine-grained
recognition within local windows, thereby preserving native resolution and ensuring high accuracy.
This decoupled strategy not only reduces computational cost by an order of magnitude, primarily
by avoiding the enormous number of visual tokens with O(N2) complexity inherent in end-to-end
native-resolution approaches [3, 6, 37], but also brings multiple advantages: it significantly enhances
the interpretability of parsing, effectively mitigates the common hallucination problem in VLMs, and
allows the two stages to be independently optimized and iterated, resulting in more robust and efficient
parsing capabilities. Ultimately, with its lightweight design of only 1.2B parameters, MinerU2.5 exhibits
strong adaptability and efficiency in scenarios with long documents and high-density content while
ensuring high parsing accuracy. Furthermore, to overcome the challenges of insufficient data diversity,
sample imbalance, and inconsistent annotation quality in document parsing, we have developed a
closed-loop data engine for complex documents. This engine systematically collects, processes, and
generates large-scale, high-quality document corpora. This ensures that our model exhibits precise
parsing capabilities and robustness across a wide spectrum of layouts, document types, and complex
elements.

MinerU2.5 not only achieves state-of-the-art (SOTA) performance across a wide range of public
benchmarks but also represents a qualitative leap in practical application and user experience over the
previous MinerU2 version, as demonstrated by the examples in Appendix A . Its key improvements
include:

• Comprehensive and Granular Layout Analysis: It not only preserves non-body elements like
headers, footers, and page numbers to ensure full content integrity, but also employs a refined
and standardized labeling schema. This enables a clearer, more structured representation of
elements such as lists, references, and code blocks.

• Breakthroughs in Formula Parsing: Delivers high-quality parsing of complex, lengthy mathe-

matical formulae and accurately recognizes mixed-language (Chinese-English) equations.

• Enhanced Robustness in Table Parsing: Effortlessly handles challenging cases, including rotated

tables, borderless tables, and tables with partial borders.

2 Related Work

2.1 Traditional Pipelines

Early OCR systems [8, 24, 32, 46] decompose document parsing into modular pipelines, sequentially
executing layout detection [44, 59], text recognition [8], and reading order [50]. For instance, Marker
[32] implements a sequential pipeline integrating Surya OCR [33] with layout analysis and reading
order prediction modules to process diverse document types. MinerU [46] leverages PDF-Extract-Kit
[30] to orchestrate multiple specialized models for layout detection, formula recognition and table
extraction. This modular architecture enables specialized optimization of individual components and
facilitates targeted refinement of specific subtasks through well-defined module boundaries. However,
pipeline-based methods are prone to error propagation across stages and exhibit limited robustness
when confronted with complex layouts such as multi-column text or cross-page structures. Moreover,

5

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

modular systems often entail multiple interdependencies in practice, rendering usage, maintenance,
and updates cumbersome and less efficient.

2.2 General-Purpose Vision Language Models

General-purpose vision language models (VLMs) [1, 3, 7, 63] have emerged as an alternative paradigm
for document understanding. Gemini2.5 Pro [7] demonstrates strong OCR capabilities among general
VLMs, surpassing traditional pipeline models like MinerU [46] in text parsing and approaching spe-
cialized systems like UniMERNet [45] in formula recognition, showcasing the potential of VLMs in
OCR applications. Among open-source models, Qwen2.5-VL-72B [3] achieves the best results, using
native-resolution vision encoders [10] to adapt to different image sizes, demonstrating the effectiveness
of arbitrary-resolution processing in OCR tasks. However, these general models exhibit inherent limi-
tations for document-centric tasks. Proprietary models like Gemini2.5 Pro [7] are expensive and slow
in processing, while open-source models require massive parameter scales for optimal performance,
limiting practical deployment. Additionally, both types remain susceptible to hallucinations in densely
populated text regions, affecting reliability in complex document layouts.

2.3 Domain-Specific Vision Language Models

End-to-End Approaches. Recent domain-specific models [4, 6, 15, 23, 35, 37, 52] adopt end-to-end
architectures that unify document parsing within a single model, eliminating the need for cascaded
processing stages. GOT [52], as an early representative of end-to-end approaches, pioneered the OCR
2.0 paradigm by establishing both model architecture and data methodology that unified recognition
across diverse modalities—text, formulas, tables, and charts—within a single framework. Subsequent
models like Ocean-OCR [6], olmOCR [35], and dots.ocr [37] leverage native resolution vision encoders
to process documents and construct massive document corpora, further advancing the performance of
end-to-end architectures. However, end-to-end designs face scalability challenges: joint optimization of
layout and content often reduces accuracy on complex documents, while native-resolution processing
introduces prohibitive O(N2) complexity. Despite strengths in semantic modeling, these models suffer
from hallucinations on long documents and severe inefficiency with high-resolution inputs, where
token redundancy from blank or low-information regions becomes a major bottleneck.

Multi-Stage Approaches. Recently, multi-stage methods [11, 17] leveraging VLMs decouple layout
analysis from content recognition, combining the efficiency of pipeline approaches with the accuracy
of unified models. Dolphin [11] employs a Swin-Transformer VLM that first performs page-level
layout, then conducts efficient parallel parsing of identified regions. However, Swin-Transformer’s
fixed resolution severely limits crop parsing—sub-regions with extreme aspect ratios suffer from
distortion when resized to predetermined dimensions, degrading recognition quality while increasing
computational overhead. MonkeyOCR [17] adopts a similar multi-stage strategy but employs a
native resolution vision encoder in its second stage, improving both performance and efficiency.
However, MonkeyOCR requires multiple specialized models across different stages, increasing system
complexity and deployment overhead. A single unified model with native resolution parsing presents
a promising direction to address these limitations, which is precisely the goal that MinerU2.5 pursues.

3 MinerU2.5

3.1 Model Architecture

Figure 2 illustrates the overall architecture of MinerU2.5, which is inspired by the classical Qwen2-VL
framework [48]. The overall model architecture consists of three major components:

6

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 2: The framework of MinerU2.5. In stage I, MinerU2.5 performs rapid, global layout analysis
on a downsampled page. In stage II, MinerU2.5 leverages the layout results to crop key regions from
the original high-resolution document, performing fine-grained content recognition (e.g., text, table,
and formula recognition) within these native-resolution local regions. The detailed prompts used in
the inference are illustrated in Appendix B.

Language Model. For the decoder, we employ a 0.5B-parameter Qwen2-Instruct model [42], as
document parsing tasks typically exhibit relatively low dependency on large-scale language models.
To better accommodate diverse resolutions and aspect ratios in cropped image parsing, we replace
the original 1D-RoPE [39] with M-RoPE [48], thus enhancing the model’s generalization ability across
varying resolutions.

Vision Encoder.
Inspired by Qwen2-VL, MinerU2.5 incorporates a native-resolution encoding mech-
anism. Although the Qwen2.5-VL series [3] adopts window attention to improve efficiency, this design
causes performance degradation in document parsing tasks. Therefore, we employ a 675M-parameter
NaViT [10] initialized from Qwen2-VL. This vision encoder supports dynamic image resolutions and
employs 2D-RoPE for positional encoding, enabling it to flexibly handle inputs of various resolutions
and aspect ratios.

Patch Merger. To balance efficiency and performance, the architecture uses pixel-unshuffle [38] on
adjacent 2 × 2 vision tokens, preprocessing the aggregated vision tokens before passing them into the
large language model. This design effectively achieves a trade-off between computational efficiency
and task performance.

3.2 Two-Stage Parsing Strategy

In high-resolution document parsing with VLMs, a large proportion of low-information blank regions
introduces severe token redundancy, which substantially reduces overall efficiency. Existing end-to-end
visual encoding strategies for VLMs face inherent limitations:

7

NativeRes-ViTLM Decoder2.5675M0.5BDocumentMarkdownTable 1: Results of DocLayout-YOLO with di:erent optimization strategies. Pretrain denotes DocSynth-300K pre-training. Resulting DocLayout-YOLO signiﬁcantly outperforms the baseline model. \\(\\uparrow \\Delta\\) denotes improvements compared with baseline YOLO-v10 model.<table><tr><td>Improvement\nGL-CRM</td><td>Pretrain</td><td>D4LA\nmAP</td><td>AP50</td><td>DocLayNet\nmAP</td><td>AP50</td><td>Academic\nmAP</td><td>AP50</td><td>Textbook\nmAP</td><td>AP50</td><td>Market Analysis\nmAP</td><td>AP50</td><td>Financial\nmAP</td><td>AP50</td>...Figure 4: Illustration of Controllable Receptive Module (CRM), which extracts and fuses features of varying scales and granularities.<img_url># 4.1 CONTROLLABLE RECEPTIVE MODULE......\\[\n\\hat {F} = \\operatorname {C o n c a t} \\left(\\left[ F _ {1}, F _ {2}, \\dots , F _ {n} \\right]\\right) \\tag {2}\n\\]\\[\nM = \\sigma (G E L U (B N (C o n v _ {g a t e} (\\hat {F})))) \\tag {3}\n\\]StageI: Layout AnalysisLayoutDetection1036px1036px<|box_start|>163 081 836 129 <|box_end|><|ref_start|>table_caption<|ref_end|><|rotateup|><|box_start|>169138 831 250 <|box_end|><|ref_start|>table<|ref_end|><|rotateup|><|box_start|>162 264 486 471 <|box_end|><|ref_start|>text<|ref_end|><|rotateup|><|box_start|>496 268 832 421 <|box_end|><|ref_start|>image<|ref_end|><|rotateup|><|box_start|>493 422 832 468 <|box_end|><|ref_start|>title<|ref_end|><|rotateup|>...2640px3320pxResizeDropNative-ResFigureStageII: Content RecognitionOrder: 1Box：[163 81 836 129]Type: Table CaptionOrientation: CropFormula RecognitionTable RecognitionParallel decoding1715 px1541687 px359 px1124 px64Crop 1Crop 2...Crop 8...Text RecognitionAdjustADRpxpxFigure⚙⚙⚙Merge by Order< Crop 1>< Crop 2>< Crop 3>< Crop 4>< Crop 5>MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

• Crop-based approaches [52, 63] can partially reduce computational overhead but inevitably

sacrifice semantic consistency and layout information.

• Native-resolution approaches [3, 13, 37, 29] preserve fine-grained details in high-resolution
inputs, yet produce an enormous number of visual tokens with O(N2) complexity, rendering
them computationally impractical.

To address this dilemma, we propose a two-stage parsing strategy. This design decouples layout
analysis from local content recognition, thereby improving interpretability, enhancing optimization
potential for downstream tasks such as OCR, and effectively reducing the risk of hallucinations. Below,
we provide more details of each stage.

Stage I: Layout Analysis.
In the first stage, the input image is uniformly resized to a thumbnail
of 1036 × 1036 pixels, enabling global layout analysis while controlling computational cost. The
parameter choice is determined through systematic analysis: the thumbnail size must balance global
visibility and efficiency—too small leads to detail loss, while too large triggers the quadratic complexity
of NaViT. In contrast to native-aspect-ratio thumbnails, adopting a fixed thumbnail size results in more
stable bounding-box localization and facilitates more efficient training.

Stage II: Content Recognition.
In the second stage, the model leverages the detected layout to crop
the native high-resolution image into local regions, which are then parsed at fine granularity. Cropped
regions are fed at native resolution with an upper bound of 2048 × 28 × 28 pixels, avoiding detail loss
from overly small crops while preventing redundant computation from excessively large ones. This
design ensures a robust trade-off between accuracy and efficiency across diverse document parsing
scenarios.

3.3 Training Recipe

As described in Section 3.1, MinerU2.5 consists of three core components: vision encoder, patch merger,
and language model. Prior to the pre-training phase of MinerU2.5, the vision encoder is initialized
from Qwen2-VL-2B-Instruct, while the language model is initialized from Qwen2-Instruct-0.5B. The
overall training procedure of MinerU2.5 is divided into three stages, as summarized in Table 1.

3.3.1 Stage 0-Modality Alignment

To ensure that MinerU2.5 acquires the fundamental vision–language alignment ability as well as the
OCR recognition capability, we first conduct two-stage modality alignment training on Visual Question
Answering (VQA) datasets.

Language-Image Alignment. Only the two-layer MLP within the patch merger is trained, while
both the vision encoder and the language model are frozen. We use image-caption pairs1 for training
to effectively project visual features into the LLM embedding space, thus achieving alignment of the
modal representation.

Visual Instruction Tuning. All model parameters are unfrozen. The focus is on knowledge accu-
mulation and ability expansion, particularly strengthening visual alignment and OCR capability. The
training data2 mainly covers image captioning, interleaved text-image pairs, visual alignment, and
OCR data. The goal is to enable MinerU2.5 to follow instructions across diverse visual tasks and
generate reasonable responses.

1This dataset is sourced from LLaVA-Pretrain.
2This dataset is sourced from LLaVA-Instruct.

8

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Stage-0

Stage-1

Stage-2

a

b

n
o
i
s
i
V

a
t
a
D

l
e
d
o
M

g
n
i
n
i
a
r
T

Max Resolution
#Tokens per Image

2048 × 28 × 28
4 ∼ 2048

4096 × 28 × 28
4 ∼ 4096

2048 × 28 × 28
4 ∼ 2048

2048 × 28 × 28
4 ∼ 2048

Dataset
#Samples

Trainable
Sequence Length
Data Augmentation

Batch Size
LR: ψViT
LR: {θMLP, ϕLM}
Epoch

Image Caption
558K

MLP Adaptor
4096
No

128
1 ×10−3
1 ×10−3
1

VQA
665K

All
4096
No

64
1 ×10−5
1 ×10−5
1

Layout&OCR
6.9M

Layout&OCR
630K

All
8192
Yes

256
4 ×10−6
4 ×10−5
2

All
16384
Yes

256
4 ×10−6
4 ×10−5
3

Table 1: Training setup and hyperparameters in three training stages.

Empirical results demonstrate that MinerU2.5, after VQA-based modality alignment training, exhibits
significant improvements in tasks such as layout analysis and content recognition. Conversely, skipping
this stage leads to higher losses and a clear drop in overall performance.

3.3.2 Stage 1-Document Parsing Pre-training

The objective of the document parsing pre-training stage is to enable MinerU2.5 to acquire two
fundamental capabilities: layout analysis and content recognition. At this stage, all parameters of the
model remain fully trainable.

Training Data. We leveraged a large-scale mixture of model-labeled data and public datasets to
ensure both sufficient scale and document diversity. For layout analysis, in consideration of train-
ing efficiency, full document images were resized to a fixed resolution with corresponding relative
coordinates, and the prompt “Layout Detection:” was used. For content recognition, we employed
single-element image samples of text blocks, formula blocks, and table blocks as inputs, with prompts
“Text Recognition:”, “Formula Recognition:”, and “Table Recognition:” respectively. More details
are shown in the Appendix B.

Training Configuration. The model, initialized from Stage 0, was trained for 2 epochs. Each epoch
consisted of a total of 6.9M samples, including 2.3M for layout analysis, 2.4M for text blocks, 1.1M for
formula blocks, and 1.1M for table blocks.

Through this document parsing pre-training, the model has acquired strong layout analysis and content
recognition capabilities, demonstrating excellent performance across most simple and medium-level
scenarios. The resulting model not only serves as a strong baseline for downstream fine-tuning, but
also functions as an efficient hard-sample miner within our data engineering pipeline, facilitating the
identification of challenging cases for human annotation and further improving document parsing
performance.

3.3.3 Stage 2-Document Parsing Fine-tuning

The objective of the document parsing fine-tuning stage is to further enhance parsing performance in
challenging scenarios, while maintaining the detection and parsing capabilities already acquired by

9

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

MinerU2.5.

Training Data. To achieve this goal, it is crucial to construct a compact yet high-quality dataset:

• To preserve the model’s fundamental capabilities, we sampled high-quality and diverse examples
from the pre-training dataset via data engineering and incorporated them into Stage 2 training,
ensuring broad coverage across different document element types.

• From a large-scale, multi-source PDF corpus, we employed data engineering to identify cases
where the model still underperformed. We summarized these difficult scenarios and conducted
targeted data collection with manual annotation to obtain high-quality samples representing
challenging cases.

Training Configuration. We fine-tuned the pre-trained model for 3 epochs. Each epoch contained a
total of 630K samples, consisting of 43K for layout analysis, 300K for text blocks, 147K for formula
blocks, and 140K for table blocks.

With this targeted data iteration strategy, Stage 2 fine-tuning enables the model to not only retain
its established document parsing abilities but also achieve significant improvements in previously
challenging scenarios.

3.3.4 Data Augmentation Strategies

To enhance the model’s robustness in handling diverse documents in an open-world setting, we
designed a variety of targeted data augmentation strategies during both Stage 1 and Stage 2. These
augmentations simulate common types of document interference, and can be categorized as shown
in Table 2.

Augmentation Type

Operations

Spatial Transformations
Background Transformations

Color Transformations
Degradation Transformations

Scaling, Grid Distortion, Rotation
Texture, Weather effect, Image background,
Watermark, Scanlines, Shadow
Brightness Contrast, Illumination, RGB Shift
PSF Blur, Vibration Blur, Gaussian Blur,
Erosion / Dilation

Table 2: Data augmentation strategies for document parsing.

Note that spatial transformations are not applied to layout analysis samples. For different element
types, we carefully design augmentation parameters and probabilities in order to strike a balance
between model performance and robustness.

3.4 Model Deployment

We implement an efficient offline inference pipeline for MinerU2.5 based on vLLM [16]. While
vLLM provides high-throughput serving for large language models, we introduce two additional
optimizations tailored for our two-stage document parsing pipeline to further minimize end-to-end
latency. First, we employ an asynchronous backend to handle batching submission of page-level
requests, enabling better overlap between CPU and GPU workloads. Second, we decouple Stage I
and Stage II into independent inference tasks, allowing downstream processing to begin as soon as
individual results become available, rather than waiting for entire batches.

10

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

A key challenge during deployment was suppressing degenerate token repetition without penalizing
legitimate repetitive structures (e.g., tables, equations, or structured content). To address this, we dy-
namically adjust sampling parameters like frequency penalty and presence penalty in Stage II based
on the layout type detected in Stage I. For instance, higher penalties are applied to text paragraphs,
while lower values are used for tabular content.

Furthermore, we carefully tuned key vLLM scheduling parameters, including max num batched tokens,
max num seqs, and cuda graph sizes, to improve batch utilization and kernel launch efficiency.

We evaluate all compared models on OmniDocBench [31], a dataset of 1,355 document pages with
an average of over 1,100 tokens per page. All models are tested using their official inference scripts
under a consistent batched parallel processing protocol, with vLLM startup overhead excluded for
fair comparison. After preliminary optimization, MinerU2.5 achieves an end-to-end throughput of
2.12 pages/s. The end-to-end generation speed, measured only on valid output tokens from Stage II,
reaches 2337.25 tokens/s3. As shown in Table 3, MinerU2.5 outperforms MonkeyOCR-Pro-3B by
4× and dots.ocr by 7× in page throughput, demonstrating strong inherent efficiency for large-scale
document parsing. Notably, even without any deployment optimizations, MinerU2.5 achieves a
baseline throughput of 0.95 pages/s and 1045.14 tokens/s, already surpassing other compared models
under default configurations.

Model

Parameters

Backend

Hardware

Tokens/sec

Pages/sec

MinerU2-VLM [46]
dots.ocr [37]
MonkeyOCR-pro-3B [17]
MonkeyOCR-pro-1.2B [17]
Nanonets-OCR-s [26]

MinerU2.5

0.9B
3.0B
3.7B
1.9B
3.7B

1.2B

SGLang [60]

vLLM [16]

A100 80G

vLLM

RTX 4090 48G
A100 80G
H200 141G

3091.23
311.06
520.16
589.76
605.92

1875.82
2337.25
4938.31

2.84
0.28
0.47
0.53
0.55

1.70
2.12
4.47

Table 3: Inference performance comparison of specialized VLMs and MinerU2.5 across different
backends and GPUs.

4 Data Engine

The state-of-the-art performance of MinerU2.5 is underpinned by a systematic Data Engine designed
to generate large-scale, high-quality training data with uniform annotation standards. This engine
first establishes a vast and diverse foundation through rigorous data curation and refined automated
annotation for pre-training. Building upon this foundation, we introduce our novel Iterative Mining
via Inference Consistency (IMIC) strategy, which efficiently identifies complex “hard cases” for targeted
human annotation. This multi-stage approach creates a virtuous cycle of improvement, progressively
enhancing the model’s capabilities. The entire process is illustrated in Figure 3.

4.1 Overall Workflow

4.1.1 Data Curation

Our process begins with a large-scale internal document pool comprising publicly available web
data and commercially procured documents. While diverse, this raw pool suffers from a significant

3The end-to-end generation speed is calculated based on the number of valid tokens produced by Stage II divided by the

total processing time for both stages.

11

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 3: Overview of the Data Engine. Our data pipeline consists of three core stages. (1) Data
Curation: We filter a massive, raw document pool to construct a diverse and balanced dataset based
on layout, document type, element balance, and language. (2) Pre-training Data Preparation: We
generate automated annotations for the curated data and then refine them using specialized, powerful
models for text, tables, and formulas to ensure high quality. (3) Fine-tuning Dataset Construction: We
employ our Iterative Mining via Inference Consistency (IMIC) strategy to automatically discover hard
cases, which then undergo meticulous expert curation to create a high-quality SFT dataset.

long-tail distribution. To mitigate this imbalance and enhance training robustness, we implement
a rigorous curation process to build a balanced Chinese-English dataset with high diversity across
multiple dimensions:

• Layout Diversity: We employ page-level image clustering to select exemplars from a wide

spectrum of visual layouts and styles.

• Document Type Diversity: Using document metadata (e.g., discipline, tags), we perform strati-
fied sampling to ensure a balanced representation of types such as academic papers, textbooks,
reports, and presentations.

• Element Balance: A preliminary detection model helps ensure a balanced class distribution of

key elements like titles, paragraphs, tables, formulas, and figures in the curated set.

• Language Balance: We filter the data to maintain a comparable volume of Chinese and English

documents.

4.1.2 Pre-training Dataset Preparation

Initial annotations for the curated dataset are generated using our MinerU2-pipeline, establishing a
baseline for subsequent refinement. To move beyond this baseline quality, we perform a multi-step
refinement process using specialized, expert models for different content types:

• Textual Content: We leverage the powerful Qwen2.5-VL-72B-Instruct to verify and correct initial

text recognition results on cropped text regions.

• Formula Content: Recognized formulas are substituted with higher-fidelity outputs from an

12

Data CurationPre-training Dataset PreparationFine-tuning Dataset Construction12PDFData PoolNetwork-AccessibleInternally ProcuredPublic DatasetLATSRMERAaPretrainingDatasetFinetuningDatasetLATabRMEROCRLarge-scaledata&FourTasksDiverse CasesHard CasesWithHigh-Precision Annotations1233Diverse Documents3PrimaryParsingQuality ImprovementImageFeatureClusteringMetaInfoSelectionLayoutAnalysisen&zhBalancingFilteringVisualDomainElementLanguageTextFormulaTableQwenVLTextResultFormulaResultTableResultUniMERNetSelf-TabRModel++GroundTruthTableTEDSFormulaCDMLayoutPageIoUHard CasesMinerU2.5 Stage-1 CheckpointMinerU2-VLMSampling n Times𝑹𝟏𝑹𝟐𝑹𝒏...High Internal VarianceGroundTruth(1) High-Value Data Screening(2) High-Quality AnnotationsModel Pre-annotation+Expert CurationRandomSelectionVisualDomainElementLanguageDiverseCasesMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

in-house UniMERNet model, which we retrained on our extensive formula dataset to boost its
accuracy.

• Table Content: All table structures are re-generated using an in-house, high-performance table

parsing model.

This refinement workflow yields a high-quality pre-training dataset of image-annotation pairs, covering
our four core tasks: layout analysis, text recognition, formula recognition, and table recognition.

4.1.3 Fine-tuning Dataset Construction

While pre-training ensures broad capabilities, the noise inherent in automated annotations creates
a ceiling for model performance. To break through this ceiling, our fine-tuning strategy pivots to
high-value, difficult examples. We designed an Iterative Mining via Inference Consistency (IMIC)
strategy to automatically filter these hard cases from the large-scale data pool. To ensure annotation
quality, these select samples are processed through an AI-assisted pipeline: they are first pre-annotated
by a foundation model, such as Gemini-2.5-Pro for complex tables, and then meticulously reviewed
and corrected by human experts4. The final Supervised Fine-Tuning (SFT) dataset combines these high-
quality hard cases with a smaller, randomly sampled set of regular examples, equipping MinerU2.5 to
excel in complex, real-world parsing scenarios.

4.2 Task Reformulation and Enhancement

To move beyond the limitations of existing document analysis methods, we systematically reformulated
the core tasks of layout analysis, formula recognition, and table recognition. This involved defining
more robust standards, designing novel task paradigms, and introducing specialized metrics and
representations.

4.2.1 Layout Analysis

A Unified Tagging System. A fundamental challenge in layout analysis is the lack of a standardized
tagging system. Existing datasets suffer from widespread inconsistencies in element definitions,
granularity, and scope. To address this, we engineered a hierarchical and comprehensive tagging
system by analyzing a vast corpus of documents. Our system is defined by three key principles:

• Comprehensive Coverage: It includes non-body content often ignored by others, such as headers,

footers, and page numbers, which is critical for downstream applications like RAG.

• Fine Granularity: It decomposes complex elements. For instance, figures are sub-categorized

into image, chart, and chemical structure, with distinct tags for their associated captions.

• Semantic Distinction: Visually distinct text blocks like code, algorithms, references, and lists are

assigned their own categories to preserve crucial semantic information.

Table 4 presents a comparison with mainstream tagging systems, highlighting the superior coverage
and granularity of our proposed system.

An Enhanced Multi-Task Paradigm. Traditional methods often treat layout analysis as a standard
object detection task, which ignores element rotation and defers reading order prediction to down-
stream modules. This approach not only impairs the recognition of rotated elements but also increases
system coupling. We propose an enhanced paradigm that redefines layout analysis as a multi-task
problem. This paradigm simultaneously predicts four key attributes for each document element in a
single inference pass: its Position, Class, Rotation Angle, and Reading Order. This integrated design

4Human review is augmented by our open-source QA tool, Dingo, which applies both rule-based and model-based checks.

See https://github.com/MigoXLab/dingo.

13

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Category

MinerU2-pipeline PaddleOCR

MinerU2.5

Textual

text
title
×
image caption
image footnote
table caption
table footnote
×
×
×
×
×

text, toc, abstract
title, page title
×
common caption
common footnote
common caption
common footnote
code
×
×
ref text, ref block
×

text
title
phonetic
image caption
image footnote
table caption
table footnote
code
code caption
algorithm
reference
list

Image

Table

Equation

image

table

equation
×

Page Margins

×
×
×
×
×

image, seal, chart, molecular

image

table

equation
×

header
footer
aside text
page number
page footnote

table

equation
equation block

header
footer
aside text
page number
page footnote

Table 4: Comparison of category support across different OCR systems.

effectively resolves the challenge of parsing rotated elements and streamlines the entire document
analysis pipeline.

PageIoU: A New Metric for Layout Quality. Layout analysis is typically evaluated with object
detection metrics like mAP, which rely on a fixed Intersection over Union (IoU) threshold. While
effective for well-defined objects, this approach is ill-suited for document layouts where text block
boundaries are often ambiguous. This can lead to a discrepancy where quantitative IoU-based scores
do not align with qualitative visual assessment.

As illustrated in Figure 4, a prediction that coarsely covers a paragraph (Case 1) can achieve a perfect
recall score (Recall@IoU0.5 = 1.0), while a more accurate line-by-line prediction (Case 2) is penalized for
not matching the paragraph-level ground truth, yielding a lower score (Recall@IoU0.5 = 0.6). Visually,
however, Case 2 is clearly a better fit.

To better evaluate document layout analysis, we introduce PageIoU, a page-level coverage metric that
measures the spatial consistency between predicted layouts and ground-truth annotations. Let the
predicted layout be

P = {bboxi | i = 1, 2, . . . , n},

and the ground truth be

G = {bboxj | j = 1, 2, . . . , m},
where each bbox denotes a bounding box on the page. We first compute coverage maps for both
prediction and ground truth. For example, the ground-truth coverage map is defined as:
(cid:40) m
∑
j=1

Gcover =

1p∈bboxj

p ∈ M

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

(cid:41)

,

14

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 4: Illustration of the proposed PageIoU metric. Case 1 and Case 2 show that IoU-based recall
may produce contradictory results compared with visual inspection, whereas PageIoU provides a
page-level coverage score that aligns more closely with qualitative observations.

where p is a page pixel and M denotes the non-background region of the page. Similarly, Pcover can be
obtained. Based on these, PageIoU is defined as:

PageIoU(P, G) =

|Pcover ∩ Gcover|
|Pcover ∪ Gcover|

=

∑p∈M min{Pcover(p), Gcover(p)}
∑p∈M max{Pcover(p), Gcover(p)}

.

Here, | · | denotes the summation over all pixel values, while ∩ and ∪ correspond to the pixel-wise
minimum and maximum of coverage counts, respectively. As shown in Figure 4, PageIoU aligns with
human perception, scoring the qualitatively poor prediction 0.78 and the superior one 0.97.

4.2.2 Formula Recognition

Decoupling Atomic and Compound Formulas. Existing models struggle with long or multi-line
formulas, and VLMs are prone to severe structural hallucinations. We identify the root cause as the
tendency to treat all formulas as monolithic entities, failing to account for internal complexity. To this
end, MinerU2.5 introduces a ”whole-part” decoupling philosophy, classifying formulas into two types
based on their structural and semantic integrity:

• Atomic Formulas: The smallest, indivisible semantic units with a tight 2D topology (e.g., a single

fraction, a matrix).

15

Document PageGcover1All/GT/Pred over GTBoxPredBoxMin(Num of GT cover, Num of Pred cover)Case1Case2𝑹𝒆𝒄𝒂𝒍𝒍@𝑰𝒐𝑼𝟎.𝟓=𝟏.𝟎𝑹𝒆𝒄𝒂𝒍𝒍@𝑰𝒐𝑼𝟎.𝟓=𝟎.𝟔𝟎=∑𝒎𝒊𝒏(𝑷𝒓𝒆𝒅,𝑮𝑻)∑𝒎𝒂𝒙(𝑷𝒓𝒆𝒅,𝑮𝑻)𝑹𝒆𝒄𝒂𝒍𝒍@𝑷𝒂𝒈𝒆𝑰𝒐𝒖≈𝟎.𝟕𝟖=𝟓𝟗𝟕𝟔𝑹𝒆𝒄𝒂𝒍𝒍@𝑷𝒂𝒈𝒆𝑰𝒐𝒖≈𝟎.𝟗𝟕=𝟔𝟓𝟔𝟕222222222222211111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111------11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111221111111111111111112222222221111111111111111111111111111111111111221111111111111111111111111111111111111111111111111111111111111111=∑𝒎𝒊𝒏(𝑷𝒓𝒆𝒅,𝑮𝑻)∑𝒎𝒂𝒙(𝑷𝒓𝒆𝒅,𝑮𝑻)nMax(Num of GT cover, Num of Pred cover)n1111111nNum of covered Pixel111111111111111111111111111111111111111111111111111111111111111111GroundTruthPredictionPcoverPredictionPcoverMIN (Pcover , Gcover)MAX (Pcover , Gcover)MAX (Pcover , Gcover)MIN (Pcover , Gcover)🤔💡😃💡😃🤔MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 5: The proposed ADR framework. First, a compound formula is decomposed into atomic
lines via layout analysis. Next, each line is individually recognized into LaTeX. Finally, the individual
results are structurally recombined to produce the complete output.

• Compound Formulas: An ordered set of atomic formulas composed vertically with specific

alignment relationships (e.g., a multi-line derivation aligned at the equal signs).

The Atomic Decomposition & Recombination (ADR) Framework. To handle the complexity of
compound formulas, we propose the ADR framework, which implements a multi-stage ”divide and
conquer” strategy. As illustrated in Figure 5, the ADR pipeline is powered by our versatile MinerU2.5
model, which acts as both a layout analyzer and a recognition engine, guided by task-specific prompts.
The process begins with an initial layout analysis pass, where MinerU2.5, guided by a layout detection
prompt, identifies and classifies all formula regions on the page as either atomic or compound. Next,
in the decomposition stage, each identified compound formula is segmented into an ordered sequence
of its constituent atomic formula lines, which are then cropped as individual images. In the third
stage, these simple, semantically independent atomic formula images are fed back into the MinerU2.5
model. This time, using a formula recognition prompt, the model performs high-precision translation
of each image into its corresponding LaTeX string. Finally, a lightweight recombination step uses the
positional information from the initial layout pass to structurally reassemble the individual LaTeX
strings into a single, coherent block, correctly formatting them within environments like align. This
approach transforms a single, difficult recognition task into a series of simpler ones, ensuring both
high-fidelity recognition of each component and the logical integrity of the overall structure.

4.2.3 Table Recognition

Overcoming Long-Sequence Dependencies. A primary challenge in table recognition is parsing
complex, long tables, especially for VLM-based approaches that target HTML. We attribute this

16

Formula Detection 1PromptPage Image“<image>Layout Detection:”+AtomicFormulaCompositeFormulaAtomic DecompositionFormula RecognitionStructural Reconstruction...Decompose Composite FormulaPromptCropped Formula“<image>Formula Recognition:”+......Reconstruction2342.5LaTeX2.5RenderLayoutInfoBox：:[72, 45, 869,712]Type: equationblockOrientation: Box：:[74, 46,725,101]Type: equationOrientation: Box：:[327,752,615,799]Type: equationOrientation: Box：:[74, 46,725,101]Type: equationOrientation: MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 6: The Table Recognition Pipeline. The pipeline first detects a table and its rotation, then
corrects its geometry. Next, the rectified image is recognized into the OTSL result, which is finally
converted to standard HTML.

difficulty to two inherent weaknesses of the HTML representation: (1) its complex, non-visual syntax
must be learned implicitly by the model; and (2) its high token redundancy results in excessively long
sequences, degrading performance on large tables. (The issue of rotated tables is effectively handled
by our enhanced layout paradigm.)

OTSL: An Optimized Table Structure Language. To robustly handle complex tables, we propose
a four-stage recognition pipeline, as depicted in Figure 6. The first two stages handle geometric
normalization: the system detects the table’s bounding box and rotation angle, then corrects the image
by cropping and rotating it to a canonical orientation. For the crucial third stage, table recognition,
we leverage the Optimized Table-Structure Language (OTSL) [25], an intermediate representation
developed by IBM [citation, 2023]. We adopted OTSL for its significant advantages over HTML as a
target for VLMs. Its minimalist design features a direct structural correspondence to a table’s visual
2D matrix, reducing the number of structural tokens from over 28 to just 5 and shortening the average
sequence length by approximately 50%. This makes it a far more effective target for model generation.
The final stage is a straightforward conversion from the OTSL output into standard HTML.

4.3

Iterative Mining via Inference Consistency

To enable continuous model improvement and the efficient expansion of our high-quality training
dataset, we introduce the IMIC (Iterative Mining via Inference Consistency) strategy. IMIC automati-
cally identifies the most challenging samples—or ”hard cases”—for the current model from a large
corpus of unlabeled data. This allows us to direct limited human annotation efforts toward the data
that offers the maximum value for model improvement.

The core principle of IMIC leverages the stochasticity inherent in model inference. For a given sample,
if the model has learned its features robustly, multiple inference passes with stochastic sampling
enabled should yield highly consistent outputs. Conversely, significant divergence across outputs
suggests the sample lies near the model’s decision boundary—a ’hard case’ where its predictions are
uncertain. Such samples are the most valuable candidates for manual annotation, as they directly
target the model’s specific weaknesses.

As illustrated in Figure 7, the implementation is tailored to each recognition task:

• Layout analysis: For full document pages, we perform multiple inference runs and measure
consistency by calculating the pairwise PageIoU between the resulting layouts. Samples falling
below a predefined similarity threshold are flagged as hard cases for precise manual annotation.

17

Table&Rotation Detection1PromptPage Image“<image>Layout Detection:”+Box：[152 140 421 960]Type: TableOrientation: Crop & Rotation Correction2Crop&Rotation2.5🛠PromptCropped Table“<image>Table Recognition:”+Table Recognition3OTSLOTSL to HTML4<table>  <tr><td>Table I. Anticon...</tr>  ...<table>RenderMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 7: llustration of the proposed IMIC (Iterative Mining via Inference Consistency) strategy.
From top to bottom: (a) Layout analysis, (b) Table recognition, and (c) Formula recognition. For each
task, the model performs multiple stochastic inference runs, and the pairwise consistency between
outputs is calculated with task-specific metrics (PageIoU, TEDS, CDM). Samples with low consistency
are automatically identified as hard cases and prioritized for manual annotation.

• Formula Recognition: For cropped formula images, consistency is assessed using the pairwise
CDM [47] across multiple outputs. Samples with low consistency are prioritized for manual
correction.

• Table Recognition: For cropped table images, we use the TEDS (Tree-Edit-Distance-based
Similarity) score to evaluate consistency across multiple recognized structures. Low-consistency
samples are routed to the manual annotation workflow.

18

Hard CasemPageIoU < 0.8Easy CasemPageIoU > 0.9InputImageInference Three TimesCalculate≈≠≈≠≠≠≈≈Hard CasemCDM < 0.3Easy CasemCDM > 0.7≠≠≈≈Hard CasemTEDS < 0.6Easy CasemTEDS > 0.9LayoutTableFormulaMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Model Type Methods

Parameters Overall↑ TextEdit↓

FormulaCDM↑ TableTEDS↑ TableTEDS-S↑ Read OrderEdit↓

Pipeline
Tools

General
VLMs

Specialized
VLMs

Marker-1.8.2 [32]
MinerU2-pipeline [46]
PP-StructureV3 [8]

GPT-4o [1]
InternVL3-76B [63]
InternVL3.5-241B [49]
Qwen2.5-VL-72B [3]
Gemini-2.5 Pro [7]

Dolphin [11]
OCRFlux [5]
Mistral-OCR [41]
POINTS-Reader [22]
olmOCR-7B [35]
MinerU2-VLM[46]
Nanonets-OCR-s [26]
MonkeyOCR-pro-1.2B [17]
MonkeyOCR-3B [17]
dots.ocr [37]
MonkeyOCR-pro-3B [17]
MinerU2.5

-
-
-

-
76B
241B
72B
-

322M
3B
-
3B
7B
0.9B
3.7B
1.9B
3.7B
3B
3.7B
1.2B

71.30
75.51
86.73

75.02
80.33
82.67
87.02
88.03

74.67
74.82
78.83
80.98
81.79
85.56
85.59
86.96
87.13
88.41
88.85
90.67

0.206
0.209
0.073

0.217
0.131
0.142
0.094
0.075

0.125
0.193
0.164
0.134
0.096
0.078
0.093
0.084
0.075
0.048
0.075
0.047

76.66
76.55
85.79

79.70
83.42
87.23
88.27
85.82

67.85
68.03
82.84
79.20
86.04
80.95
85.90
85.02
87.45
83.22
87.25
88.46

57.88
70.90
81.68

67.07
70.64
75.00
82.15
85.71

68.70
75.75
70.03
77.13
68.92
83.54
80.14
84.24
81.39
86.78
86.78
88.22

71.17
79.11
89.48

76.09
77.74
81.28
86.22
90.29

77.77
80.23
78.04
81.66
74.77
87.66
85.57
89.02
85.92
90.62
90.63
92.38

0.250
0.225
0.073

0.148
0.113
0.125
0.102
0.097

0.124
0.202
0.144
0.145
0.121
0.086
0.108
0.130
0.129
0.053
0.128
0.044

Table 5: Performance comparison of document parsing methods on OmniDocBench across text,
formula, table, and reading order extraction tasks.

Model Type Models

Pipeline
Tools

General
VLMs

Specialized
VLMs

Marker-1.8.2 [32]
MinerU2-pipeline [46]
PP-StructureV3 [8]

GPT-4o [1]
InternVL3-76B [63]
InternVL3.5-241B [49]
Qwen2.5-VL-72B [3]
Gemini-2.5 Pro [7]

Dolphin [11]
OCRFlux [5]
Mistral-OCR [41]
POINTS-Reader [22]
olmOCR-7B [35]
MinerU2-VLM[46]
Nanonets-OCR-s [26]
MonkeyOCR-pro-1.2B [17]
MonkeyOCR-3B [17]
dots.ocr [37]
MonkeyOCR-pro-3B [17]
MinerU2.5

Slides

0.1796
0.4244
0.0794

0.1019
0.0349
0.0475
0.0422
0.0326

0.0957
0.0870
0.0917
0.0334
0.0497
0.0745
0.0551
0.0961
0.0904
0.0290
0.0879
0.0294

Academic
Papers

Book

Textbook

0.0412
0.0230
0.0236

0.1203
0.1052
0.0857
0.0801
0.0182

0.0453
0.0867
0.0531
0.0779
0.0365
0.0104
0.0578
0.0354
0.0362
0.0231
0.0459
0.0235

0.1010
0.2628
0.0415

0.1288
0.0629
0.0237
0.0586
0.0694

0.0616
0.0818
0.0610
0.0671
0.0539
0.0357
0.0606
0.0530
0.0489
0.0433
0.0517
0.0332

0.2908
0.1224
0.1107

0.1599
0.0827
0.1061
0.1146
0.1618

0.1333
0.1843
0.1349
0.1372
0.1204
0.1276
0.0931
0.1110
0.1072
0.0788
0.1067
0.0499

Exam
Papers

0.2958
0.0822
0.0945

0.1939
0.1007
0.0933
0.0681
0.0937

0.1684
0.2072
0.1341
0.1901
0.0728
0.0698
0.0834
0.0887
0.0745
0.0467
0.0726
0.0681

Magazine Newspaper Notes

Financial
Report

0.1111
0.3950
0.0722

0.1420
0.0406
0.0577
0.0964
0.0161

0.0702
0.1048
0.0581
0.1343
0.0697
0.0652
0.0917
0.0494
0.0475
0.0221
0.0482
0.0316

0.2717
0.0736
0.0617

0.6254
0.5826
0.6403
0.2380
0.1347

0.2388
0.7304
0.5643
0.3789
0.2916
0.1831
0.1965
0.0995
0.0962
0.0667
0.0937
0.0540

0.4656
0.2603
0.1236

0.2611
0.0924
0.1357
0.1232
0.1169

0.2561
0.1567
0.3097
0.0937
0.1220
0.0803
0.1606
0.1686
0.1165
0.1116
0.1141
0.1161

0.0341
0.0411
0.0181

0.3343
0.0665
0.1117
0.0264
0.0169

0.0186
0.0193
0.0523
0.0951
0.0459
0.0236
0.0395
0.0198
0.0196
0.0076
0.0211
0.0104

Table 6: Document Parsing Performance in Text Edit Distance on OmniDocBench: evaluation using
edit distance across 9 PDF page types.

5 Evaluation

In this section, we present a comprehensive quantitative evaluation of MinerU2.5 to demonstrate
its effectiveness in document parsing tasks. Specifically, we compare MinerU2.5 against leading
general-purpose VLMs including GPT-4o [1], Gemini-2.5 Pro [7], and Qwen2.5-VL [3], as well as
state-of-the-art domain-specific VLMs such as dots.ocr [37], MonkeyOCR [17], and olmOCR [35].
The evaluation is organized into two parts: Section 5.1 presents full-document parsing results across
multiple benchmarks, while Section 5.2 focuses on element-specific capabilities including layout
analysis, formula recognition, and table recognition.

19

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Model

Mathpix [27]
PP-StructureV3 [8]
MinerU2-pipeline [46]
PaddleOCR [8]

Gemini-2.5 Pro [7]
GPT-4o [1]
Qwen2.5-VL-72B [3]
InternVL3-76B [63]
Qwen2-VL-7B [48]
MiniCPM-V2.6-8B [54]

0.064
0.068
0.099
0.323

0.080
0.085
0.093
0.125
0.165
0.244

MinerU2-VLM [46]
0.048
0.057
Ocean-OCR [6]
MonkeyOCR-pro-1.2B [17] 0.064
0.080
SmolDocling [28]
0.083
dots.ocr [37]
0.084
GOT[52]
0.033
MinerU2.5

Edit Distance ↓

F1-score ↑

Precision↑

Recall↑

BLEU↑

METEOR↑

en

zh

en

zh

en

zh

en

zh

en

zh

en

zh

0.223
0.210
0.225
0.649

0.204
0.450
0.140
0.282
0.270
0.437

0.182
0.062
0.190
0.878
0.179
0.117
0.082

0.930 0.919 0.950 0.952 0.911 0.889 0.901 0.593 0.924 0.768
0.871 0.929 0.856 0.924 0.892 0.935 0.796 0.570 0.902 0.802
0.663 0.919 0.635 0.908 0.703 0.934 0.504 0.571 0.670 0.810
0.707 0.864 0.690 0.912 0.730 0.821 0.517 0.537 0.674 0.699

0.922 0.927 0.940 0.959 0.906 0.898 0.877 0.690 0.921 0.862
0.919 0.686 0.929 0.694 0.910 0.703 0.870 0.354 0.922 0.495
0.923 0.940 0.936 0.956 0.912 0.926 0.879 0.798 0.924 0.876
0.828 0.871 0.842 0.889 0.817 0.856 0.728 0.527 0.829 0.759
0.849 0.883 0.834 0.847 0.873 0.942 0.795 0.578 0.859 0.763
0.804 0.778 0.793 0.721 0.837 0.875 0.695 0.431 0.640 0.642

0.936 0.941 0.926 0.927 0.947 0.958 0.893 0.611 0.950 0.837
0.937 0.962 0.932 0.956 0.956 0.974 0.906 0.912 0.945 0.916
0.929 0.934 0.918 0.925 0.944 0.948 0.884 0.699 0.941 0.850
0.899 0.157 0.895 0.140 0.912 0.268 0.839 0.048 0.907 0.151
0.904 0.931 0.920 0.951 0.890 0.913 0.849 0.639 0.911 0.842
0.895 0.928 0.891 0.934 0.906 0.929 0.835 0.805 0.874 0.848
0.945 0.965 0.948 0.966 0.942 0.964 0.909 0.817 0.950 0.887

Table 7: Evaluation results on Ocean-OCR bench on dense English (en) and Chinese (zh) OCR for
document-level pages. Some model results are sourced from the OceanOCR official reports.

Model

Overall AR OSM TA

OS

HF MC LTT

Base

MinerU2-pipeline[46]
Nanonets-OCR-s[26]
GPT-4o[1]
MonkeyOCR-pro-1.2B[17]
Qwen2.5-VL-72B[3]
MonkeyOCR-pro-3B[17]
olmOCR[35]
dots.ocr[37]
MinerU2.5

55.6
60.7
63.2
64.3
64.8
68.8
71.8
73.6
75.2

61.8
63.9
44.1
65.4
72.2
67.7
63.9
66.3
76.6

13.5
41.0
37.6
26.9
51.1
28.4
41.0
35.8
54.6

60.9
77.7
69.1
60.3
67.3
74.6
72.9
88.3
84.9

17.3
39.5
40.9
31.2
38.6
36.1
43.9
40.9
33.7

96.6
40.7
94.2
93.3
73.6
91.2
95.1
94.1
96.6

59.0
69.9
68.9
66.2
68.3
76.6
77.3
82.4
78.2

39.1
53.4
54.1
81.7
49.1
80.1
81.2
81.2
83.5

96.6
99.3
96.7
89.5
98.3
95.3
98.9
99.5
93.7

Table 8: Evaluation results on olmOCR-bench grouped by document types, including arXiv Math(AR),
Old Scans Math (OSM), Tables (TA), Old Scans (OS), Headers Footers (HF), Multi Column (MC) and
Long Tiny Text (LTT). Results on AR and OSM are replaced with ExpRate, and other results are sourced
from the official reports of olmOCR-bench and dots.ocr. The Overall Score (Overall) represents the
average across all document types.

5.1 Full-Document Parsing Task

We evaluate MinerU2.5’s full document parsing performance on three prominent benchmarks: Om-
niDocBench [31], Ocean-OCR [6] benchmarks, and olmOCR-bench [35]. These benchmarks provide
comprehensive evaluation from different dimensions, covering diverse document types, various
quality conditions, and different parsing challenges to thoroughly assess the model’s robustness and
generalization capabilities.

• OmniDocBench [31]: This evaluation dataset is designed for diverse document parsing in real-
world scenarios, encompassing nine document types, four layout types, and three language
types. It offers a comprehensive assessment of parsing scores for text, formulas, tables, and
reading order in full-document parsing, as well as for element-specific parsing tasks.

• olmOCR-bench [35]: This evaluation dataset comprises 1,402 PDF documents sourced from

20

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

various repositories, organized into seven subsets. Certain test patterns are applicable across all
document types (e.g., presence, absence, reading order), while others are specifically targeted at
challenging yet crucial content extraction objectives (e.g., tables, mathematical formulas).

• Ocean-OCR benchmark [6]: This evaluation dataset consists of 100 images from English papers
and 100 images from Chinese papers. It primarily evaluates the ability of text parsing and
employs several text OCR-related evaluation metrics, such as Normalized Edit Distance, F1
Score, Precision, Recall, BLEU, and METEOR.

5.1.1 Evaluation Details and Metrics

For OmniDocBench [31], we evaluate on the latest version with three key improvements:

• Enhanced resolution for Notes and Newspapers from 72 to 200 DPI, enabling more accurate

evaluation of fine-grained text and handwritten content.

• An addition of 374 pages to balance Chinese-English content distribution and enrich mathemati-

cal formula coverage. Currently, it contains a total of 1,355 pages.

• Evaluation methodology updated to hybrid matching algorithm.

The Overall score combines three core metrics:

Overall =

(1 − TextEdit) × 100 + TableTEDS + FormulaCDM
3

For olmOCR-bench [35], we replace the formula scores of Arxiv Math (AR) and Old Scans Math (OSM)
with the more reliable ExpRate of CDM [47]. The original evaluation compares LaTeX formulas by
parsing them into abstract syntax trees and matching Unicode tokens, which is overly sensitive to
syntax variations (e.g., \cdots vs. \dotsb) that render identically but are scored as different. To avoid
this bias, we adopt ExpRate, which directly compares rendered outputs, assigning 1 for exact matches
and 0 otherwise.

5.1.2 Evaluation Results

MinerU2.5 demonstrates exceptional performance across all benchmarks, achieving state-of-the-art
results in most metrics (Tables 5 to 8).

As shown in Table 5, MinerU2.5 achieves an overall score of 90.67 on OmniDocBench, outperforming
the second-best model MonkeyOCR-pro-3B [17] by 1.82 and dots.ocr [37] by 2.26 points. In text
recognition tasks, MinerU2.5 achieves the lowest edit distance of 0.047, marginally better than dots.ocr
at 0.048 and significantly outperforming PP-StructureV3 [8], which scores 0.073. For formula recog-
nition, MinerU2.5 leads with a CDM score of 88.46, exceeding both Qwen2.5-VL-72B at 88.27 and
MonkeyOCR-3B at 87.45. In table recognition tasks, MinerU2.5 achieves the highest TEDS score of 88.22
and TEDS-S score of 92.38. For reading order evaluation, it maintains the best edit distance of 0.044.
The document-type specific results presented in Table 6 demonstrate that MinerU2.5 achieves best or
second-best performance in 6 out of 9 categories. For textbooks, it delivers the best performance with
an edit distance of 0.0499, substantially outperforming dots.ocr’s 0.0788. For newspapers, MinerU2.5
leads with a score of 0.0540, surpassing all competing models. In both financial reports and slides
categories, MinerU2.5 achieves second-best performance with scores of 0.0104 and 0.0294 respectively.

For the results of the Ocean-OCR benchmark presented in Table 7, MinerU2.5 demonstrates exceptional
performance in dense OCR tasks. On English documents, it achieves the lowest edit distance of 0.033
and the highest F1-score of 0.945, accompanied by best-in-class BLEU and METEOR scores of 0.909
and 0.950 respectively. For Chinese documents, MinerU2.5 achieves the highest F1-score of 0.965 and
Precision of 0.966, while maintaining strong BLEU and METEOR scores of 0.817 and 0.887 respectively.

21

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Method

Textual
R↑

P↑

F1↑ P↑

Image
R↑

F1↑ P↑

Table
R↑

F1↑ P↑

OmniDocBench [31]

Equation
R↑

Page Margins
R↑

F1↑ P↑

Full Page
R↑

F1↑

F1↑ P↑

90.4 48.2 58.1 72.1 51.2 57.2 72.6 55.1 61.0
LayoutLMv3 [14]
MinerU2-VLM [46]
90.3 95.6 91.9 87.2 91.0 90.9 96.0 97.1 97.8 87.4
DocLayout-YOLO [59] 95.4 98.3 96.5 87.6 96.7 94.7 94.9 98.1 98.4 95.3
96.8 96.7 96.6 86.4 92.1 92.9 96.6 97.4 98.2 96.5
PP-StructureV3 [8]
97.2 98.0 97.5 89.6 94.3 95.0 96.0 98.1 98.4 92.4
MinerU2.5

-

36.9
95.8
90.6
97.6
99.6

D4LA [9]

-
-
-

-
-
98.7

-
-
-
-
-
90.5
92.3 97.7 94.1
93.8
-
96.7 92.9 86.2 88.1 94.8 96.2 94.6
94.7 89.9 95.4 91.4 95.8 97.0 95.9

-
-

-
-

86.9 41.2 52.4 59.3 32.0 31.4 59.3 41.8 43.3
-
-
-
LayoutLMv3 [14]
-
-
79.1
MinerU2-VLM [46]
88.3 88.9 87.9 56.7 35.0 38.1 89.1 84.1 90.6 38.3
91.1
82.6 95.4 87.3
DocLayout-YOLO [59] 86.3 97.8 90.8 41.5 92.9 62.6 87.6 89.0 89.8 31.9
-
92.1 76.8 84.2 79.1 85.7 91.0 86.0
88.5 93.5 90.0 50.1 82.3 67.9 87.1 81.1 89.7 24.6
PP-StructureV3 [8]
91.8 98.3 94.6 53.8 94.3 72.8 91.9 78.9 91.4 46.0 100.0 91.0 75.9 97.6 84.2 90.4 92.5 90.2
MinerU2.5

50.5
99.4
80.2
85.9

-
-
95.0

-
-
-

-
-

-
-

-

88.8 59.3 67.9 79.0 50.3 61.9 75.2 54.9 61.8
LayoutLMv3 [14]
MinerU2-VLM [46]
88.1 96.1 91.7 85.5 78.1 91.3 94.9 94.4 95.6 83.9
DocLayout-YOLO [59] 86.9 96.8 91.2 85.8 96.2 91.3 92.0 95.7 94.8 80.5
90.9 97.3 93.8 91.7 90.4 94.2 96.4 93.7 96.7 88.8
PP-StructureV3 [8]
90.2 99.6 94.8 92.5 96.3 95.9 96.3 93.5 97.1 88.9
MinerU2.5

-

31.9
97.0
86.9
96.0
98.6

DocLaynet [34]

-
-
-

-
-
97.7

-
-
-
-
-
90.0
82.8
88.0 96.3 90.9
-
92.1 76.8 79.3 77.4 92.4 95.7 93.0
93.5 76.3 98.9 86.3 92.8 97.7 94.6

-
-

-
-

Table 9: Comparison of layout analysis performance (Precision@PageIoU, Recall@PageIoU, F1-
score@PageIoU) across different methods and content types on multiple layout analysis benchmarks.

The results of olmOCR-bench are shown in Table 8, where MinerU2.5 achieves an overall score of
75.2, surpassing dots.ocr’s 73.6 by 1.6 points. In the arXiv Math category, it leads with a score of 76.6,
outperforming Qwen2.5-VL-72B [3]’s 72.2 by 4.4 points. For Old Scans Math, MinerU2.5 dominates
with a score of 54.6, exceeding all other evaluated models. In the Long Tiny Text category, it achieves
83.5, surpassing MonkeyOCR-pro-1.2B [17] which scores 81.7.

5.2 Element-Specific Parsing Task

5.2.1 Layout Analysis

We validate the effectiveness of our layout analysis by performing a fair, zero-shot comparison with
leading methods on three publicly available datasets:

• OmniDocBench [31]: A recent benchmark for document parsing that includes detailed layout

annotations.

• D4LA [9]: Contains 11,092 noisy document images annotated with 27 categories, split into 8,868

training and 2,224 test images. We use its test set with annotations for evaluation.

• DocLayNet [34]: A large-scale dataset of 80,863 pages from 7 document types, manually anno-

tated with 11 categories. We use its validation set with annotations for evaluation.

We compare our MinerU2.5 with several recent methods, including LayoutLMv3 [14], MinerU2-
VLM [46], DocLayout-YOLO [59] and PP-StructureV3 [8]. For a equitable assessment, we evaluate
all models without dataset-specific training. To account for differences in detection granularity and
category definitions, we unified the evaluation by mapping all labels to five broad categories and
using the PageIoU metric, which assesses the spatial overlap without considering category labels for
the “Full Page” score.

The results in Table 9 show that MinerU2.5 significantly outperforms other models, achieving the top
Full Page F1-score@PageIoU across all benchmarks. It also secures leading F1-scores@PageIoU for the

22

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Method

PubTabNet

FinTabNet

CC-OCR

OCRBench v2

In-house TR Benchmark

TEDS↑ TEDS-S↑ TEDS↑ TEDS-S↑ TEDS↑ TEDS-S↑ TEDS↑ TEDS-S↑ TEDS↑

TEDS-S↑

RapidTable [36]
MiniCPM-V 4.5 [55]
InternVL3.5-241B [49]
Qwen2.5-VL-7B [3]
Qwen2.5-VL-72B [3]
GPT-4o [1]
Gemini-2.5 Pro [7]
dots.ocr [37]
Nanonets-OCR-s [26]
MinerU2-VLM [46]
MinerU2.5

86.57
80.30
83.75
81.60
84.39
76.53
-
90.65
63.58
88.11
89.07

96.43
87.67
88.76
86.78
87.91
86.16
-
93.76
75.68
90.85
93.11

73.77
85.41
84.74
82.58
82.90
83.94
-
84.12
68.06
78.49
95.97

84.84
89.18
87.92
87.46
87.13
87.00
-
87.86
73.6
83.03
97.61

50.93
68.49
62.87
78.29
81.22
66.98
85.56
75.42
66.15
64.61
79.76

65.84
77.55
69.52
84.26
86.48
79.04
90.07
81.65
71.33
71.8
85.16

65.55
80.28
79.5
77.44
81.33
70.51
88.94
82.04
69.66
73.22
87.13

77.73
85.65
85.81
84.71
86.58
79.55
89.47
86.27
76.28
78.24
90.62

51.96
55.47
56.32
57.34
62.79
46.99
69.72
66.91
54.35
63.54
71.48

71.94
69.61
69.3
73.17
76.91
70.29
81.29
79.27
66.12
76.66
82.83

Table 10: Table Recognition Performance. MinerU2.5 achieves SOTA performance on most benchmarks
among TEDS and TEDS-S metrics, and the remaining ones are also generally competitive with the
SOTA. (CCOCR and OCRBench v2 are OCR evaluation benchmarks, we only select the subsets that
contain tables. PubTabNet and FinTabNet have a large number of images, so we have not evaluate
Gemini-2.5 Pro on them.).

majority of individual element types. This consistent superiority confirms that the PageIoU metric
provides a robust basis for comparison, effectively capturing model performances independent of
annotation inconsistencies.

5.2.2 Table Recognition

We evaluate representative methods, covering traditional table recognition methods, general multi-
modal large models and document parsing models, on five table recognition benchmarks as shown in
Table 10. Below is an introduction to each benchmark:

• PubTabNet [62] is the first large-scale table recognition dataset that provides annotations (in
HTML format) of table images, captured from scientific articles. PubTabNet contains 9k tables in
its test set.

• FinTabNet [61] is a dataset containing tables from the annual reports of 500 companies. The
major challenge of this benchmark is that financial tables largely differ from scientific and
government document tables in that the former has fewer graphical lines, larger gaps within
each table, and more color variations. FinTabNet contains 10k tables in its test set.

• CC-OCR [53] and OCRBench v2 [12] are both designed to evaluate the OCR capabilities of
multimodal large models and contain several OCR tasks. We only retain the data related to
document recognition and those images that include tables. After filtering, CC-OCR remains 300
images and OCRBench v2 remains 700 images.

• In-house TR Benchmark. To better evaluate the table recognition accuracy of different methods,
we considering various table attributes such as the number of table rows and columns, the
number of merged cells, the length of the table, the length of the cell content, the type of cell
content, the line style of the table, and construct a very diverse evaluation set, which contains
approximately 500 tables.

MinerU2.5 achieves SOTA performance on most benchmarks, and shows competitive results with
the SOTA on the remaining ones. Specifically, for PubTabNet, Rapidtable [36] achieves the best
performance in the TEDS-S metric, while dots.ocr [37] excel in the TEDS metric. Meanwhile, despite
using only 20% of the PubTabNet training set, MinerU2.5 still demonstrate comparable results, coming
second and third in TEDS and TEDS-S, respectively. For FinTabNet, MinerU2.5 achieves the best result

23

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Method

Public Dataset

In-house Dataset

CPE HWE SCE SPE LaTeX-80MM Chinese Fuzzy Math Complex

UniMERNet∗ [45]
98.2
PP-Formula plus-L [21] 98.2
89.2
Gemini-2.5-flash [7]
88.9
Qwen2.5-VL-72B [3]
82.7
GPT-4o [1]
91.7
InternVL3.5-241B [49]
86.8
dots.ocr [37]
96.6
MinerU2.5

96.5
94.7
90.0
91.8
85.9
93.2
90.5
94.4

95.4 99.2
95.7 99.2
85.1 97.5
95.5 96.2
87.8 96.7
95.1 97.8
94.7 97.5
96.4 98.4

83.9
85.9
78.7
83.4
73.4
86.9
81.8
90.6

84.0
84.0
88.1
90.8
88.3
82.7
74.4
90.7

84.3
86.5
89.4
86.7
85.0
90.3
86.2
92.6

67.9
76.5
80.1
81.4
78.6
82.0
77.4
82.2

Table 11: Formula Recognition Performance (CDM metric used for evaluation). MinerU2.5 achieves
4 SOTA results and one second-best result across 7 benchmarks. Latex-80MM denotes the matrix
benchmark of Latex-80M dataset. ∗ indicates that the UniMERNet results are based on an improved
version compared to the publicly available open-source implementation.

and outperform other methods by a significant margin, this could be mainly credited to the large-scale
high-quality table data we extracted from financial reports for training. On CC-OCR benchmark,
MinerU2.5 came third after Gemini-2.5 Pro and Qwen2.5-VL-72B. On OCRBench v2 benchmark,
MinerU2.5’s performance is competitive to that of Gemini-2.5 Pro, and it significantly outperform other
methods. On the diverse In-house TR Benchmark, MinerU2.5 and Gemini-2.5 Pro both significantly
outperform other methods, with MinerU2.5 achieving a slight advantage over Gemini-2.5 Pro.

5.2.3 Formula Recognition

For formula recognition, comparison models include various approaches, covering specialized formula
recognition models, document parsing models, and general vision-language models. The evaluation
datasets consist of the following:

• UniMER-Test [45] is a comprehensive evaluation dataset for general formula recognition. Tar-
geted at real-world formula recognition across various scenarios, UniMER-Test includes four
subsets: CPE (complex printed equations), HWE (handwritten equations), SPE (screen printed
equations), and SCE (simple printed equations).

• LaTeX-80MM is a matrix subset of LaTeX-80M5, featuring intricate mathematical structures

encompassing matrices, conditional expressions, and nested combinations.

• In-house dataset consists of the following subsets: (1) Chinese, targeted at evaluation on real-
world document equations which contain Chinese characters. (2) Fuzzy math, which focuses on
authentic mathematics textbooks and exam documents characterized by compromised visual
quality due to factors like blur, degeneration, watermarks, and so on. (3) Complex, an extremely
difficult dataset aimed at assessing the ability of converting the most complex mathematical
formulas to LaTeX codes.

Results are shown in Table 11 and the CDM [47] metric is used for evaluation. Across all seven
evaluation datasets, MinerU2.5 achieves the best results in four datasets and one second-best result,
demonstrating SOTA formula recognition capabilities. Specifically, on public datasets, MinerU2.5
achieves best CDM results of 96.4 on SCE and 90.6 on LaTeX-80MM, showcasing leading performance
in scenarios involving blurred screenshots and complex matrices. Besides, on CPE, HWE, and SPE,
while being slightly outperformed by specialized formula recognition models, MinerU2.5 still deliver

5https://github.com/OleehyO/TexTeller

24

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

comparable performance. On in-house evaluation datasets, MinerU2.5’s performance in Chinese text
recognition is on par with Qwen2.5-VL-72B, leading to a second-place result of 90.6. Meanwhile,
MinerU2.5 achieves the best results on both the real-world mathematic documents (Fuzzy Math) and
extremely hard formula recognition (Complex).

6 Conclusion

In this paper, we present MinerU2.5, a 1.2B-parameter vision-language model that achieves a new
state-of-the-art in efficient document parsing through its innovative decoupled, coarse-to-fine strategy.
By separating global layout analysis from local recognition, it delivers unprecedented accuracy in
a lightweight model, effectively resolving the trade-off between performance and cost. Beyond its
standalone capabilities, the primary significance of MinerU2.5 lies in its role as a foundational tool for
the LLM era. Its ability to rapidly convert vast, unstructured document collections into clean, structured
data is invaluable for curating high-quality pre-training corpora. Furthermore, by preserving the
semantic integrity of tables, formulas, and layouts, it is poised to significantly enhance the quality
and reliability of Retrieval-Augmented Generation (RAG) systems, unlocking the vast knowledge
contained within complex documents for next-generation AI applications.

25

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

References
[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo
Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint
arXiv:2303.08774, 2023.

[2] Haoli Bai, Zhiguang Liu, Xiaojun Meng, Wentao Li, Shuang Liu, Nian Xie, Rongfu Zheng, Liangwei Wang,
Lu Hou, Jiansheng Wei, et al. Wukong-reader: Multi-modal pre-training for fine-grained visual document
understanding. arXiv preprint arXiv:2212.09621, 2022.

[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang,

Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

[4] Lukas Blecher, Guillem Cucurull, Thomas Scialom, and Robert Stojnic. Nougat: Neural optical understanding

for academic documents. arXiv preprint arXiv:2308.13418, 2023.

[5] chatdoc com. Ocrflux. https://github.com/chatdoc-com/OCRFlux, 2025. Accessed:2025-09-25.

[6] Song Chen, Xinyu Guo, Yadong Li, Tao Zhang, Mingan Lin, Dongdong Kuang, Youwei Zhang, Lingfeng
Ming, Fengyu Zhang, Yuran Wang, et al. Ocean-ocr: Towards general ocr application via a vision-language
model. arXiv preprint arXiv:2501.15558, 2025.

[7] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel
Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning,
multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

[8] Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang,
Changda Zhou, Hongen Liu, et al. Paddleocr 3.0 technical report. arXiv preprint arXiv:2507.05595, 2025.

[9] Cheng Da, Chuwei Luo, Qi Zheng, and Cong Yao. Vision grid transformer for document layout analysis. In

Proceedings of the IEEE/CVF international conference on computer vision, pages 19462–19472, 2023.

[10] Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron,
Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n’pack: Navit, a
vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36:
2252–2274, 2023.

[11] Hao Feng, Shu Wei, Xiang Fei, Wei Shi, Yingdong Han, Lei Liao, Jinghui Lu, Binghong Wu, Qi Liu, Chun-
hui Lin, et al. Dolphin: Document image parsing via heterogeneous anchor prompting. arXiv preprint
arXiv:2505.14059, 2025.

[12] Ling Fu, Zhebin Kuang, Jiajun Song, Mingxin Huang, Biao Yang, Yuzhe Li, Linghao Zhu, Qidi Luo, Xinyu
Wang, Hao Lu, et al. Ocrbench v2: An improved benchmark for evaluating large multimodal models on
visual text localization and reasoning. arXiv preprint arXiv:2501.00321, 2024.

[13] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu

Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025.

[14] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai
with unified text and image masking. In Proceedings of the 30th ACM international conference on multimedia,
pages 4083–4091, 2022.

[15] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok
Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer.
In European Conference on Computer Vision, pages 498–517. Springer, 2022.

[16] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gon-
zalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with
pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

[17] Zhang Li, Yuliang Liu, Qiang Liu, Zhiyin Ma, Ziyang Zhang, Shuo Zhang, Zidun Guo, Jiarui Zhang, Xinyu
Wang, and Xiang Bai. Monkeyocr: Document parsing with a structure-recognition-relation triplet paradigm.
arXiv preprint arXiv:2506.05218, 2025.

26

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

[18] Haofu Liao, Aruni RoyChowdhury, Weijian Li, Ankan Bansal, Yuting Zhang, Zhuowen Tu, Ravi Kumar
Satzoda, R Manmatha, and Vijay Mahadevan. Doctr: Document transformer for structured information
extraction in documents. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages
19584–19594, 2023.

[19] Demiao Lin. Revolutionizing retrieval-augmented generation with enhanced pdf structure recognition. arXiv

preprint arXiv:2401.12599, 2024.

[20] Chaohu Liu, Kun Yin, Haoyu Cao, Xinghua Jiang, Xin Li, Yinsong Liu, Deqiang Jiang, Xing Sun, and Linli
Xu. Hrvda: High-resolution visual document assistant. In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pages 15534–15545, 2024.

[21] Hongen Liu, Cheng Cui, Yuning Du, Yi Liu, and Gang Pan. Pp-formulanet: Bridging accuracy and efficiency

in advanced formula recognition. arXiv preprint arXiv:2503.18382, 2025.

[22] Yuan Liu, Zhongyin Zhao, Le Tian, Haicheng Wang, Xubing Ye, Yangxiu You, Zilin Yu, Chuhan Wu, Xiao
Zhou, Yang Yu, et al. Points-reader: Distillation-free adaptation of vision-language models for document
conversion. arXiv preprint arXiv:2509.01215, 2025.

[23] Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, and Xiang Bai. Textmonkey: An
ocr-free large multimodal model for understanding document. arXiv preprint arXiv:2403.04473, 2024.

[24] Nikolaos Livathinos, Christoph Auer, Maksym Lysak, Ahmed Nassar, Michele Dolfi, Panos Vagenas, Ce-
sar Berrospi Ramis, Matteo Omenetti, Kasper Dinkla, Yusik Kim, et al. Docling: An efficient open-source
toolkit for ai-driven document conversion. arXiv preprint arXiv:2501.17887, 2025.

[25] Maksym Lysak, Ahmed Nassar, Nikolaos Livathinos, Christoph Auer, and Peter Staar. Optimized table
tokenization for table structure recognition. In International Conference on Document Analysis and Recognition,
pages 37–50. Springer, 2023.

[26] Souvik Mandalm. Nanonets-ocr-s. https://nanonets.com/research/nanonets-ocr-s/, 2025. Accessed:2025-

09-25.

[27] Mathpix. Mathpix. https://mathpix.com/, 2025. Accessed:2025-09-25.

[28] Ahmed Nassar, Andres Marafioti, Matteo Omenetti, Maksym Lysak, Nikolaos Livathinos, Christoph Auer,
Lucas Morin, Rafael Teixeira de Lima, Yusik Kim, A Said Gurbuz, et al. Smoldocling: An ultra-compact
vision-language model for end-to-end multi-modal document conversion. arXiv preprint arXiv:2503.11576,
2025.

[29] Junbo Niu, Yuanhong Zheng, Ziyang Miao, Hejun Dong, Chunjiang Ge, Hao Liang, Ma Lu, Bohan Zeng,
Qiahao Zheng, Conghui He, et al. Native visual understanding: Resolving resolution dilemmas in vision-
language models. arXiv preprint arXiv:2506.12776, 2025.

[30] OpenDataLab. Pdf-extract-kit. https://github.com/opendatalab/PDF-Extract-Kit, 2025. Accessed:2025-

09-25.

[31] Linke Ouyang, Yuan Qu, Hongbin Zhou, Jiawei Zhu, Rui Zhang, Qunshu Lin, Bin Wang, Zhiyuan Zhao,
Man Jiang, Xiaomeng Zhao, et al. Omnidocbench: Benchmarking diverse pdf document parsing with
comprehensive annotations. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages
24838–24848, 2025.

[32] Vik Paruchuri. Marker. https://github.com/datalab-to/marker, 2025. Accessed:2025-09-25.

[33] Vikas Paruchuri and Datalab Team. Surya: A lightweight document ocr and analysis toolkit. https:

//github.com/VikParuchuri/surya, 2025. Accessed:2025-09-25.

[34] Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S Nassar, and Peter Staar. Doclaynet: A large
In Proceedings of the 28th ACM SIGKDD

human-annotated dataset for document-layout segmentation.
conference on knowledge discovery and data mining, pages 3743–3751, 2022.

[35] Jake Poznanski, Aman Rangapur, Jon Borchardt, Jason Dunkelberger, Regan Huff, Daniel Lin, Christopher
Wilhelm, Kyle Lo, and Luca Soldaini. olmocr: Unlocking trillions of tokens in pdfs with vision language
models. arXiv preprint arXiv:2502.18443, 2025.

27

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

[36] RapidAI. Rapid table. https://github.com/RapidAI/RapidTable, 2024. Accessed: 2025-9-25.

[37] rednote. dots.ocr: Multilingual document layout parsing in a single vision-language model. https://github.

com/rednote-hilab/dots.ocr, 2025. Accessed:2025-09-25.

[38] Wenzhe Shi, Jose Caballero, Ferenc Husz´ar, Johannes Totz, Andrew P Aitken, Rob Bishop, Daniel Rueckert,
and Zehan Wang. Real-time single image and video super-resolution using an efficient sub-pixel convolu-
tional neural network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages
1874–1883, 2016.

[39] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced

transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

[40] Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang,
and Mohit Bansal. Unifying vision, text, and layout for universal document processing. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, pages 19254–19264, 2023.

[41] Mistral AI Team. Mistral-ocr.

https://mistral.ai/news/mistral-ocr?utm source=ai-bot.cn, 2025.

Accessed:2025-09-25.

[42] Qwen Team. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

[43] Jianqiang Wan, Sibo Song, Wenwen Yu, Yuliang Liu, Wenqing Cheng, Fei Huang, Xiang Bai, Cong Yao,
and Zhibo Yang. Omniparser: A unified framework for text spotting key information extraction and
table recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages
15641–15653, 2024.

[44] Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, et al. Yolov10: Real-time end-to-end

object detection. Advances in Neural Information Processing Systems, 37:107984–108011, 2024.

[45] Bin Wang, Zhuangcheng Gu, Guang Liang, Chao Xu, Bo Zhang, Botian Shi, and Conghui He. Unimernet: A
universal network for real-world mathematical expression recognition. arXiv preprint arXiv:2404.15254, 2024.

[46] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu,
Fukai Shang, et al. Mineru: An open-source solution for precise document content extraction. arXiv preprint
arXiv:2409.18839, 2024.

[47] Bin Wang, Fan Wu, Linke Ouyang, Zhuangcheng Gu, Rui Zhang, Renqiu Xia, Botian Shi, Bo Zhang, and
Conghui He.
Image over text: Transforming formula recognition evaluation with character detection
matching. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19681–19690, 2025.

[48] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang,
Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution.
arXiv preprint arXiv:2409.12191, 2024.

[49] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin
Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility,
reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

[50] Zilong Wang, Yiheng Xu, Lei Cui, Jingbo Shang, and Furu Wei. Layoutreader: Pre-training of text and layout

for reading order detection. arXiv preprint arXiv:2108.11591, 2021.

[51] Zilong Wang, Yichao Zhou, Wei Wei, Chen-Yu Lee, and Sandeep Tata. Vrdu: A benchmark for visually-rich
document understanding. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data
Mining, pages 5184–5193, 2023.

[52] Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao,
Jianjian Sun, Yuang Peng, et al. General ocr theory: Towards ocr-2.0 via a unified end-to-end model. arXiv
preprint arXiv:2409.01704, 2024.

[53] Zhibo Yang, Jun Tang, Zhaohai Li, Pengfei Wang, Jianqiang Wan, Humen Zhong, Xuejing Liu, Mingkun
Yang, Peng Wang, Shuai Bai, et al. Cc-ocr: A comprehensive and challenging ocr benchmark for evaluating
large multimodal models in literacy. arXiv preprint arXiv:2412.02210, 2024.

28

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

[54] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao,
Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.

[55] Tianyu Yu, Zefan Wang, Chongyi Wang, Fuwei Huang, Wenshuo Ma, Zhihui He, Tianchi Cai, Weize Chen,
Yuxiang Huang, Yuanqian Zhao, Bokai Xu, Junbo Cui, Yingjing Xu, Liqing Ruan, Luoyuan Zhang, Hanyu
Liu, Jingkun Tang, Hongyuan Liu, Qining Guo, Wenhao Hu, Bingxiang He, Jie Zhou, Jie Cai, Ji Qi, Zonghao
Guo, Chi Chen, Guoyang Zeng, Yuxuan Li, Ganqu Cui, Ning Ding, Xu Han, Yuan Yao, Zhiyuan Liu, and
Maosong Sun. Minicpm-v 4.5: Cooking efficient mllms via architecture, data, and training recipe. arXiv
preprint arXiv:2509.18154, 2025.

[56] Junyuan Zhang, Qintong Zhang, Bin Wang, Linke Ouyang, Zichen Wen, Ying Li, Ka-Ho Chow, Conghui
He, and Wentao Zhang. Ocr hinders rag: Evaluating the cascading impact of ocr on retrieval-augmented
generation. arXiv preprint arXiv:2412.02592, 2024.

[57] Qintong Zhang, Bin Wang, Victor Shea-Jay Huang, Junyuan Zhang, Zhengren Wang, Hao Liang, Conghui
He, and Wentao Zhang. Document parsing unveiled: Techniques, challenges, and prospects for structured
information extraction. arXiv preprint arXiv:2410.21169, 2024.

[58] Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng, Fangcheng Fu, Ling Yang, Wentao
Zhang, Jie Jiang, and Bin Cui. Retrieval-augmented generation for ai-generated content: A survey. arXiv
preprint arXiv:2402.19473, 2024.

[59] Zhiyuan Zhao, Hengrui Kang, Bin Wang, and Conghui He. Doclayout-yolo: Enhancing document layout anal-
ysis through diverse synthetic data and global-to-local adaptive perception. arXiv preprint arXiv:2410.12628,
2024.

[60] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Livia Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao,
Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. Sglang: Efficient execution of structured language
model programs. Advances in neural information processing systems, 37:62557–62583, 2024.

[61] Xinyi Zheng, Douglas Burdick, Lucian Popa, Xu Zhong, and Nancy Xin Ru Wang. Global table extractor (gte):
A framework for joint table identification and cell structure recognition using visual context. In Proceedings of
the IEEE/CVF winter conference on applications of computer vision, pages 697–706, 2021.

[62] Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno Yepes. Image-based table recognition: data, model,

and evaluation. In European conference on computer vision, pages 564–580. Springer, 2020.

[63] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan,
Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source
multimodal models. arXiv preprint arXiv:2504.10479, 2025.

29

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Appendix

A Qualitative examples

This section presents qualitative examples illustrating the capabilities of the MinerU2.5 through docu-
ment parsing outputs generated for various pages. This section is structured as follows: Section A.1
illustrates the MinerU2.5’s performance on Document Parsing, Table Recognition and Formula Recog-
nition among all types of documents. Section A.2 showcases specific attribute pages with improved
performance. Section A.3 demonstrates MinerU2.5’s performance on some complex pages compared
to other models.

Examples demonstrating the Document Parsing performance among PDF types are provided in Fig-
ures 8 to 10, including Academic literature, Books, Textbooks, Research Report, Financial Report,
Slides, Exam Paper, Note, Newspaper and Magazine.

Table Recognition performance among various types of tables is demonstrated in Figures 11 and 12,
including the photograph of the table, table with colorful background, table with formula, table with
empty cells, handwritten table, large table, rotated table, no-line table, three-line table, and full-line
table.

The performance of Formula Recognition among types of formulas is demonstrated in Figures 13
and 14, including formula with background, formula with Chinese, formula with matrix, formula
with condition and nested condition, handwritten formula, blurred formula, multi-column formula,
degradation formula.

Figures 15 to 18 demonstrate that MinerU2.5’s document parsing ability improved when encounter
rotated tables, table with merged cells, formula with Chinese and multi-line and complex formula,
comparing with previous version (MinerU2-VLM, MinerU2-pipeline). Moreover, MinerU2.5 achieves
finer bounding bbox in layout detection and performs better on watermark pages than previous
version, as illustrated in Figures 19 and 20.

MinerU2.5 achieves outstanding performance in scenarios involving PDF pages with complex elements,
and its performance is relatively better compared to existing state-of-the-art models.

Figures 21 to 26 showcase the scenarios with complex tables in the page, including full-page table,
content dense table, colorful table with amounts of empty cells, a tightly-arranged multiple table, table
with irregular merged cells, a table without lines. MinerU2.5 can achieve better parsing outputs on
these pages, while other models encounter errors such as table structure error, table structure lost,
table content lost and table split error.

Figures 27 to 29 illustrates the performance of MinerU2.5 in the page with nested conditional expres-
sions, complex matrix and nested matrix compared to other SOTA models, MinerU2.5 can correctly
parse the complex formula while others might generate wrong outputs.

Figures 30 to 32 shows MinerU2.5’s outstanding performance in pages with complex layout, e.g.,
alternating texts and images, with very-few frame tables, and pages with watermark compared with
others.

30

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

A.1 Overview

A.1.1 Among PDF types

Figure 8: The Layout and rendered markdown output for Academic literature, Books, Textbooks.

31

Academic LiteratureBooksTextbooksMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 9: The Layout and rendered markdown output for Research Report, Financial Report, Slides
and Exam Paper.

32

Research ReportFinancial ReportPPT & Exam PaperMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 10: The Layout and rendered markdown output for Note, Newspaper and Magazine.

33

NoteNewspaperMagazineMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

A.1.2 Among Table types

Figure 11: The rendered outputs for various types of Tables.

34

Handwritten TableColorful TablePhotograph of the TableLarge TableMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 12: The rendered outputs for various types of Tables.

35

Table with FormulaBorderless TableThree-Line TableTable with Empty CellRotated TableFully-Lined TableMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

A.1.3 Among Formula types

Figure 13: The rendered outputs for various types of Formulas.

36

Formula with BackgroudFormula with ChineseHandwritten FormulaMatricesBlurred FormulaMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 14: The rendered outputs for various types of Formulas.

37

Multiline Formula Nested MatricesConditional ExpressionsNested Conditional ExpressionsDegradation FormulaMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

A.2 Compare to Previous Versions

A.2.1 Table

Figure 15: Compare with Previous Version, MinerU2.5 performs better in rotated tables.

38

MinerU2.5ImageMinerU2-VLMTable RepeatTable Structure ErrorMinerU2-pipelineMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 16: Compare with Previous Version, MinerU2.5 performs better in tables with merged cells.

39

MinerU2.5ImageMinerU2-VLMMinerU2-pipelineTable Structure ErrorTable Structure ErrorTable Content LostMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

A.2.2 Formula

Figure 17: Compare with Previous Version, MinerU2.5 performs better in Formula with Chinese.

40

Image#DFAE36MinerU2.5Missing Formula and TextMissing Formula and TextMinerU2-VLMMinerU2-pipelineMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 18: Compare with Previous Version, MinerU2.5 performs better in multi-lines and complex
Formula.

41

Formula Recognition ErrorFormula Rendering ErrorFormula Rendering ErrorMinerU2-VLMMinerU2-pipelineImageMinerU2.5MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

A.2.3 Layout&OCR

Figure 19: Compare with Previous Version, MinerU2.5 achieve finer layout detection.

42

和自己历史版本对比：粒度更细，格式更好ImageMinerU2.5MinerU2-VLMMinerU2-pipelineMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 20: Compare with Previous Version, MinerU2.5 achieve fewer detection omissions in watermark
page.

43

MinerU2-pipelinePP-StructureV3优势：包含水印页面检测遗漏较少ImageMinerU2.5MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

A.3 Compare with Others

A.3.1 Table

Figure 21: Compare with others in Full-page table.

44

Qwen2.5-VL-72BMinerU2.5dots.ocrImageTable Structure ErrorTable Content LostTable Structure ErrorMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 22: Compare with others in content dense table.

45

Qwen2.5-VL-72BTable Structure ErrorMinerU2.5dots.ocrTable Structure ErrorImageMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 23: Compare with others in Colored table with many empty cells.

46

MinerU2.5dots.ocrImageTable Structure ErrorTable Content LostTable Structure ErrorGemini-2.5-ProMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 24: Compare with others in Tightly arranged multiple tables.

47

MinerU2.5dots.ocrImageGemini-2.5-ProTable Split ErrorTable Split ErrorMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 25: Compare with others in Table with irregular merged cells.

48

MinerU2.5Imagedots.ocrTable Structure ErrorGemini-2.5-ProTable Structure ErrorMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 26: Compare with others in Tables with No Frame.

49

MinerU2.5Imagedots.ocrGemini-2.5-ProTable Structure LostTable Structure LostMinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

A.3.2 Formula

Figure 27: Compare with others in Nested conditional expressions.

50

Gemini-2.5-Prodots.ocr条件表达式嵌套Formula Recognition ErrorFormula Recognition ErrorImageMinerU2.5MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 28: Compare with others in Complex matrix.

51

Gemini-2.5-Prodots.ocr困难矩阵Formula Recognition ErrorFormula Rendering ErrorFormula Recognition ErrorFormula Rendering ErrorMissing Formula ErrorImageMinerU2.5MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 29: Compare with others in Nested matrix.

52

PP-StructureV3dots.ocr嵌套矩阵Formula Recognition ErrorFormula Rendering ErrorFormula Recognition ErrorFormula Rendering ErrorImageMinerU2.5MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

A.3.3 Layout&OCR

Figure 30: Compare with others in Academic literature with alternating text and images.

53

PP-StructureV3MonkeyOCR-pro-1.2B优势：复杂排版检测效果很好ImageMinerU2.5MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 31: Compare with others in Financial Report with Few Frame Tables.

54

PP-StructureV3MonkeyOCR-pro-1.2B优势：财报研报的少线表检测和拆分粒度比较好ImageMinerU2.5MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

Figure 32: Compare with others in Textbooks with watermarks.

55

PP-StructureV3MonkeyOCR-pro-1.2B优势：背景水印干扰小ImageMinerU2.5MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

B Prompt Details

Here, we provide a detailed description of the different prompts used during the two-stage inference
of MinerU2.5, along with their corresponding output formats.

B.1 Layout Detection

The layout detection output will include the relative coordinates, category, and rotation direction of
each element in the document. Each element will be output in sequence, ensuring traceability for all
layout data. The input image will be resized to a resolution of 1036 × 1036.

Output format:

• Box Coordinates: x1, y1, x2, y2

• Document Element Category: title, text, image, etc.

• Rotation Direction: up, down, left, right

Example:

<| box_start | >100 200 300 400 <| box_end | > <| ref_start |> title <| ref_end | > <| rotate_up |>
<| box_start | >400 500 600 700 <| box_end | > <| ref_start |> text <| ref_end | > <| rotate_up |>

B.2 Text Recognition

The output will contain the recognized text results. The input image will retain its native resolution;
however, the number of image tokens will be limited to the range of 4 to 2048. If this limit is exceeded,
the image will be scaled accordingly.

Output format:

• OCR Results: The raw OCR output

Example:

The results of the analyses of the uncertainty of the field data and related assumptions are

shown in Figs 13 and 14.

B.3 Formula Recognition

Any formulas found in the image will be extracted and converted into LaTeX format. The input image
will retain its native resolution; however, the number of image tokens will be limited to the range of 4
to 2048. If this limit is exceeded, the image will be scaled accordingly.

Output format:

• LaTeX Format: The LaTeX representation of the formula

Example:

\[
\ hat {F} = \ operatorname { Concat }\ left (\ left [ F_ {1} , F_ {2} , \ dots , F_ {n} \ right ]\ right ) \ tag

{2}

\]

\[
M = \ sigma \ bigl ( \ mathrm { GELU }(\ mathrm { BN }(\ mathrm { Conv }_{ gate }(\ hat {F }) )) \ bigr ) \ tag {3}
\]

56

MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing

B.4 Table Recognition

The output will include the recognized tables, structured in an OTSL (Open Table Structure Language)
format for easy data processing. The input image will retain its native resolution; however, the number
of image tokens will be limited to the range of 4 to 2048. If this limit is exceeded, the image will be
scaled accordingly.

Output format:

• OTSL Format: The table represented in OTSL format

Example:

<fcel > Site < fcel >Cl < fcel > NO3 < fcel > SO4 < fcel >Na < fcel >Ca < fcel >K < fcel >Mg < fcel > NH4 < fcel > References

<nl >

<fcel > Cl dominance sites < lcel >< lcel >< lcel >< lcel >< lcel >< lcel >< lcel >< lcel >< lcel ><nl >
<fcel > Comba < fcel >109.8 < fcel >12.1 < fcel >23.3 < fcel >86.8 < fcel >43.4 < fcel >4.8 < fcel >15.1 < fcel >13.2 <

fcel > Present study <nl >

<fcel > Alibagh < fcel >236 < fcel >9 < fcel >36 < fcel >220 < fcel >46 < fcel >5 < fcel >64 < fcel >8 < fcel > Naik et al

. (2002) <nl >

<fcel > Goa < fcel >113.4 < fcel >5.5 < fcel >27.4 < fcel >97.2 < fcel >41.5 < fcel >2.5 < fcel >24.5 < fcel >5.5 < fcel

> Parashar et al . (2001) <nl >

<fcel > Bombay < fcel >138 < fcel >-< fcel >10 < fcel >115 < fcel >36 < fcel >3.6 < fcel >24 < fcel >-< fcel > Sequeira

(1976) <nl >

<fcel > Na dominance sites < lcel >< lcel >< lcel >< lcel >< lcel >< lcel >< lcel >< lcel >< lcel ><nl >
<fcel > Colaba < fcel >171 < fcel >34 < fcel >52 < fcel >179 < fcel >133 < fcel >6 < fcel >59 < fcel >12 < fcel > Naik et

al . (2002) <nl >

<fcel > Silent Valley < fcel >43.0 < fcel >21.0 < fcel >20.0 < fcel >46.0 < fcel >43.0 < fcel >4.0 < fcel >14.0 <

fcel >3.0 < fcel > Rao et al . (1995) <nl >

<fcel > Chembur < fcel >164.5 < fcel >29.5 < fcel >70.4 < fcel >168.2 < fcel >89.5 < fcel >6.9 < fcel >36.5 < fcel

>41.1 < fcel > Khemani et al . (1994) <nl >

<fcel > Bhubaneswar < fcel >18 < fcel >10 < fcel >19.1 < fcel >15 < fcel >20.2 < fcel >1.8 < fcel >5.2 < fcel >18.7 <

fcel > Das et al . (2005) <nl >

57

