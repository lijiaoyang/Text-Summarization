# 文本摘要发展调研
抽取式摘要能简单保证语句通顺，但对文本语义概况程度较低。生成式摘要是随着深度学习兴起才出现，存在句子欠通顺，对源文本概况程度低的问题。目前多数采用**抽取式**摘要技术。

## 1.1 传统方法
抽取式
+ Lead-3 ：抽取文章前三句作为摘要

+ TF-IDF： 一个词在一篇文章频率高，在其他文章很少出现，则值高

+ TextRank：仿照PageRank，句子是节点，使用句子间相似度，构造无向有权边。使用边上的权值迭代更新节点值，最后选取N个得分最高的节点，作为摘要。

+ LexRank：基于TF-IDF向量的余弦相似度

+ 聚类：文章的句子视为一个点，聚类完成摘要

+ LDA：多层次的贝叶斯概率模型，计算文档与主题、主题与词之间的概率分布，检测文本隐含的主题。计算段落的主题分布，然后计算各个句子的主题分布，求得两者的相似度，从中得到摘要。

## 1.2 基于神经网络

### 1.2.1 抽取式
+ 序列标注：原文每个句子打上一个二分类标签，0表示不属于摘要，1属于。把句子编码为向量，根据向量二分类。
+ 强化学习： 训练时交叉熵损失，测试时ROUGE，存在曝光偏差，强化学习去优化ROUGE分数。
+ 预训练模型

### 1.2.2 生成式
+ seq2seq： 对长文本不友好，更像是句子压缩
+ copy机制：为了解决OOV问题，采用copy机制，直接从源文本copy OOV词到生成的摘要里
+ coverage机制： 为了缓解生成重复的问题，coverage机制在解码的每一步考虑之前时间步的attention权重，结合coverage损失，避免继续考虑已经获得高权重的部分。
+ 强化学习： SeqGAN思想，通过生成模型G生成摘要，判别模型D区分真实摘要与生成摘要。强化学习更新参数。
+ 指针生成网络： 利用指针网络从原文提取上下文向量和当前解码器的隐藏状态计算生成概率，根据概率决定是拷贝源文本的单纯还是选择通过抽象语义生成。在遇到未登录词，将生成的新词加入词表，动态扩充词表。在copy时无法很好地定位关键词语
+ selective encoding： 关注重要的编码，忽略次要
+ Multi-Agents： 把很难的编码任务分解多个子任务，多个编码器分别编码，然后attention机制把这些编码融合，得到摘要
+ 预训练模型

### 1.2.3 抽取生成式摘要
+ hard方式：将抽取式模型抽取的关键句作为生成式模型的输入
+ soft方式：将抽取式模型的输出概率用来调整词语级别的权重

## 1.3 论文复现代码集合
![https://github.com/bojone/SPACES](https://github.com/bojone/SPACES)    端到端的长文本摘要模型CNN， 苏剑林，tensorflow 1.14 + keras 2.3.1 + bert4keras 0.9.7

![https://github.com/nlpyang/PreSumm](https://github.com/nlpyang/PreSumm)  code for EMNLP 2019 paper Text Summarization with Pretrained Encoders

![https://github.com/maszhongming/MatchSum](https://github.com/maszhongming/MatchSum)  Code for ACL 2020 paper: "Extractive Summarization as Text Matching"

![https://github.com/dmmiller612/bert-extractive-summarizer](https://github.com/dmmiller612/bert-extractive-summarizer)  Easy to use extractive text summarization with BERT

![https://github.com/rohithreddy024/Text-Summarizer-Pytorch](https://github.com/rohithreddy024/Text-Summarizer-Pytorch) Pytorch implementation of "A Deep Reinforced Model for Abstractive Summarization" paper and pointer generator network

![https://github.com/lancopku/superAE](https://github.com/lancopku/superAE)   Code for "Autoencoder as Assistant Supervisor: Improving Text Representation for Chinese Social Media Text Summarization"

![https://github.com/jiacheng-xu/DiscoBERT](https://github.com/jiacheng-xu/DiscoBERT ) Code for paper "Discourse-Aware Neural Extractive Text Summarization" (ACL20)

![https://github.com/yaushian/Unparalleled-Text-Summarization-using-GAN](https://github.com/yaushian/Unparalleled-Text-Summarization-using-GAN)Implementation of paper "Learning to Encode Text as Human-Readable Summaries using GAN    TensorFlow

![https://github.com/lipiji/DRGD-LCSTS](https://github.com/lipiji/DRGD-LCSTS)code for "Deep Recurrent Generative Decoder for Abstractive Text Summarization"

![https://github.com/kedz/nnsum](https://github.com/kedz/nnsum) An extractive neural network text summarization library for the EMNLP 2018 paper "Content Selection in Deep Learning Models of Summarization" 

![https://github.com/lancopku/SRB](https://github.com/lancopku/SRB)  Code for "Improving Semantic Relevance for Sequence-to-Sequence Learning of Chinese Social Media Text Summarization"  seq2seq

## 1.4 开箱即用工具包


![https://github.com/letiantian/TextRank4ZH](https://github.com/letiantian/TextRank4ZH)  textrank4zh

![https://github.com/hankcs/pyhanlp](https://github.com/hankcs/pyhanlp)  pyhanlp

![https://github.com/fxsjy/jieba](https://github.com/fxsjy/jieba) jiaba

![https://github.com/stanfordnlp/CoreNLP](https://github.com/stanfordnlp/CoreNLP)  CoreNLP

![https://github.com/isnowfy/snownlp](https://github.com/isnowfy/snownlp) snownlp

![https://github.com/yongzhuo/Macropodus](https://github.com/yongzhuo/Macropodus)
