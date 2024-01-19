---
permalink: /
title: "Homepage"
excerpt: "About"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

<span class='anchor' id='about-me'></span>

# About

I was a Principal Researcher at Microsoft Research (MSR) in the Natural Language Computing group, working with Dr. [Furu Wei](https://www.microsoft.com/en-us/research/people/fuwei/) and Dr. [Ming Zhou](https://www.langboat.com/en/about/resume). My research encompassed a broad range of topics in Natural Language Processing and Machine Learning, with a particular focus on practical technologies for generative language models as well as efficient modeling and inference. I have published over 40 papers at leading Artificial Intelligence conferences such as ACL, NeurIPS, ICLR, EMNLP, and NAACL. I have served as a (senior) area chair and program committee member for these conferences as well as an action editor for top journals like TACL.

Beyond academia, I was responsible for the development and deployment of advanced generative models within Microsoft 365 and Office products. I architected and delivered state-of-the-art generative models with [highly optimized efficiency](https://www.microsoft.com/en-us/research/blog/achieving-zero-cogs-with-microsoft-editor-neural-grammar-checker/) that stems from my proposed novel approaches for low-cost GenAI (e.g., [Aggressive Decoding](https://arxiv.org/abs/2106.04970), [Speculative Decoding](https://arxiv.org/abs/2203.16487) and [EdgeFormer](https://arxiv.org/abs/2202.07959)), enabling the large-scale deployment of these models and allowing them to become the most visible and used writing assistance features in Microsoft Editor that transform the writing experience for millions of M365 and Office users.

Prior to joining Microsoft, I earned my Ph.D. from Peking University in 2017, advised by Prof. [Zhifang Sui](https://cs.pku.edu.cn/info/1226/2014.htm). During my doctoral studies, I also had the privilege of working with Prof. [Heng Ji](https://cs.illinois.edu/about/people/department-faculty/hengji) during my visit to Rensselaer Polytechnic Institute (RPI), and with Dr. [Jun Suzuki](https://www.fai.cds.tohoku.ac.jp/members/js/) and Dr. [Masaaki Nagata](https://www.rd.ntt/e/organization/researcher/superior/s_011.html) during my visit to NTT Communication Science Laboratories.


<span class='anchor' id='publications'></span>

# Publications (<sup>\#</sup>: students I mentored; <sup>\*</sup>: equal contributions; <sup>\+</sup>: Corresponding author)

## Preprint

- [23.11] [ALYMPICS: Language Agents Meet Game Theory](https://arxiv.org/abs/2311.03220)  

  Shaoguang Mao, Yuzhe Cai, Yan Xia, Wenshan Wu, Xun Wang, Fengyi Wang, **Tao Ge**, Furu Wei

- [23.09] [SCALE: Synergized Collaboration of Asymmetric Language Translation Engines](https://arxiv.org/abs/2309.17061)  

  Xin Cheng<sup>\#</sup>, Xun Wang, **Tao Ge**, Si-Qing Chen, Furu Wei, Dongyan Zhao, Rui Yan

- [23.07] [Unleashing Cognitive Synergy in Large Language Models: A Task-Solving Agent through Multi-Persona Self-Collaboration](https://arxiv.org/abs/2307.05300)  

  Zhenhailong Wang<sup>\#</sup>, Shaoguang Mao, Wenshan Wu, **Tao Ge**, Furu Wei, Heng Ji

- [23.04] [Low-code LLM: Visual Programming over LLMs](https://arxiv.org/abs/2304.08103)

  Yuzhe Cai, Shaoguang Mao, Wenshan Wu, Zehua Wang, Yaobo Liang, **Tao Ge**, Chenfei Wu, Wang You, Ting Song, Yan Xia, Jonathan Tien, Nan Duan

- [23.04] [Inference with Reference: Lossless Acceleration of Large Language Models](https://arxiv.org/abs/2304.04487)  

  Nan Yang, **Tao Ge**, Liang Wang, Binxing Jiao, Daxin Jiang, Linjun Yang, Rangan Majumder, Furu Wei

- [23.03] [Semiparametric Language Models Are Scalable Continual Learners](https://arxiv.org/abs/2303.01421)  

  Guangyue Peng<sup>\#</sup>, **Tao Ge<sup>\+</sup>**, Si-Qing Chen, Furu Wei, Houfeng Wang

## Tech Report
- [22.05] [Lossless Acceleration for Seq2seq Generation with Aggressive Decoding](https://arxiv.org/abs/2205.10350) (an earlier tech report of my proposed Speculative Decoding)

  **Tao Ge**, Heming Xia, Xin Sun, Si-Qing Chen, Furu Wei

- [18.07] [Reaching Human-level Performance in Automatic Grammatical Error Correction: An Empirical Study](https://arxiv.org/abs/1807.01270)  

  **Tao Ge**, Furu Wei, Ming Zhou


## Peer-reviewed

- **In-context Autoencoder for Context Compression in a Large Language Model**

  **Tao Ge<sup>\+</sup>**, Jing Hu<sup>\#</sup>, Lei Wang<sup>\#</sup>, Xun Wang, Si-Qing Chen, Furu Wei

  To appear in **ICLR 2024**

- **Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation**
  
  Heming Xia<sup>\#</sup>, **Tao Ge<sup>\*</sup><sup>\+</sup>**, Peiyi Wang, Si-Qing Chen, Furu Wei, Zhifang Sui
  
  In Findings of **EMNLP 2023** (Originally announced in March 2022: [https://arxiv.org/abs/2203.16487](https://arxiv.org/abs/2203.16487), **the first work proposing Speculative Decoding** that introduces an independent draft model to accelerate generation explicitly with the idea of speculative execution)

- **Extensible Prompts for Language Models on Zero-shot Language Style Customization**
  
  **Tao Ge<sup>\+</sup>**, Jing Hu<sup>\#</sup>, Li Dong, Shaoguang Mao, Yan Xia, Xun Wang, Si-Qing Chen, Furu Wei
  
  In **NeurIPS 2023**

- **Smart Word Suggestions for Writing Assistance**
  
  Chenshuo Wang, Shaoguang Mao, **Tao Ge**, Wenshan Wu, Xun Wang, Yan Xia, Jonathan Tien, Dongyan Zhao
  
  In Findings of **ACL 2023**

- **Enhancing Detailed Feedback to Chinese Writing Learners Using a Soft-Label Driven Approach and Tag-Aware Ranking Model**
  
  Yuzhe Cai, Shaoguang Mao, Chenshuo Wang, **Tao Ge**, Wenshan Wu, Yan Xia, Chanjin Zheng, Qiang Guan
  
  In **NLPCC 2023**

- **Overview of the NLPCC 2023 Shared Task: Chinese Essay Discourse Coherence Evaluation**
  
  Hongyi Wu, Xinshu Shen, Man Lan, Xiaopeng Bai, Yuanbin Wu, Aimin Zhou, Shaoguang Mao, **Tao Ge**, Yan Xia
  
  In **NLPCC 2023**

- **Overview of CCL23-Eval Task 8: Chinese Essay Fluency Evaluation (CEFE) Task**
  
  Xinshu Shen, Hongyi Wu, Xiaopeng Bai, Yuanbin Wu, Aimin Zhou, Shaoguang Mao, **Tao Ge**, Yan Xia
  
  In **CCL 2023**

- **EdgeFormer: A Parameter-Efficient Transformer for On-Device Seq2seq Generation**
  
  **Tao Ge**, Si-Qing Chen, Furu Wei
  
  In **EMNLP 2022**

- **Plug and Play Knowledge Distillation for kNN-LM with External Logits**
  
  Xuyang Jin<sup>\#</sup>, **Tao Ge<sup>\+</sup>**, Furu Wei
  
  In **AACL-IJCNLP 2022**

- **A Unified Strategy for Multilingual Grammatical Error Correction with Pre-trained Cross-lingual Language Model**
  
  Xin Sun<sup>\#</sup>, **Tao Ge<sup>\+</sup>**, Shuming Ma, Jingjing Li, Furu Wei, Houfeng Wang
  
  In **IJCAI 2022**

- **Text Revision by On-the-Fly Representation Optimization**
  
  Jingjing Li<sup>\#</sup>, Zichao Li, **Tao Ge**, Irwin King, Michael Lyu
  
  In **AAAI 2022**

- **Beyond Preserved Accuracy: Evaluating Loyalty and Robustness of BERT Compression**
  
  Canwen Xu<sup>\#</sup>, Wangchunshu Zhou<sup>\#</sup>, **Tao Ge<sup>\+</sup>**, Ke Xu, Julian McAuley, Furu Wei
  
  In **EMNLP 2021**

- **Improving Sequence-to-Sequence Pre-training via Sequence Span Rewriting**
  
  Wangchunshu Zhou<sup>\#</sup>, **Tao Ge<sup>\+</sup>**, Canwen Xu<sup>\#</sup>, Ke Xu, Furu Wei
  
  In **EMNLP 2021**

- **Instantaneous Grammatical Error Correction with Shallow Aggressive Decoding**
  
  Xin Sun<sup>\*</sup><sup>\#</sup>, **Tao Ge<sup>\*</sup><sup>\+</sup>**, Furu Wei, Houfeng Wang
  
  In **ACL 2021**

- **Blow the Dog Whistle: A Dataset for Cant Creation, Understanding and Decryption in Chinese**
  
  Canwen Xu<sup>\*</sup><sup>\#</sup>, Wangchunshu Zhou<sup>\*</sup><sup>\#</sup>, **Tao Ge<sup>\+</sup>**, Ke Xu, Julian McAuley, Furu Wei
  
  In **NAACL 2021**

- **BERT Loses Patience: Fast and Robust Inference with Early Exit**
  
  Wangchunshu Zhou<sup>\#</sup>, **Tao Ge<sup>\+</sup>**, Canwen Xu<sup>\#</sup>, Ke Xu, Julian McAuley, Furu Wei
  
  In **NeurIPS 2020**

- **UnihanLM: Coarse-to-Fine Chinese-Japanese Language Model Pretraining with the Unihan Database**
  
  Canwen Xu<sup>\#</sup>, **Tao Ge**, Chenliang Li, Furu Wei
  
  In **AACL 2020**

- **Improving the Efficiency of Grammatical Error Correction with Erroneous Span Detection and Correction**
  
  Mengyun Chen<sup>\#</sup><sup>\*</sup>, **Tao Ge<sup>\*</sup><sup>\+</sup>**, Xingxing Zhang, Furu Wei, Ming Zhou
  
  In **EMNLP 2020**

- **BERT-of-Theseus: Compressing BERT by Progressive Module Replacing**
  
  Canwen Xu<sup>\*</sup><sup>\#</sup>, Wangchunshu Zhou<sup>\*</sup><sup>\#</sup>, **Tao Ge<sup>\+</sup>**, Ke Xu, Julian McAuley, Furu Wei, Ming Zhou
  
  In **EMNLP 2020**

- **Pseudo-Bidirectional Decoding for Local Sequence Transduction**
  
  Wangchunshu Zhou<sup>\#</sup>, **Tao Ge**, Chang Mu, Ke Xu, Furu Wei, Ming Zhou
  
  In Findings of **EMNLP 2020**

- **Improving Grammatical Error Correction with Machine Translation Pairs**
  
  Wangchunshu Zhou<sup>\#</sup>, **Tao Ge**, Ke Xu, Furu Wei, Ming Zhou
  
  In Findings of **EMNLP 2020**

- **Scheduled DropHead: A Regularization Method for Transformer Models**
  
  Wangchunshu Zhou<sup>\#</sup>, **Tao Ge**, Furu Wei, Ming Zhou, Ke Xu
  
  In Findings of **EMNLP 2020**

- **Parallel Data Augmentation for Formality Style Transfer**
  
  Yi Zhang<sup>\#</sup>, **Tao Ge**, Xu Sun
  
  In **ACL 2020**

- **Self-Adversarial Learning with Comparative Discrimination for Text Generation**
  
  Wangchunshu Zhou<sup>\#</sup>, **Tao Ge**, Ke Xu, Furu Wei, Ming Zhou
  
  In **ICLR 2020**

- **Fact-aware Sentence Split and Rephrase with Permutation Invariant Training**
  
  Yinuo Guo<sup>\#</sup>, **Tao Ge**, Furu Wei
  
  In **AAAI 2020**

- **Bert-based Lexical Substitution**
  
  Wangchunshu Zhou<sup>\#</sup>, **Tao Ge**, Ke Xu, Furu Wei, Ming Zhou
  
  In **ACL 2019**

- **Automatic Grammatical Error Correction for Sequence-to-sequence Text Generation: An Empirical Study**
  
  **Tao Ge**, Xingxing Zhang, Furu Wei, Ming Zhou
  
  In **ACL 2019**

- **Fine-grained Coordinated Cross-lingual Text Stream Alignment for Endless Language Knowledge Acquisition**
  
  **Tao Ge**, Qing Dou, Heng Ji, Lei Cui, Baobao Chang, Zhifang Sui, Furu Wei, Ming Zhou
  
  In **EMNLP 2018**

- **Fluency Boost Learning and Inference for Neural Grammatical Error Correction**
  
  **Tao Ge**, Furu Wei, Ming Zhou
  
  In **ACL 2018**

- **EventWiki: A Knowledge Base of Major Events**
  
  **Tao Ge**, Lei Cui, Baobao Chang, Zhifang Sui, Furu Wei, Ming Zhou
  
  In **LREC 2018**

- **SeRI: A Dataset for Sub-event Relation Inference from an Encyclopedia**
  
  **Tao Ge**, Lei Cui, Baobao Chang, Zhifang Sui, Furu Wei, Ming Zhou
  
  In **NLPCC 2018**

- **Event detection with Burst Information Networks**
  
  **Tao Ge**, Lei Cui, Baobao Chang, Zhifang Sui, Ming Zhou
  
  In **COLING 2016**

- **News Stream Summarization using Burst Information Networks**
  
  **Tao Ge**, Lei Cui, Heng Ji, Baobao Chang, Sujian Li, Ming Zhou, Zhifang Sui
  
  In **EMNLP 2016**

- **Discovering Concept-level Event Associations from a Text Stream**
  
  **Tao Ge**, Lei Cui, Heng Ji, Baobao Chang, Zhifang Sui
  
  In **NLPCC 2016** (Best student paper)

- **Towards Time-aware Knowledge Graph Completion**
  
  Tingsong Jiang, Tianyu Liu, **Tao Ge**, Lei Sha, Baobao Chang, Sujian Li, Zhifang Sui
  
  In **COLING 2016**

- **Encoding Temporal Information for Time-aware Link Prediction**
  
  Tingsong Jiang, Tianyu Liu, **Tao Ge**, Lei Sha, Sujian Li, Baobao Chang, Zhifang Sui
  
  In **EMNLP 2016**

- **One Tense per Scene: Predicting Tense in Chinese Conversations**
  
  **Tao Ge**, Heng Ji, Baobao Chang, Zhifang Sui
  
  In **ACL 2015**

- **Bring you to the past: Automatic Generation of Topically Relevant Event Chronicles**
  
  **Tao Ge**, Wenzhe Pei, Heng Ji, Sujian Li, Baobao Chang, Zhifang Sui
  
  In **ACL 2015**

- **An Effective Neural Network Model for Graph-based Dependency Parsing**
  
  Wenzhe Pei, **Tao Ge**, Baobao Chang
  
  In **ACL 2015**

- **Exploiting task-oriented resources to learn word embeddings for clinical abbreviation expansion**
  
  Yue Liu, **Tao Ge**, Kusum S Mathews, Heng Ji, Deborah McGuinness
  
  In **BioNLP 2015**

- **Max-Margin Tensor Neural Network for Chinese Word Segmentation**
  
  Wenzhe Pei, **Tao Ge**, Baobao Chang
  
  In **ACL 2014**

- **A semi-supervised method for opinion target extraction**
  
  **Tao Ge**, Wenjie Li, Zhifang Sui
  
  In **WWW 2014**

- **The CIPS-SIGHAN CLP 2014 Chinese Word Segmentation Bake-off**
  
  Huiming Duan, Zhifang Sui, **Tao Ge**
  
  In **CIPS-SIGHAN 2014**

- **Exploiting Collaborative Filtering Techniques for Automatic Assessment of Student Free-text Responses**
  
  **Tao Ge**, Zhifang Sui, Baobao Chang
  
  In **CIKM 2013**

- **Event-Based Time Label Propagation for Automatic Dating of News Articles**
  
  **Tao Ge**, Baobao Chang, Sujian Li, Zhifang Sui
  
  In **EMNLP 2013***