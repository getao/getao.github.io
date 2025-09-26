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

I was a Principal Researcher at Microsoft Research (MSR) in the Natural Language Computing group, working with Dr. [Furu Wei](https://www.microsoft.com/en-us/research/people/fuwei/) and Dr. [Ming Zhou](https://www.langboat.com/en/about/resume). My research encompassed a broad range of topics in Natural Language Processing and Machine Learning, with a particular focus on practical technologies for generative language models as well as efficient modeling and inference. I have published over 50 papers at leading Artificial Intelligence conferences such as ACL, NeurIPS, ICLR, EMNLP, and NAACL. I have served as a (senior) area chair and program committee member for these conferences as well as an action editor for top journals like TACL.

Beyond research, I was responsible for the development and deployment of advanced generative models within Microsoft 365 and Office products. I architected and delivered state-of-the-art generative models with [highly optimized efficiency](https://www.microsoft.com/en-us/research/blog/achieving-zero-cogs-with-microsoft-editor-neural-grammar-checker/) that stems from my proposed novel approaches for low-cost GenAI (e.g., [Aggressive Decoding](https://arxiv.org/abs/2106.04970), [Speculative Decoding](https://arxiv.org/abs/2203.16487) and [EdgeFormer](https://arxiv.org/abs/2202.07959)), enabling the large-scale deployment of these models across various endpoints within our product suite (e.g., Word, Outlook and Edge). The innovative models I developed and rolled out handle **billions of user requests daily**, amounting to a yearly total that reaches into the **trillions**, as the most visible and used generative models in Microsoft Editor that transform the writing experience for billions of M365 and Office users with significant product impact.

Prior to joining Microsoft, I earned my Ph.D. from Peking University in 2017, advised by Prof. [Zhifang Sui](https://cs.pku.edu.cn/info/1226/2014.htm), [Baobao Chang](https://cs.pku.edu.cn/info/1237/2150.htm) and [Sujian Li](https://cs.pku.edu.cn/info/1219/1998.htm). During my doctoral studies, I also had the privilege of working with Prof. [Heng Ji](https://cs.illinois.edu/about/people/department-faculty/hengji) during my visit to Rensselaer Polytechnic Institute (RPI), and with Dr. [Jun Suzuki](https://www.fai.cds.tohoku.ac.jp/members/js/) and Dr. [Masaaki Nagata](https://www.rd.ntt/e/organization/researcher/superior/s_011.html) during my visit to NTT Communication Science Laboratories.


<span class='anchor' id='publications'></span>

# Publications (<sup>\*</sup>: equal contributions; ✉: Corresponding author)

## Preprint

- [25.01] [OpenCharacter: Training Customizable Role-Playing LLMs with Large-Scale Synthetic Personas](https://arxiv.org/abs/2501.15427)

Xiaoyang Wang, Hongming Zhang, **Tao Ge**, Wenhao Yu, Dian Yu, Dong Yu

- [24.10] [ParallelSpec: Parallel Drafter for Efficient Speculative Decoding](https://arxiv.org/pdf/2410.05589)

Zilin Xiao, Hongming Zhang, **Tao Ge**, Siru Ouyang, Vicente Ordonez, Dong Yu

- [23.04] [Inference with Reference: Lossless Acceleration of Large Language Models](https://arxiv.org/abs/2304.04487) (the innovation used in [OpenAI's Predicted Output](https://platform.openai.com/docs/guides/predicted-outputs)) 

  Nan Yang, **Tao Ge**, Liang Wang, Binxing Jiao, Daxin Jiang, Linjun Yang, Rangan Majumder, Furu Wei

## Tech Report

- [24.06] [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/pdf/2406.20094)

  **Tao Ge**, Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, Dong Yu

- [22.05] [Lossless Acceleration for Seq2seq Generation with Aggressive Decoding](https://arxiv.org/abs/2205.10350) (an earlier tech report of my proposed Speculative Decoding)

  **Tao Ge**, Heming Xia, Xin Sun, Si-Qing Chen, Furu Wei

- [18.07] [Reaching Human-level Performance in Automatic Grammatical Error Correction: An Empirical Study](https://arxiv.org/abs/1807.01270)  

  **Tao Ge**, Furu Wei, Ming Zhou


## Peer-reviewed

- ![](https://img.shields.io/badge/NeurIPS%20(Spotlight)-2025-red) [**Improving LLM General Preference Alignment via Optimistic Online Mirror Descent**](https://arxiv.org/abs/2410.13184)

  Yuheng Zhang, Dian Yu, **Tao Ge**, Linfeng Song, Zhichen Zeng, Haitao Mi, Nan Jiang, Dong Yu

- ![](https://img.shields.io/badge/EMNLP-2025-red) [**Router-Tuning: A Simple and Effective Approach for Dynamic Depth**](https://arxiv.org/abs/2410.13184)

  Shwai He, **Tao Ge**, Guoheng Sun, Bowei Tian, Xiaoyang Wang, Dong Yu

- ![](https://img.shields.io/badge/ACL-2025-red) [**Low-Bit Quantization Favors Undertrained LLMs**](https://aclanthology.org/2025.acl-long.1555/)

  Xu Ouyang, **Tao Ge**, Thomas Hartvigsen, Zhisong Zhang, Haitao Mi, Dong Yu

- ![](https://img.shields.io/badge/ACL-2025-red) [**Learn to Memorize: Scalable Continual Learning in Semiparametric Models with Mixture-of-Neighbors Induction Memory**](https://aclanthology.org/2025.acl-long.1385/)

  Guangyue Peng, **<span class="corresponding-author">Tao Ge</span>**, Si-Qing Chen, Furu Wei, Houfeng Wang

- ![](https://img.shields.io/badge/NAACL-2025-red) [**K-Level Reasoning: Establishing Higher Order Beliefs in Large Language Models for Strategic Reasoning**](https://aclanthology.org/2025.naacl-long.370/)

  Yadong Zhang, Shaoguang Mao, **Tao Ge**, Xun Wang, Yan Xia, Man Lan, Furu Wei

- ![](https://img.shields.io/badge/COLING-2025-red) [**ALYMPICS: Language Agents Meet Game Theory**](https://arxiv.org/abs/2311.03220)

  Shaoguang Mao, Yuzhe Cai, Yan Xia, Wenshan Wu, Xun Wang, Fengyi Wang, **Tao Ge**, Furu Wei

- ![](https://img.shields.io/badge/NLPCC-2024-orange) [**Overview of the NLPCC 2024 Shared Task: Chinese Essay Discourse Logic Evaluation and Integration**](https://link.springer.com/chapter/10.1007/978-981-97-9443-0_19)

  Yuhao Zhou, Hongyi Wu, Xinshu Shen, Man Lan, Yuanbin Wu, Xiaopeng Bai, Shaoguang Mao, **Tao Ge**, Yan Xia

- ![](https://img.shields.io/badge/EMNLP-2024-red) [**Learn Beyond The Answer: Training Language Models with Reflection for Mathematical Reasoning**](https://aclanthology.org/2024.emnlp-main.817/)

  Zhihan Zhang, **Tao Ge**, Zhenwen Liang, Wenhao Yu, Dian Yu, Mengzhao Jia, Dong Yu, Meng Jiang

- ![](https://img.shields.io/badge/COLM-2024-orange) [**LLM as a Mastermind: A Survey of Strategic Reasoning with Large Language Models**](https://arxiv.org/abs/2404.01230)

  Yadong Zhang, Shaoguang Mao, **Tao Ge**, Xun Wang, Adrian de Wynter, Yan Xia, Wenshan Wu, Ting Song, Man Lan, Furu Wei

- ![](https://img.shields.io/badge/NeurIPS-2024-red) [**xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token**](https://arxiv.org/abs/2405.13792)

  Xin Cheng, Xun Wang, Xingxing Zhang, **Tao Ge**, Si-Qing Chen, Furu Wei, Huishuai Zhang, Dongyan Zhao

- ![](https://img.shields.io/badge/ACL%20Findings-2024-lightblue) **Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding**

  Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, **Tao Ge**, Tianyu Liu, Wenjie Li, Zhifang Sui

- ![](https://img.shields.io/badge/ACL%20Findings-2024-lightblue) **SCALE: Synergized Collaboration of Asymmetric Language Translation Engines**

  Xin Cheng, Xun Wang, **Tao Ge**, Si-Qing Chen, Furu Wei, Dongyan Zhao, Rui Yan

- ![](https://img.shields.io/badge/ACL%20Findings-2024-lightblue) **Refining Corpora from a Model Calibration Perspective for Chinese Spelling Correction**

  Dingyao Yu, Yang An, Wei Ye, xiongfeng xiao, Shaoguang Mao, **Tao Ge**, Shikun Zhang

- ![](https://img.shields.io/badge/NAACL%20Demo-2024-purple) **Low-code LLM: Visual Programming over LLMs**

  Yuzhe Cai, Shaoguang Mao, Wenshan Wu, Zehua Wang, Yaobo Liang, **Tao Ge**, Chenfei Wu, Wang You, Ting Song, Yan Xia, Jonathan Tien, Nan Duan, Furu Wei

- ![](https://img.shields.io/badge/NAACL-2024-orange) **Unleashing Cognitive Synergy in Large Language Models: A Task-Solving Agent through Multi-Persona Self-Collaboration**

  Zhenhailong Wang, Shaoguang Mao, Wenshan Wu, **<span class="corresponding-author">Tao Ge</span>**, Furu Wei, Heng Ji

- ![](https://img.shields.io/badge/ICLR-2024-red) **In-context Autoencoder for Context Compression in a Large Language Model**

  **<span class="corresponding-author">Tao Ge</span>**, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, Furu Wei

- ![](https://img.shields.io/badge/EMNLP%20Findings-2023-lightblue) **Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation**

  Heming Xia<sup>*</sup>, **<span class="corresponding-author">Tao Ge</span>**<sup>*</sup>, Peiyi Wang, Si-Qing Chen, Furu Wei, Zhifang Sui

  (Originally announced in March 2022: [https://arxiv.org/abs/2203.16487](https://arxiv.org/abs/2203.16487), **the first work proposing Speculative Decoding** that introduces an independent draft model to accelerate generation explicitly with the idea of speculative execution)

- ![](https://img.shields.io/badge/NeurIPS-2023-red) **Extensible Prompts for Language Models on Zero-shot Language Style Customization**

  **<span class="corresponding-author">Tao Ge</span>**, Jing Hu, Li Dong, Shaoguang Mao, Yan Xia, Xun Wang, Si-Qing Chen, Furu Wei

- ![](https://img.shields.io/badge/ACL%20Findings-2023-lightblue) **Smart Word Suggestions for Writing Assistance**

  Chenshuo Wang, Shaoguang Mao, **Tao Ge**, Wenshan Wu, Xun Wang, Yan Xia, Jonathan Tien, Dongyan Zhao

- ![](https://img.shields.io/badge/NLPCC-2023-orange) **Enhancing Detailed Feedback to Chinese Writing Learners Using a Soft-Label Driven Approach and Tag-Aware Ranking Model**

  Yuzhe Cai, Shaoguang Mao, Chenshuo Wang, **Tao Ge**, Wenshan Wu, Yan Xia, Chanjin Zheng, Qiang Guan

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
  
  Xuyang Jin, **<span class="corresponding-author">Tao Ge</span>**, Furu Wei
  
  In **AACL-IJCNLP 2022**

- **A Unified Strategy for Multilingual Grammatical Error Correction with Pre-trained Cross-lingual Language Model**
  
  Xin Sun, **<span class="corresponding-author">Tao Ge</span>**, Shuming Ma, Jingjing Li, Furu Wei, Houfeng Wang
  
  In **IJCAI 2022**

- **Text Revision by On-the-Fly Representation Optimization**
  
  Jingjing Li, Zichao Li, **Tao Ge**, Irwin King, Michael Lyu
  
  In **AAAI 2022**

- **Beyond Preserved Accuracy: Evaluating Loyalty and Robustness of BERT Compression**
  
  Canwen Xu, Wangchunshu Zhou, **<span class="corresponding-author">Tao Ge</span>**, Ke Xu, Julian McAuley, Furu Wei
  
  In **EMNLP 2021**

- **Improving Sequence-to-Sequence Pre-training via Sequence Span Rewriting**
  
  Wangchunshu Zhou, **<span class="corresponding-author">Tao Ge</span>**, Canwen Xu, Ke Xu, Furu Wei
  
  In **EMNLP 2021**

- **Instantaneous Grammatical Error Correction with Shallow Aggressive Decoding**
  
  Xin Sun<sup>\*</sup>, **Tao Ge<sup>\*</sup><sup>\+</sup>**, Furu Wei, Houfeng Wang
  
  In **ACL 2021**

- **Blow the Dog Whistle: A Dataset for Cant Creation, Understanding and Decryption in Chinese**
  
  Canwen Xu<sup>\*</sup>, Wangchunshu Zhou<sup>\*</sup>, **<span class="corresponding-author">Tao Ge</span>**, Ke Xu, Julian McAuley, Furu Wei
  
  In **NAACL 2021**

- **BERT Loses Patience: Fast and Robust Inference with Early Exit**
  
  Wangchunshu Zhou, **<span class="corresponding-author">Tao Ge</span>**, Canwen Xu, Ke Xu, Julian McAuley, Furu Wei
  
  In **NeurIPS 2020**

- **UnihanLM: Coarse-to-Fine Chinese-Japanese Language Model Pretraining with the Unihan Database**
  
  Canwen Xu, **Tao Ge**, Chenliang Li, Furu Wei
  
  In **AACL 2020**

- **Improving the Efficiency of Grammatical Error Correction with Erroneous Span Detection and Correction**
  
  Mengyun Chen<sup>\*</sup>, **Tao Ge<sup>\*</sup><sup>\+</sup>**, Xingxing Zhang, Furu Wei, Ming Zhou
  
  In **EMNLP 2020**

- **BERT-of-Theseus: Compressing BERT by Progressive Module Replacing**
  
  Canwen Xu<sup>\*</sup>, Wangchunshu Zhou<sup>\*</sup>, **<span class="corresponding-author">Tao Ge</span>**, Ke Xu, Julian McAuley, Furu Wei, Ming Zhou
  
  In **EMNLP 2020**

- **Pseudo-Bidirectional Decoding for Local Sequence Transduction**
  
  Wangchunshu Zhou, **Tao Ge**, Chang Mu, Ke Xu, Furu Wei, Ming Zhou
  
  In Findings of **EMNLP 2020**

- **Improving Grammatical Error Correction with Machine Translation Pairs**
  
  Wangchunshu Zhou, **Tao Ge**, Ke Xu, Furu Wei, Ming Zhou
  
  In Findings of **EMNLP 2020**

- **Scheduled DropHead: A Regularization Method for Transformer Models**
  
  Wangchunshu Zhou, **Tao Ge**, Furu Wei, Ming Zhou, Ke Xu
  
  In Findings of **EMNLP 2020**

- **Parallel Data Augmentation for Formality Style Transfer**
  
  Yi Zhang, **Tao Ge**, Xu Sun
  
  In **ACL 2020**

- **Self-Adversarial Learning with Comparative Discrimination for Text Generation**
  
  Wangchunshu Zhou, **Tao Ge**, Ke Xu, Furu Wei, Ming Zhou
  
  In **ICLR 2020**

- **Fact-aware Sentence Split and Rephrase with Permutation Invariant Training**
  
  Yinuo Guo, **Tao Ge**, Furu Wei
  
  In **AAAI 2020**

- **Bert-based Lexical Substitution**
  
  Wangchunshu Zhou, **Tao Ge**, Ke Xu, Furu Wei, Ming Zhou
  
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

- ![](https://img.shields.io/badge/NLPCC%20(Best%20Student%20Paper)-2016-gold) **Discovering Concept-level Event Associations from a Text Stream**

  **Tao Ge**, Lei Cui, Heng Ji, Baobao Chang, Zhifang Sui

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
