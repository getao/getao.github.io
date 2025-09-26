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

- ![](https://img.shields.io/badge/NeurIPS-25-lightgrey?labelColor=red) ![](https://img.shields.io/badge/Spotlight-gold) [**Improving LLM General Preference Alignment via Optimistic Online Mirror Descent**](https://arxiv.org/abs/2410.13184)

  Yuheng Zhang, Dian Yu, **Tao Ge**, Linfeng Song, Zhichen Zeng, Haitao Mi, Nan Jiang, Dong Yu

- ![](https://img.shields.io/badge/EMNLP-25-lightgrey?labelColor=red) [**Router-Tuning: A Simple and Effective Approach for Dynamic Depth**](https://arxiv.org/abs/2410.13184)

  Shwai He, **Tao Ge**, Guoheng Sun, Bowei Tian, Xiaoyang Wang, Dong Yu

- ![](https://img.shields.io/badge/ACL-25-lightgrey?labelColor=red) [**Low-Bit Quantization Favors Undertrained LLMs**](https://aclanthology.org/2025.acl-long.1555/)

  Xu Ouyang, **Tao Ge**, Thomas Hartvigsen, Zhisong Zhang, Haitao Mi, Dong Yu

- ![](https://img.shields.io/badge/ACL-25-lightgrey?labelColor=red) [**Learn to Memorize: Scalable Continual Learning in Semiparametric Models with Mixture-of-Neighbors Induction Memory**](https://aclanthology.org/2025.acl-long.1385/)

  Guangyue Peng, **<span class="corresponding-author">Tao Ge</span>**, Si-Qing Chen, Furu Wei, Houfeng Wang

- ![](https://img.shields.io/badge/NAACL-25-lightgrey?labelColor=red) [**K-Level Reasoning: Establishing Higher Order Beliefs in Large Language Models for Strategic Reasoning**](https://aclanthology.org/2025.naacl-long.370/)

  Yadong Zhang, Shaoguang Mao, **Tao Ge**, Xun Wang, Yan Xia, Man Lan, Furu Wei

- ![](https://img.shields.io/badge/COLING-25-lightgrey?labelColor=red) [**ALYMPICS: Language Agents Meet Game Theory**](https://arxiv.org/abs/2311.03220)

  Shaoguang Mao, Yuzhe Cai, Yan Xia, Wenshan Wu, Xun Wang, Fengyi Wang, **Tao Ge**, Furu Wei

- ![](https://img.shields.io/badge/NLPCC-24-lightgrey?labelColor=orange) [**Overview of the NLPCC 2024 Shared Task: Chinese Essay Discourse Logic Evaluation and Integration**](https://link.springer.com/chapter/10.1007/978-981-97-9443-0_19)

  Yuhao Zhou, Hongyi Wu, Xinshu Shen, Man Lan, Yuanbin Wu, Xiaopeng Bai, Shaoguang Mao, **Tao Ge**, Yan Xia

- ![](https://img.shields.io/badge/EMNLP-24-lightgrey?labelColor=red) [**Learn Beyond The Answer: Training Language Models with Reflection for Mathematical Reasoning**](https://aclanthology.org/2024.emnlp-main.817/)

  Zhihan Zhang, **Tao Ge**, Zhenwen Liang, Wenhao Yu, Dian Yu, Mengzhao Jia, Dong Yu, Meng Jiang

- ![](https://img.shields.io/badge/COLM-24-lightgrey?labelColor=orange) [**LLM as a Mastermind: A Survey of Strategic Reasoning with Large Language Models**](https://arxiv.org/abs/2404.01230)

  Yadong Zhang, Shaoguang Mao, **Tao Ge**, Xun Wang, Adrian de Wynter, Yan Xia, Wenshan Wu, Ting Song, Man Lan, Furu Wei

- ![](https://img.shields.io/badge/NeurIPS-24-lightgrey?labelColor=red) [**xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token**](https://arxiv.org/abs/2405.13792)

  Xin Cheng, Xun Wang, Xingxing Zhang, **Tao Ge**, Si-Qing Chen, Furu Wei, Huishuai Zhang, Dongyan Zhao

- ![](https://img.shields.io/badge/ACL%20Findings-24-lightgrey?labelColor=orangered) **Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding**

  Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, **Tao Ge**, Tianyu Liu, Wenjie Li, Zhifang Sui

- ![](https://img.shields.io/badge/ACL%20Findings-24-lightgrey?labelColor=orangered) **SCALE: Synergized Collaboration of Asymmetric Language Translation Engines**

  Xin Cheng, Xun Wang, **Tao Ge**, Si-Qing Chen, Furu Wei, Dongyan Zhao, Rui Yan

- ![](https://img.shields.io/badge/ACL%20Findings-24-lightgrey?labelColor=orangered) **Refining Corpora from a Model Calibration Perspective for Chinese Spelling Correction**

  Dingyao Yu, Yang An, Wei Ye, xiongfeng xiao, Shaoguang Mao, **Tao Ge**, Shikun Zhang

- ![](https://img.shields.io/badge/NAACL%20Demo-24-lightgrey?labelColor=purple) **Low-code LLM: Visual Programming over LLMs**

  Yuzhe Cai, Shaoguang Mao, Wenshan Wu, Zehua Wang, Yaobo Liang, **Tao Ge**, Chenfei Wu, Wang You, Ting Song, Yan Xia, Jonathan Tien, Nan Duan, Furu Wei

- ![](https://img.shields.io/badge/NAACL-24-lightgrey?labelColor=red) **Unleashing Cognitive Synergy in Large Language Models: A Task-Solving Agent through Multi-Persona Self-Collaboration**

  Zhenhailong Wang, Shaoguang Mao, Wenshan Wu, **<span class="corresponding-author">Tao Ge</span>**, Furu Wei, Heng Ji

- ![](https://img.shields.io/badge/ICLR-24-lightgrey?labelColor=red) **In-context Autoencoder for Context Compression in a Large Language Model**

  **<span class="corresponding-author">Tao Ge</span>**, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, Furu Wei

- ![](https://img.shields.io/badge/EMNLP%20Findings-23-lightgrey?labelColor=orangered) **Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation**

  Heming Xia<sup>\*</sup>, **<span class="corresponding-author">Tao Ge</span>**<sup>\*</sup>, Peiyi Wang, Si-Qing Chen, Furu Wei, Zhifang Sui

  (Originally announced in March 2022: [https://arxiv.org/abs/2203.16487](https://arxiv.org/abs/2203.16487), **the first work proposing Speculative Decoding** that introduces an independent draft model to accelerate generation explicitly with the idea of speculative execution)

- ![](https://img.shields.io/badge/NeurIPS-23-lightgrey?labelColor=red) **Extensible Prompts for Language Models on Zero-shot Language Style Customization**

  **<span class="corresponding-author">Tao Ge</span>**, Jing Hu, Li Dong, Shaoguang Mao, Yan Xia, Xun Wang, Si-Qing Chen, Furu Wei

- ![](https://img.shields.io/badge/ACL%20Findings-23-lightgrey?labelColor=orangered) **Smart Word Suggestions for Writing Assistance**

  Chenshuo Wang, Shaoguang Mao, **Tao Ge**, Wenshan Wu, Xun Wang, Yan Xia, Jonathan Tien, Dongyan Zhao

- ![](https://img.shields.io/badge/NLPCC-23-lightgrey?labelColor=orange) **Enhancing Detailed Feedback to Chinese Writing Learners Using a Soft-Label Driven Approach and Tag-Aware Ranking Model**

  Yuzhe Cai, Shaoguang Mao, Chenshuo Wang, **Tao Ge**, Wenshan Wu, Yan Xia, Chanjin Zheng, Qiang Guan

- ![](https://img.shields.io/badge/NLPCC-23-lightgrey?labelColor=orange) **Overview of the NLPCC 2023 Shared Task: Chinese Essay Discourse Coherence Evaluation**

  Hongyi Wu, Xinshu Shen, Man Lan, Xiaopeng Bai, Yuanbin Wu, Aimin Zhou, Shaoguang Mao, **Tao Ge**, Yan Xia

- ![](https://img.shields.io/badge/CCL-23-lightgrey?labelColor=orange) **Overview of CCL23-Eval Task 8: Chinese Essay Fluency Evaluation (CEFE) Task**

  Xinshu Shen, Hongyi Wu, Xiaopeng Bai, Yuanbin Wu, Aimin Zhou, Shaoguang Mao, **Tao Ge**, Yan Xia

- ![](https://img.shields.io/badge/EMNLP-22-lightgrey?labelColor=red) **EdgeFormer: A Parameter-Efficient Transformer for On-Device Seq2seq Generation**

  **Tao Ge**, Si-Qing Chen, Furu Wei

- ![](https://img.shields.io/badge/AACL--IJCNLP-22-lightgrey?labelColor=orange) **Plug and Play Knowledge Distillation for kNN-LM with External Logits**

  Xuyang Jin, **<span class="corresponding-author">Tao Ge</span>**, Furu Wei

- ![](https://img.shields.io/badge/IJCAI-22-lightgrey?labelColor=red) **A Unified Strategy for Multilingual Grammatical Error Correction with Pre-trained Cross-lingual Language Model**

  Xin Sun, **<span class="corresponding-author">Tao Ge</span>**, Shuming Ma, Jingjing Li, Furu Wei, Houfeng Wang

- ![](https://img.shields.io/badge/AAAI-22-lightgrey?labelColor=red) **Text Revision by On-the-Fly Representation Optimization**

  Jingjing Li, Zichao Li, **Tao Ge**, Irwin King, Michael Lyu

- ![](https://img.shields.io/badge/EMNLP-21-lightgrey?labelColor=red) **Beyond Preserved Accuracy: Evaluating Loyalty and Robustness of BERT Compression**

  Canwen Xu, Wangchunshu Zhou, **<span class="corresponding-author">Tao Ge</span>**, Ke Xu, Julian McAuley, Furu Wei

- ![](https://img.shields.io/badge/EMNLP-21-lightgrey?labelColor=red) **Improving Sequence-to-Sequence Pre-training via Sequence Span Rewriting**

  Wangchunshu Zhou, **<span class="corresponding-author">Tao Ge</span>**, Canwen Xu, Ke Xu, Furu Wei

- ![](https://img.shields.io/badge/ACL-21-lightgrey?labelColor=red) **Instantaneous Grammatical Error Correction with Shallow Aggressive Decoding**

  Xin Sun<sup>\*</sup>, **<span class="corresponding-author">Tao Ge</span>**<sup>\*</sup>, Furu Wei, Houfeng Wang

- ![](https://img.shields.io/badge/NAACL-21-lightgrey?labelColor=red) **Blow the Dog Whistle: A Dataset for Cant Creation, Understanding and Decryption in Chinese**

  Canwen Xu<sup>\*</sup>, Wangchunshu Zhou<sup>\*</sup>, **<span class="corresponding-author">Tao Ge</span>**, Ke Xu, Julian McAuley, Furu Wei

- ![](https://img.shields.io/badge/NeurIPS-20-lightgrey?labelColor=red) **BERT Loses Patience: Fast and Robust Inference with Early Exit**

  Wangchunshu Zhou, **<span class="corresponding-author">Tao Ge</span>**, Canwen Xu, Ke Xu, Julian McAuley, Furu Wei

- ![](https://img.shields.io/badge/AACL-20-lightgrey?labelColor=orange) **UnihanLM: Coarse-to-Fine Chinese-Japanese Language Model Pretraining with the Unihan Database**

  Canwen Xu, **Tao Ge**, Chenliang Li, Furu Wei

- ![](https://img.shields.io/badge/EMNLP-20-lightgrey?labelColor=red) **Improving the Efficiency of Grammatical Error Correction with Erroneous Span Detection and Correction**

  Mengyun Chen<sup>\*</sup>, **<span class="corresponding-author">Tao Ge</span>**<sup>\*</sup>, Xingxing Zhang, Furu Wei, Ming Zhou

- ![](https://img.shields.io/badge/EMNLP-20-lightgrey?labelColor=red) **BERT-of-Theseus: Compressing BERT by Progressive Module Replacing**

  Canwen Xu<sup>\*</sup>, Wangchunshu Zhou<sup>\*</sup>, **<span class="corresponding-author">Tao Ge</span>**, Ke Xu, Julian McAuley, Furu Wei, Ming Zhou

- ![](https://img.shields.io/badge/EMNLP%20Findings-20-lightgrey?labelColor=orangered) **Pseudo-Bidirectional Decoding for Local Sequence Transduction**

  Wangchunshu Zhou, **Tao Ge**, Chang Mu, Ke Xu, Furu Wei, Ming Zhou

- ![](https://img.shields.io/badge/EMNLP%20Findings-20-lightgrey?labelColor=orangered) **Improving Grammatical Error Correction with Machine Translation Pairs**

  Wangchunshu Zhou, **Tao Ge**, Ke Xu, Furu Wei, Ming Zhou

- ![](https://img.shields.io/badge/EMNLP%20Findings-20-lightgrey?labelColor=orangered) **Scheduled DropHead: A Regularization Method for Transformer Models**

  Wangchunshu Zhou, **Tao Ge**, Furu Wei, Ming Zhou, Ke Xu

- ![](https://img.shields.io/badge/ACL-20-lightgrey?labelColor=red) **Parallel Data Augmentation for Formality Style Transfer**

  Yi Zhang, **Tao Ge**, Xu Sun

- ![](https://img.shields.io/badge/ICLR-20-lightgrey?labelColor=red) **Self-Adversarial Learning with Comparative Discrimination for Text Generation**

  Wangchunshu Zhou, **Tao Ge**, Ke Xu, Furu Wei, Ming Zhou

- ![](https://img.shields.io/badge/AAAI-20-lightgrey?labelColor=red) **Fact-aware Sentence Split and Rephrase with Permutation Invariant Training**

  Yinuo Guo, **Tao Ge**, Furu Wei

- ![](https://img.shields.io/badge/ACL-19-lightgrey?labelColor=red) **Bert-based Lexical Substitution**
  
  Wangchunshu Zhou, **Tao Ge**, Ke Xu, Furu Wei, Ming Zhou
  

- ![](https://img.shields.io/badge/ACL-19-lightgrey?labelColor=red) **Automatic Grammatical Error Correction for Sequence-to-sequence Text Generation: An Empirical Study**
  
  **Tao Ge**, Xingxing Zhang, Furu Wei, Ming Zhou
  

- ![](https://img.shields.io/badge/EMNLP-18-lightgrey?labelColor=red) **Fine-grained Coordinated Cross-lingual Text Stream Alignment for Endless Language Knowledge Acquisition**
  
  **Tao Ge**, Qing Dou, Heng Ji, Lei Cui, Baobao Chang, Zhifang Sui, Furu Wei, Ming Zhou
  

- ![](https://img.shields.io/badge/ACL-18-lightgrey?labelColor=red) **Fluency Boost Learning and Inference for Neural Grammatical Error Correction**
  
  **Tao Ge**, Furu Wei, Ming Zhou
  

- ![](https://img.shields.io/badge/LREC-18-lightgrey?labelColor=orange) **EventWiki: A Knowledge Base of Major Events**
  
  **Tao Ge**, Lei Cui, Baobao Chang, Zhifang Sui, Furu Wei, Ming Zhou
  

- ![](https://img.shields.io/badge/NLPCC-18-lightgrey?labelColor=orange) **SeRI: A Dataset for Sub-event Relation Inference from an Encyclopedia**
  
  **Tao Ge**, Lei Cui, Baobao Chang, Zhifang Sui, Furu Wei, Ming Zhou
  

- ![](https://img.shields.io/badge/COLING-16-lightgrey?labelColor=red) **Event detection with Burst Information Networks**
  
  **Tao Ge**, Lei Cui, Baobao Chang, Zhifang Sui, Ming Zhou
  

- ![](https://img.shields.io/badge/EMNLP-16-lightgrey?labelColor=red) **News Stream Summarization using Burst Information Networks**
  
  **Tao Ge**, Lei Cui, Heng Ji, Baobao Chang, Sujian Li, Ming Zhou, Zhifang Sui
  

- ![](https://img.shields.io/badge/NLPCC-16-lightgrey?labelColor=orange) ![](https://img.shields.io/badge/Best%20Student%20Paper-gold) **Discovering Concept-level Event Associations from a Text Stream**

  **Tao Ge**, Lei Cui, Heng Ji, Baobao Chang, Zhifang Sui

- ![](https://img.shields.io/badge/COLING-16-lightgrey?labelColor=red) **Towards Time-aware Knowledge Graph Completion**
  
  Tingsong Jiang, Tianyu Liu, **Tao Ge**, Lei Sha, Baobao Chang, Sujian Li, Zhifang Sui
  

- ![](https://img.shields.io/badge/EMNLP-16-lightgrey?labelColor=red) **Encoding Temporal Information for Time-aware Link Prediction**
  
  Tingsong Jiang, Tianyu Liu, **Tao Ge**, Lei Sha, Sujian Li, Baobao Chang, Zhifang Sui
  

- ![](https://img.shields.io/badge/ACL-15-lightgrey?labelColor=red) **One Tense per Scene: Predicting Tense in Chinese Conversations**
  
  **Tao Ge**, Heng Ji, Baobao Chang, Zhifang Sui
  

- ![](https://img.shields.io/badge/ACL-15-lightgrey?labelColor=red) **Bring you to the past: Automatic Generation of Topically Relevant Event Chronicles**
  
  **Tao Ge**, Wenzhe Pei, Heng Ji, Sujian Li, Baobao Chang, Zhifang Sui
  

- ![](https://img.shields.io/badge/ACL-15-lightgrey?labelColor=red) **An Effective Neural Network Model for Graph-based Dependency Parsing**
  
  Wenzhe Pei, **Tao Ge**, Baobao Chang
  

- ![](https://img.shields.io/badge/BioNLP-15-lightgrey?labelColor=orange) **Exploiting task-oriented resources to learn word embeddings for clinical abbreviation expansion**
  
  Yue Liu, **Tao Ge**, Kusum S Mathews, Heng Ji, Deborah McGuinness
  

- ![](https://img.shields.io/badge/ACL-14-lightgrey?labelColor=red) **Max-Margin Tensor Neural Network for Chinese Word Segmentation**
  
  Wenzhe Pei, **Tao Ge**, Baobao Chang
  

- ![](https://img.shields.io/badge/WWW-14-lightgrey?labelColor=red) **A semi-supervised method for opinion target extraction**
  
  **Tao Ge**, Wenjie Li, Zhifang Sui
  

- ![](https://img.shields.io/badge/CIPS--SIGHAN-14-lightgrey?labelColor=orange) **The CIPS-SIGHAN CLP 2014 Chinese Word Segmentation Bake-off**
  
  Huiming Duan, Zhifang Sui, **Tao Ge**
  

- ![](https://img.shields.io/badge/CIKM-13-lightgrey?labelColor=red) **Exploiting Collaborative Filtering Techniques for Automatic Assessment of Student Free-text Responses**
  
  **Tao Ge**, Zhifang Sui, Baobao Chang
  

- ![](https://img.shields.io/badge/EMNLP-13-lightgrey?labelColor=red) **Event-Based Time Label Propagation for Automatic Dating of News Articles**
  
  **Tao Ge**, Baobao Chang, Sujian Li, Zhifang Sui
  
