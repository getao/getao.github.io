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

Dr. Tao Ge is a Principal Science Lead at Microsoft in Redmond, leading research and development of state-of-the-art large language models (LLMs), spanning synthetic data creation, mid-/post-training of OpenAI models (GPT-4/5, and o3/o4-mini), and agentic approaches powering Microsoft products (e.g., Office/Copilot). Prior to his current role, Tao was a Principal Researcher at Tencent AI Lab (Seattle) and Microsoft Research Asia (MSRA) after earning his Ph.D. in Computer Science from Peking University. 

Tao has published more than 60 papers at top AI/ML conferences. Two of his most known and widely adopted tech innovations are:

1. **Speculative Decoding**: Tao pioneered the seminal study of Speculative Decoding beginning in [2021](https://aclanthology.org/2021.acl-long.462.pdf) (initially referred to as [Aggressive Decoding](https://arxiv.org/pdf/2205.10350)). He was **the first to introduce a separate drafter model** to achieve lossless acceleration of Transformer decoding (first made public in [March 2022](https://arxiv.org/pdf/2203.16487)), and he was also **the first to coin the term "Speculative Decoding"** for this speculative execution paradigm (made public in [September 2022](https://openreview.net/forum?id=H-VlwsYvVi)). His research was subsequently followed by the papers on Speculative Decoding/Sampling for LLMs from Google (first made public in [November 2022](https://arxiv.org/abs/2211.17192)) and DeepMind (made public in [February 2023](https://arxiv.org/abs/2302.01318)), sparking the surge of interest and adoption since mid-2023. Today, Speculative Decoding has become an industry standard for LLM inference acceleration, supported in major open-source frameworks (e.g., vLLM, PyTorch, ONNX) and widely integrated into production-scale deployments.

2. **Persona-Driven Synthetic Data Creation**: Tao proposed [persona-driven synthetic data creation](https://arxiv.org/abs/2406.20094), a novel paradigm for scaling high-quality training data generation. This innovation has been widely recognized and adopted as a core synthetic data methodology in the development of leading LLMs, including (but not limited to):
   - [OLMo](https://arxiv.org/abs/2501.00656)/[Molmo and PixMo](https://arxiv.org/abs/2409.17146) (AI2)
   - [Qwen](https://qwen.ai/home) (Alibaba)
   - [Hunyuan](https://github.com/Tencent-Hunyuan) (Tencent)
   - [LongCat](https://arxiv.org/abs/2509.18883) (Meituan)
   - [SmolLM](https://arxiv.org/abs/2502.02737) (Hugging Face)
   - [Hermes](https://arxiv.org/abs/2508.18255) (Nous Research)
   - [Minicpm](https://arxiv.org/abs/2506.07900) (ModelBest)

<span class='anchor' id='publications'></span>

# Publications (<sup>\*</sup>: equal contributions; <sup>✉</sup>: corresponding author)

## Preprint

- ![](https://img.shields.io/badge/arXiv-25.01-white?labelColor=tan) [**OpenCharacter: Training Customizable Role-Playing LLMs with Large-Scale Synthetic Personas**](https://arxiv.org/abs/2501.15427)

  Xiaoyang Wang, Hongming Zhang, **Tao Ge**, Wenhao Yu, Dian Yu, Dong Yu

- ![](https://img.shields.io/badge/arXiv-24.10-white?labelColor=tan) [**ParallelSpec: Parallel Drafter for Efficient Speculative Decoding**](https://arxiv.org/pdf/2410.05589)

  Zilin Xiao, Hongming Zhang, **Tao Ge**, Siru Ouyang, Vicente Ordonez, Dong Yu

- ![](https://img.shields.io/badge/arXiv-23.04-white?labelColor=tan) [**Inference with Reference: Lossless Acceleration of Large Language Models**](https://arxiv.org/abs/2304.04487) (the innovation used in [OpenAI's Predicted Output](https://platform.openai.com/docs/guides/predicted-outputs)) 

  Nan Yang, **Tao Ge**, Liang Wang, Binxing Jiao, Daxin Jiang, Linjun Yang, Rangan Majumder, Furu Wei

## Tech Report

- ![](https://img.shields.io/badge/arXiv-24.06-white?labelColor=tan) [**Scaling Synthetic Data Creation with 1,000,000,000 Personas**](https://arxiv.org/pdf/2406.20094)

  **Tao Ge**, Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, Dong Yu

- ![](https://img.shields.io/badge/arXiv-22.05-white?labelColor=tan) [**Lossless Acceleration for Seq2seq Generation with Aggressive Decoding**](https://arxiv.org/abs/2205.10350) (an earlier tech report of my proposed Speculative Decoding)

  **Tao Ge**, Heming Xia, Xin Sun, Si-Qing Chen, Furu Wei

- ![](https://img.shields.io/badge/arXiv-18.07-white?labelColor=tan) [**Reaching Human-level Performance in Automatic Grammatical Error Correction: An Empirical Study**](https://arxiv.org/abs/1807.01270)  

  **Tao Ge**, Furu Wei, Ming Zhou


## Peer-reviewed

- ![](https://img.shields.io/badge/NeurIPS-25-white?labelColor=red) ![](https://img.shields.io/badge/Spotlight-gold) [**Improving LLM General Preference Alignment via Optimistic Online Mirror Descent**](https://arxiv.org/abs/2410.13184)

  Yuheng Zhang, Dian Yu, **Tao Ge**, Linfeng Song, Zhichen Zeng, Haitao Mi, Nan Jiang, Dong Yu

- ![](https://img.shields.io/badge/EMNLP-25-white?labelColor=red) [**Router-Tuning: A Simple and Effective Approach for Dynamic Depth**](https://arxiv.org/abs/2410.13184)

  Shwai He, **Tao Ge**, Guoheng Sun, Bowei Tian, Xiaoyang Wang, Dong Yu

- ![](https://img.shields.io/badge/ACL-25-white?labelColor=red) [**Low-Bit Quantization Favors Undertrained LLMs**](https://aclanthology.org/2025.acl-long.1555/)

  Xu Ouyang, **Tao Ge**<sup>✉</sup>, Thomas Hartvigsen, Zhisong Zhang, Haitao Mi, Dong Yu

- ![](https://img.shields.io/badge/ACL-25-white?labelColor=red) [**Learn to Memorize: Scalable Continual Learning in Semiparametric Models with Mixture-of-Neighbors Induction Memory**](https://aclanthology.org/2025.acl-long.1385/)

  Guangyue Peng, **Tao Ge**<sup>✉</sup>, Wen Luo, Wei Li, Houfeng Wang

- ![](https://img.shields.io/badge/NAACL-25-white?labelColor=red) [**K-Level Reasoning: Establishing Higher Order Beliefs in Large Language Models for Strategic Reasoning**](https://aclanthology.org/2025.naacl-long.370/)

  Yadong Zhang, Shaoguang Mao, **Tao Ge**, Xun Wang, Yan Xia, Man Lan, Furu Wei

- ![](https://img.shields.io/badge/COLING-25-white?labelColor=red) [**ALYMPICS: Language Agents Meet Game Theory**](https://aclanthology.org/2025.coling-main.193/)

  Shaoguang Mao, Yuzhe Cai, Yan Xia, Wenshan Wu, Xun Wang, Fengyi Wang, **Tao Ge**, Furu Wei

- ![](https://img.shields.io/badge/NLPCC-24-white?labelColor=orange) [**Overview of the NLPCC 2024 Shared Task: Chinese Essay Discourse Logic Evaluation and Integration**](https://link.springer.com/chapter/10.1007/978-981-97-9443-0_19)

  Yuhao Zhou, Hongyi Wu, Xinshu Shen, Man Lan, Yuanbin Wu, Xiaopeng Bai, Shaoguang Mao, **Tao Ge**, Yan Xia

- ![](https://img.shields.io/badge/EMNLP-24-white?labelColor=red) [**Learn Beyond The Answer: Training Language Models with Reflection for Mathematical Reasoning**](https://aclanthology.org/2024.emnlp-main.817/)

  Zhihan Zhang, **Tao Ge**, Zhenwen Liang, Wenhao Yu, Dian Yu, Mengzhao Jia, Dong Yu, Meng Jiang

- ![](https://img.shields.io/badge/COLM-24-white?labelColor=orange) [**LLM as a Mastermind: A Survey of Strategic Reasoning with Large Language Models**](https://openreview.net/forum?id=iMqJsQ4evS)

  Yadong Zhang, Shaoguang Mao, **Tao Ge**, Xun Wang, Adrian de Wynter, Yan Xia, Wenshan Wu, Ting Song, Man Lan, Furu Wei

- ![](https://img.shields.io/badge/NeurIPS-24-white?labelColor=red) [**xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token**](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c5cf13bfd3762821ef7607e63ee90075-Abstract-Conference.html)

  Xin Cheng, Xun Wang, Xingxing Zhang, **Tao Ge**, Si-Qing Chen, Furu Wei, Huishuai Zhang, Dongyan Zhao

- ![](https://img.shields.io/badge/ACL%20Findings-24-white?labelColor=tomato) [**Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding**](https://aclanthology.org/2024.findings-acl.456/)

  Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, **Tao Ge**, Tianyu Liu, Wenjie Li, Zhifang Sui

- ![](https://img.shields.io/badge/ACL%20Findings-24-white?labelColor=tomato) [**SCALE: Synergized Collaboration of Asymmetric Language Translation Engines**](https://aclanthology.org/2024.findings-acl.941/)

  Xin Cheng, Xun Wang, **Tao Ge**, Si-Qing Chen, Furu Wei, Dongyan Zhao, Rui Yan

- ![](https://img.shields.io/badge/ACL%20Findings-24-white?labelColor=tomato) [**Refining Corpora from a Model Calibration Perspective for Chinese Spelling Correction**](https://aclanthology.org/2024.findings-acl.914/)

  Dingyao Yu, Yang An, Wei Ye, xiongfeng xiao, Shaoguang Mao, **Tao Ge**, Shikun Zhang

- ![](https://img.shields.io/badge/NAACL%20Demo-24-white?labelColor=tomato) [**Low-code LLM: Visual Programming over LLMs**](https://aclanthology.org/2024.naacl-demo.2/)

  Yuzhe Cai, Shaoguang Mao, Wenshan Wu, Zehua Wang, Yaobo Liang, **Tao Ge**, Chenfei Wu, Wang You, Ting Song, Yan Xia, Jonathan Tien, Nan Duan, Furu Wei

- ![](https://img.shields.io/badge/NAACL-24-white?labelColor=red) [**Unleashing Cognitive Synergy in Large Language Models: A Task-Solving Agent through Multi-Persona Self-Collaboration**](https://aclanthology.org/2024.naacl-long.15/)

  Zhenhailong Wang, Shaoguang Mao, Wenshan Wu, **Tao Ge**<sup>✉</sup>, Furu Wei, Heng Ji

- ![](https://img.shields.io/badge/ICLR-24-white?labelColor=red) [**In-context Autoencoder for Context Compression in a Large Language Model**](https://openreview.net/forum?id=uREj4ZuGJE)

  **Tao Ge**<sup>✉</sup>, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, Furu Wei

- ![](https://img.shields.io/badge/EMNLP%20Findings-23-white?labelColor=tomato) [**Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation**](https://aclanthology.org/2023.findings-emnlp.257/)

  Heming Xia<sup>\*</sup>, **Tao Ge**<sup>✉</sup><sup>\*</sup>, Peiyi Wang, Si-Qing Chen, Furu Wei, Zhifang Sui

- ![](https://img.shields.io/badge/NeurIPS-23-white?labelColor=red) [**Extensible Prompts for Language Models on Zero-shot Language Style Customization**](https://papers.nips.cc/paper_files/paper/2023/hash/6fcbfb3721c1781728b10c6685cc2f6c-Abstract-Conference.html)

  **Tao Ge**<sup>✉</sup>, Jing Hu, Li Dong, Shaoguang Mao, Yan Xia, Xun Wang, Si-Qing Chen, Furu Wei

- ![](https://img.shields.io/badge/ACL%20Findings-23-white?labelColor=tomato) [**Smart Word Suggestions for Writing Assistance**](https://aclanthology.org/2023.findings-acl.712/)

  Chenshuo Wang, Shaoguang Mao, **Tao Ge**, Wenshan Wu, Xun Wang, Yan Xia, Jonathan Tien, Dongyan Zhao

- ![](https://img.shields.io/badge/NLPCC-23-white?labelColor=orange) [**Enhancing Detailed Feedback to Chinese Writing Learners Using a Soft-Label Driven Approach and Tag-Aware Ranking Model**](https://link.springer.com/chapter/10.1007/978-3-031-44693-1_45)

  Yuzhe Cai, Shaoguang Mao, Chenshuo Wang, **Tao Ge**, Wenshan Wu, Yan Xia, Chanjin Zheng, Qiang Guan

- ![](https://img.shields.io/badge/NLPCC-23-white?labelColor=orange) [**Overview of the NLPCC 2023 Shared Task: Chinese Essay Discourse Coherence Evaluation**](https://link.springer.com/chapter/10.1007/978-3-031-44699-3_26)

  Hongyi Wu, Xinshu Shen, Man Lan, Xiaopeng Bai, Yuanbin Wu, Aimin Zhou, Shaoguang Mao, **Tao Ge**, Yan Xia

- ![](https://img.shields.io/badge/CCL-23-white?labelColor=orange) [**Overview of CCL23-Eval Task 8: Chinese Essay Fluency Evaluation (CEFE) Task**](https://aclanthology.org/2023.ccl-3.31/)

  Xinshu Shen, Hongyi Wu, Xiaopeng Bai, Yuanbin Wu, Aimin Zhou, Shaoguang Mao, **Tao Ge**, Yan Xia

- ![](https://img.shields.io/badge/EMNLP-22-white?labelColor=red) [**EdgeFormer: A Parameter-Efficient Transformer for On-Device Seq2seq Generation**](https://aclanthology.org/2022.emnlp-main.741/)

  **Tao Ge**, Si-Qing Chen, Furu Wei

- ![](https://img.shields.io/badge/AACL-22-white?labelColor=orange) [**Plug and Play Knowledge Distillation for kNN-LM with External Logits**](https://aclanthology.org/2022.aacl-short.57/)

  Xuyang Jin, **Tao Ge**<sup>✉</sup>, Furu Wei

- ![](https://img.shields.io/badge/IJCAI-22-white?labelColor=red) [**A Unified Strategy for Multilingual Grammatical Error Correction with Pre-trained Cross-lingual Language Model**](https://www.ijcai.org/proceedings/2022/606)

  Xin Sun, **Tao Ge**<sup>✉</sup>, Shuming Ma, Jingjing Li, Furu Wei, Houfeng Wang

- ![](https://img.shields.io/badge/AAAI-22-white?labelColor=red) [**Text Revision by On-the-Fly Representation Optimization**](https://ojs.aaai.org/index.php/AAAI/article/view/21343)

  Jingjing Li, Zichao Li, **Tao Ge**, Irwin King, Michael Lyu

- ![](https://img.shields.io/badge/EMNLP-21-white?labelColor=red) [**Beyond Preserved Accuracy: Evaluating Loyalty and Robustness of BERT Compression**](https://aclanthology.org/2021.emnlp-main.832/)

  Canwen Xu, Wangchunshu Zhou, **Tao Ge**<sup>✉</sup>, Ke Xu, Julian McAuley, Furu Wei

- ![](https://img.shields.io/badge/EMNLP-21-white?labelColor=red) [**Improving Sequence-to-Sequence Pre-training via Sequence Span Rewriting**](https://aclanthology.org/2021.emnlp-main.45/)

  Wangchunshu Zhou, **Tao Ge**<sup>✉</sup>, Canwen Xu, Ke Xu, Furu Wei

- ![](https://img.shields.io/badge/ACL-21-white?labelColor=red) [**Instantaneous Grammatical Error Correction with Shallow Aggressive Decoding**](https://aclanthology.org/2021.acl-long.462/)

  Xin Sun<sup>\*</sup>, **Tao Ge**<sup>✉</sup><sup>\*</sup>, Furu Wei, Houfeng Wang

- ![](https://img.shields.io/badge/NAACL-21-white?labelColor=red) [**Blow the Dog Whistle: A Dataset for Cant Creation, Understanding and Decryption in Chinese**](https://aclanthology.org/2021.naacl-main.172/)

  Canwen Xu<sup>\*</sup>, Wangchunshu Zhou<sup>\*</sup>, **Tao Ge**<sup>✉</sup>, Ke Xu, Julian McAuley, Furu Wei

- ![](https://img.shields.io/badge/NeurIPS-20-white?labelColor=red) [**BERT Loses Patience: Fast and Robust Inference with Early Exit**](https://proceedings.neurips.cc/paper/2020/hash/d4dd111a4fd973394238aca5c05bebe3-Abstract.html)

  Wangchunshu Zhou, Canwen Xu, **Tao Ge**<sup>✉</sup>, Ke Xu, Julian McAuley, Furu Wei

- ![](https://img.shields.io/badge/AACL-20-white?labelColor=orange) [**UnihanLM: Coarse-to-Fine Chinese-Japanese Language Model Pretraining with the Unihan Database**](https://aclanthology.org/2020.aacl-main.24/)

  Canwen Xu, **Tao Ge**, Chenliang Li, Furu Wei

- ![](https://img.shields.io/badge/EMNLP-20-white?labelColor=red) [**Improving the Efficiency of Grammatical Error Correction with Erroneous Span Detection and Correction**](https://aclanthology.org/2020.emnlp-main.581/)

  Mengyun Chen<sup>\*</sup>, **Tao Ge**<sup>✉</sup><sup>\*</sup>, Xingxing Zhang, Furu Wei, Ming Zhou

- ![](https://img.shields.io/badge/EMNLP-20-white?labelColor=red) [**BERT-of-Theseus: Compressing BERT by Progressive Module Replacing**](https://aclanthology.org/2020.emnlp-main.633/)

  Canwen Xu<sup>\*</sup>, Wangchunshu Zhou<sup>\*</sup>, **Tao Ge**<sup>✉</sup>, Ke Xu, Julian McAuley, Furu Wei, Ming Zhou

- ![](https://img.shields.io/badge/EMNLP%20Findings-20-white?labelColor=tomato) [**Pseudo-Bidirectional Decoding for Local Sequence Transduction**](https://aclanthology.org/2020.findings-emnlp.136/)

  Wangchunshu Zhou, **Tao Ge**, Chang Mu, Ke Xu, Furu Wei, Ming Zhou

- ![](https://img.shields.io/badge/EMNLP%20Findings-20-white?labelColor=tomato) [**Improving Grammatical Error Correction with Machine Translation Pairs**](https://aclanthology.org/2020.findings-emnlp.30/)

  Wangchunshu Zhou, **Tao Ge**, Ke Xu, Furu Wei, Ming Zhou

- ![](https://img.shields.io/badge/EMNLP%20Findings-20-white?labelColor=tomato) [**Scheduled DropHead: A Regularization Method for Transformer Models**](https://aclanthology.org/2020.findings-emnlp.178/)

  Wangchunshu Zhou, **Tao Ge**, Furu Wei, Ming Zhou, Ke Xu

- ![](https://img.shields.io/badge/ACL-20-white?labelColor=red) [**Parallel Data Augmentation for Formality Style Transfer**](https://aclanthology.org/2020.acl-main.294/)

  Yi Zhang, **Tao Ge**, Xu Sun

- ![](https://img.shields.io/badge/ICLR-20-white?labelColor=red) [**Self-Adversarial Learning with Comparative Discrimination for Text Generation**](https://openreview.net/forum?id=B1l8L6EtDS)

  Wangchunshu Zhou, **Tao Ge**, Ke Xu, Furu Wei, Ming Zhou

- ![](https://img.shields.io/badge/AAAI-20-white?labelColor=red) [**Fact-aware Sentence Split and Rephrase with Permutation Invariant Training**](https://ojs.aaai.org/index.php/AAAI/article/view/6291)

  Yinuo Guo, **Tao Ge**, Furu Wei

- ![](https://img.shields.io/badge/ACL-19-white?labelColor=red) [**Bert-based Lexical Substitution**](https://aclanthology.org/P19-1328/)

  Wangchunshu Zhou, **Tao Ge**, Ke Xu, Furu Wei, Ming Zhou
  

- ![](https://img.shields.io/badge/ACL-19-white?labelColor=red) [**Automatic Grammatical Error Correction for Sequence-to-sequence Text Generation: An Empirical Study**](https://aclanthology.org/P19-1609/)

  **Tao Ge**, Xingxing Zhang, Furu Wei, Ming Zhou
  

- ![](https://img.shields.io/badge/EMNLP-18-white?labelColor=red) [**Fine-grained Coordinated Cross-lingual Text Stream Alignment for Endless Language Knowledge Acquisition**](https://aclanthology.org/D18-1271/)

  **Tao Ge**, Qing Dou, Heng Ji, Lei Cui, Baobao Chang, Zhifang Sui, Furu Wei, Ming Zhou
  

- ![](https://img.shields.io/badge/ACL-18-white?labelColor=red) [**Fluency Boost Learning and Inference for Neural Grammatical Error Correction**](https://aclanthology.org/P18-1097/)

  **Tao Ge**, Furu Wei, Ming Zhou
  

- ![](https://img.shields.io/badge/LREC-18-white?labelColor=orange) [**EventWiki: A Knowledge Base of Major Events**](https://aclanthology.org/L18-1079/)

  **Tao Ge**, Lei Cui, Baobao Chang, Zhifang Sui, Furu Wei, Ming Zhou
  

- ![](https://img.shields.io/badge/NLPCC-18-white?labelColor=orange) [**SeRI: A Dataset for Sub-event Relation Inference from an Encyclopedia**](https://link.springer.com/chapter/10.1007/978-3-319-99501-4_23)

  **Tao Ge**, Lei Cui, Baobao Chang, Zhifang Sui, Furu Wei, Ming Zhou
  

- ![](https://img.shields.io/badge/COLING-16-white?labelColor=red) [**Event detection with Burst Information Networks**](https://aclanthology.org/C16-1309/)

  **Tao Ge**, Lei Cui, Baobao Chang, Zhifang Sui, Ming Zhou
  

- ![](https://img.shields.io/badge/EMNLP-16-white?labelColor=red) [**News Stream Summarization using Burst Information Networks**](https://aclanthology.org/D16-1075/)

  **Tao Ge**, Lei Cui, Heng Ji, Baobao Chang, Sujian Li, Ming Zhou, Zhifang Sui
  

- ![](https://img.shields.io/badge/NLPCC-16-white?labelColor=orange) ![](https://img.shields.io/badge/Best%20Student%20Paper-gold) [**Discovering Concept-level Event Associations from a Text Stream**](https://link.springer.com/chapter/10.1007/978-3-319-50496-4_34)

  **Tao Ge**, Lei Cui, Heng Ji, Baobao Chang, Zhifang Sui

- ![](https://img.shields.io/badge/COLING-16-white?labelColor=red) [**Towards Time-aware Knowledge Graph Completion**](https://aclanthology.org/C16-1161/)

  Tingsong Jiang, Tianyu Liu, **Tao Ge**, Lei Sha, Baobao Chang, Sujian Li, Zhifang Sui
  

- ![](https://img.shields.io/badge/EMNLP-16-white?labelColor=red) [**Encoding Temporal Information for Time-aware Link Prediction**](https://aclanthology.org/D16-1260/)

  Tingsong Jiang, Tianyu Liu, **Tao Ge**, Lei Sha, Sujian Li, Baobao Chang, Zhifang Sui
  

- ![](https://img.shields.io/badge/ACL-15-white?labelColor=red) [**One Tense per Scene: Predicting Tense in Chinese Conversations**](https://aclanthology.org/P15-2110/)

  **Tao Ge**, Heng Ji, Baobao Chang, Zhifang Sui
  

- ![](https://img.shields.io/badge/ACL-15-white?labelColor=red) [**Bring you to the past: Automatic Generation of Topically Relevant Event Chronicles**](https://aclanthology.org/P15-1056/)

  **Tao Ge**, Wenzhe Pei, Heng Ji, Sujian Li, Baobao Chang, Zhifang Sui
  

- ![](https://img.shields.io/badge/ACL-15-white?labelColor=red) [**An Effective Neural Network Model for Graph-based Dependency Parsing**](https://aclanthology.org/P15-1031/)

  Wenzhe Pei, **Tao Ge**, Baobao Chang
  

- ![](https://img.shields.io/badge/BioNLP-15-white?labelColor=orange) [**Exploiting task-oriented resources to learn word embeddings for clinical abbreviation expansion**](https://aclanthology.org/W15-3810/)

  Yue Liu, **Tao Ge**, Kusum S Mathews, Heng Ji, Deborah McGuinness
  

- ![](https://img.shields.io/badge/ACL-14-white?labelColor=red) [**Max-Margin Tensor Neural Network for Chinese Word Segmentation**](https://aclanthology.org/P14-1028/)

  Wenzhe Pei, **Tao Ge**, Baobao Chang
  

- ![](https://img.shields.io/badge/WWW-14-white?labelColor=red) [**A semi-supervised method for opinion target extraction**](https://dl.acm.org/doi/10.1145/2567948.2577337)
  
  **Tao Ge**, Wenjie Li, Zhifang Sui
  

- ![](https://img.shields.io/badge/SIGHAN-14-white?labelColor=orange) [**The CIPS-SIGHAN CLP 2014 Chinese Word Segmentation Bake-off**](https://aclanthology.org/W14-6814/)

  Huiming Duan, Zhifang Sui, **Tao Ge**
  

- ![](https://img.shields.io/badge/CIKM-13-white?labelColor=red) [**Exploiting Collaborative Filtering Techniques for Automatic Assessment of Student Free-text Responses**](https://dl.acm.org/doi/10.1145/2505515.2507827)
  
  **Tao Ge**, Zhifang Sui, Baobao Chang
  

- ![](https://img.shields.io/badge/EMNLP-13-white?labelColor=red) [**Event-Based Time Label Propagation for Automatic Dating of News Articles**](https://aclanthology.org/D13-1001/)

  **Tao Ge**, Baobao Chang, Sujian Li, Zhifang Sui
