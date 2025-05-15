# Codes for GraphChain

This repository is the official implementation of GraphChain.

## Abstract

Large Language Models (LLMs) face significant limitations when applied to large-scale graphs, struggling with context constraints and inflexible reasoning. We present GraphChain, a framework that enables LLMs to analyze complex graphs through dynamic sequences of specialized tools, mimicking human exploratory intelligence. Our approach introduces two key innovations: (1) Progressive Graph Distillation, a reinforcement learning mechanism that generates optimized tool sequences balancing task relevance with information compression, and (2) Structure-aware Test-Time Adaptation, which efficiently tailors tool selection strategies to diverse graph topologies using spectral properties and lightweight adapters without costly retraining. Experiments show GraphChain significantly outperforms prior methods, enabling scalable and adaptive LLM-driven graph analysis.

## 0. Environment Setup

1. Create a new Conda environment:

    ```bash
    conda create -n GraphChain python=3.10.14
    ```

2. Activate the environment:

    ```bash
    conda activate GraphChain
    ```

3. Install [Pytorch](https://pytorch.org/get-started/locally/) and other required dependencies via `pip`:

```bash
    pip install torch torchvision torchaudio
    pip install -r requirements.txt
```
**Note**: Ensure that the version of GCC/G++ is >= 9.0.0.

## 1. SFT Dataset Construction
We curated 45 commonly used APIs from the NetworkX library based on relevance and usage frequency in graph-related tasks. To ensure diverse instruction coverage, we employed ChatGPT to generate various instructions tailored to these APIs. For each iteration, we randomly sampled APIs and prompted ChatGPT to reverse-engineer instructions centered around them, ensuring comprehensive coverage across the API set.

Run the following code to generate the raw Q&A pairs:
```bash
python dataset_construction/STF_data/QAdataset_Gen.py
```
Next, you need to slice each step of the raw Q&A pairs to fit the training mode for SFT:
```bash
python dataset_construction/STF_data/format_multiStep.py
```
Note: The API names (tools) are stored in the SFT_API-name.json file. When generating Q&A data, APIs will be randomly selected from this file to form combinations for targeted generation. The specific number of APIs can be controlled in the line selected_categories = random.sample(categories, 4). It is important to also modify the prompt content that the data generation relies on accordingly.

## 2. RL Dataset Construction

For the reinforcement learning phase, we constructed a dataset with reward values for each step. We used GPT-4 to score each step based on three dimensions:
    · API Correctness: Whether the tool invocation in the current step is valid.
    · Thought and API Effectiveness: The relevance of the tool selection to solving the query.
    · Graph Distillation: Whether the tool reduces the information content of the graph data.

We implement the construction of reinforcement learning data during the inference phase, for details see the infer_score folder.

## 3. SFT Training Phase

We provide comprehensive details on our experimental setup to ensure reproducibility. All experiments were conducted on 2 NVIDIA A800 80GB GPUs, using LoRA-based fine-tuning (rank r=16,alpha=32) on the Qwen2.5-7B-instruction model.
We used a learning rate of 5 × 10−5 with 4% warmup and a cosine scheduler for 8 epochs. This initial phase established the model’s ability to follow graph reasoning instructions.

We use the ms-swift framework for SFT training.
To install using pip:

```Plain Text
pip install ms-swift -U
```



Run the following code to perform SFT training:
```bash
scripts//sft_train.sh
```

## 4. RL Phase

We implemented Proximal Policy Optimization (PPO) with step-level rewards, departing from traditional RLHF approaches that apply rewards solely to the final step.

Run the following code to perform RL training:
```bash
scripts/ppo_train_qwen2.sh
```

Run the following code to train the STTA:
```bash
scripts/train_stta.sh
```

## 5. Inference Phase

After SFT and RL training, you can use the trained model for inference:

```bash
python infer_score/inference_and_score.py
```
Note: The dataset description in the prompt needs to be modified according to the graph dataset used during inference.