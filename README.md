# RLHF

RLHF is a low-cost and efficient large language model training system including SFT, reward training and rlhf.

Feature：

- Deepspeed + TRL + QLoRA + Flash-Attntion 2
- RLHF-PPO : LLaMA-2-13B ZeRO-1 RLHF 8xA800-40GB 
- DPO: Mistral7B + DPO 1h+ in 3090x4
- Unsloth: Support LLM Finetune with Unsloth


## RLHF Pipeline & Dataset

- `Pretrained`: imdb, 20k
- `SFT`: yahma/alpaca-cleaned, 52k
- `Reward Model`: Anthropic/hh-rlhf, 160k
- `PPO`: Anthropic/hh-rlhf, 160k 
  - TODO: PPO train data use alpaca


## Installation

Git Clone RLHF

```
git clone git@github.com:Kryptonions/RLHF.git
cd RLHF
```

Create Dev Environment

```bash
conda create -n llm python=3.11
conda activate llm
pip install -r requirements.txt
pip install flash-attn
```

Create Unsloth Environment

```bash
conda create -n llm_unsloth python=3.11
conda activate llm_unsloth
pip install --upgrade pip
# install pytorch 2.1.0 and triton on cu118 or cu121 
pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.0 triton \
  --index-url https://download.pytorch.org/whl/cu118
# install packaging
pip install packaging -i https://pypi.tuna.tsinghua.edu.cn/simple
export CUDA_HOME=/usr/local/cuda-11.7
# install unsloth
pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git" -i https://pypi.tuna.tsinghua.edu.cn/simple
# install flash-attn manually (torch vesion + python version)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.1.post1/flash_attn-2.5.1.post1+cu118torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
 pip install flash_attn-2.5.1.post1+cu118torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 
```

Setting Environment

```bash
export WANDB_API_KEY={YOU_WANDB_TOKEN} # from https://wandb.ai/authorize
# export HF_ENDPOINT=https://hf-mirror.com
# export NCCL_P2P_DISABLE="1" # for GPU 3090/4090
# export NCCL_IB_DISABLE="1"  # for GPU 3090/4090
```

DeepSpeed Test

```bash
deepspeed ./test/test_QLoRA.py
```

- Deepspeed config json is `./config/ds.json`

## Quick Start

Test with 

```bash
./scripts/run_all.sh
```

Llama-2-13B

```
./scripts/run_all_13B.sh
```

Llama-2-7B

```
./scripts/run_all_7B.sh
```

🚀 Llama-2-7B-DPO

```
./scripts/run_all_7B_dpo.sh
```

🚀 Baichuan2-SFT

```
./scripts/run_all_7b_sft_baichuan2.sh
```

🚀 Unsloth-SFT

```
./scripts/run_7b_sft_unsloth.sh
```

### 0. run path

```bash
base_model_path='meta-llama/Llama-2-7b-hf'
deepspeed_config_name=./config/ds.json
output_path='./output'

model_pretrained_lora_path=${output_path}'/pretrained_lora'
model_pretrained_full_path=${output_path}'/pretrained_full'
model_sft_lora_path=${output_path}'/sft_lora'
model_sft_full_path=${output_path}'/sft_full'
model_reward_model_lora_path=${output_path}'/reward_model_lora'
model_ppo_lora_path=${output_path}'/ppo_lora'
model_ppo_full_path=${output_path}'/ppo_full'
```

### 1. Pretrained(Optional)

```bash
# stage: second pretrained
pt_dataset_name='imdb'
deepspeed ./rlhf/pretrained.py \
	--dataset_name=${pt_dataset_name} \
	--model_name=${base_model_path} \
	--seq_length=512 \
	--batch_size=16 \
	--output_name=${model_pretrained_lora_path} \
	--use_QLora=True \
	--use_flash_attention_2=True \
	--deepspeed_config_name=${deepspeed_config_name} \
	--deepspeed=${deepspeed_config_name} \
	--num_train_epochs=1
```

### 2. SFT

```bash
sft_dataset_name='yahma/alpaca-cleaned'
model_pretrained_full_path=${base_model_path}
deepspeed ./rlhf/sft.py \
	--dataset_name=${sft_dataset_name} \
	--model_name=${model_pretrained_full_path} \
	--seq_length=512 \
	--output_name=${model_sft_lora_path} \
	--use_QLora=True \
	--batch_size=16 \
	--use_flash_attention_2=True \
	--deepspeed_config_name=${deepspeed_config_name} \
	--num_train_epochs=1
```

### 3. Reward Model

```bash
# stage reward model
rm_dataset_name='Anthropic/hh-rlhf'
deepspeed ./rlhf/reward_model.py \
	--dataset_name=${rm_dataset_name} \
	--model_name=${model_sft_full_path} \
	--seq_length=512 \
	--batch_size=8 \
	--output_name=${model_reward_model_lora_path} \
	--use_QLora=True \
	--use_flash_attention_2=True \
	--deepspeed_config_name=${deepspeed_config_name} \
	--num_train_epochs=1
```

### 4. PPO

```bash
# stage ppo
rm_dataset_name='Anthropic/hh-rlhf'
deepspeed ./rlhf/ppo.py \
	--dataset_name=${rm_dataset_name} \
	--model_name=${model_sft_full_path} \
	--reward_model_name=${model_reward_model_lora_path} \
	--output_name=${model_ppo_lora_path} \
	--use_QLora=True \
	--use_flash_attention_2=True \
	--deepspeed_config_name=${deepspeed_config_name} \
	--batch_size=4 \
	--mini_batch_size=2 \
	--ppo_epochs=1 \
	--output_max_length=256 \
	--seq_length=512
```


### 5. Generate Test

```bash
echo "------------------print ppo result------------------"
python ./rlhf/generate.py \
	--model_name=${model_ppo_full_path} \
	--prompt='how to make a bomb?' \
	--max_new_token=128
```





