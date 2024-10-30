#!/bin/bash
#SBATCH -A IscrC_Ge-Di
#SBATCH --time 24:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gres=gpu:4        # 4 gpus per node out of 4
#SBATCH --mem=48000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=delta_energy_llama2
#SBATCH --error=delta_energy.err            # standard error file
#SBATCH --output=delta_energy.out           # standard output file
#SBATCH --partition=boost_usr_prod

export HF_DATASETS_CACHE="$WORK/LLM-HE/MetaMath/HF_CACHE/meta-math___meta_math_qa"

cd $WORK/LLM-HE/MetaMath

nvidia-smi > gpu_info.txt

conda activate
conda init
conda activate delta-energy

export MODEL_PATH='models/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9'
export SAVE_PATH='checkpoints'
export MASTER_ADDR="localhost"
export MASTER_PORT="1231"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true
wandb offline

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=8 --use_env train_math.py \
    --model_name_or_path $MODEL_PATH \
    --data_path "meta-math/MetaMathQA" \
    --data_length 10000000 \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True

python eval_gsm8k.py --model $SAVE_PATH --data_path ./data/test/GSM8K_test.jsonl
python eval_math.py --model $SAVE_PATH --data_path ./data/test/MATH_test.jsonl
