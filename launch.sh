# fast testting (RTX 4090)
CUDA_VISIBLE_DEVICES=7 python train.py --tokenizer-name ~/tokenizer/llama2-7b/ --parquet-path /nas/datasets/karpathy--fineweb-edu-100b/ --wandb-project aha --wandb-runname aha-0.4b --seqlen 2048 --ngpus 1 --micro-batch-size 1 --grad-accum-steps 4 --checkpoint-tokens 0.5 --eval-per-checkpoint 2 --parallel ddp --optimizer muon --optimizer-config ./optimizer.yaml --optimize-tokens 40 --eval-tasks arc_easy

# training on RTX 4090 (0.375M tokens per optim step)
# CUDA_VISIBLE_DEVICES=1,2,3,4 python train.py --tokenizer-name ~/tokenizer/llama2-7b/ --parquet-path /nas/datasets/karpathy--fineweb-edu-100b/ --log-to-wandb --wandb-project aha-0.4b --wandb-runname hybrid-0.4b --seqlen 2048 --ngpus 4 --micro-batch-size 6 --grad-accum-steps 8 --checkpoint-tokens 0.5 --eval-per-checkpoint 2 --parallel ddp --optimizer muon --optimizer-config ./optimizer.yaml --optimize-tokens 40 --eval-tasks arc_challenge,lambada_openai,niah_single_2

# training on H100 (0.5M tokens per optim step)
# python train.py --tokenizer-name /home/csy/tokenizer/llama2-7b --parquet-path /home/transformers/datasets/karpathy--fineweb-edu-100b --log-to-wandb --wandb-project aha-0.4b --wandb-runname delta-0.4b --seqlen 2048 --ngpus 2 --micro-batch-size 16 --grad-accum-steps 8 --checkpoint-tokens 0.5 --eval-per-checkpoint 2 --parallel ddp --optimizer muon --optimizer-config ./optimizer.yaml --optimize-tokens 40 --eval-tasks arc_challenge,lambada_openai,niah_single_2
