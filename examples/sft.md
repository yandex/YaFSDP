# Supervised fine-tuning example

This command launches a distributed fine-tuning setup using 🤗 trl, transformers and accelerate libraries.

```bash
docker run \
    -it \
    --rm \
    --net host \
    --gpus '"device=0,1"' \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    ya-fsdp:latest \
    accelerate launch \
        --config_file ya-fsdp/examples/fsdp_config.yaml \
        --fsdp_ya_fsdp_enabled true \
        trl/examples/scripts/sft.py \
            --do_train \
            --model_name_or_path meta-llama/Meta-Llama-3-8B \
            --max_steps 5 \
            --block_size 2048 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --dataset_name timdettmers/openassistant-guanaco \
            --save_strategy no \
            --logging_steps 1 \
            --report_to tensorboard \
            --output_dir sft
```

See `examples/clm.md` for tips on some of the options.
