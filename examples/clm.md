# Causal LM pre-training example

This command launches a distributed pre-training setup using 🤗 transformers and accelerate libraries.

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
        transformers/examples/pytorch/language-modeling/run_clm.py \
            --do_train \
            --config_name meta-llama/Meta-Llama-3-8B \
            --tokenizer_name meta-llama/Meta-Llama-3-8B \
            --max_steps 5 \
            --block_size 2048 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --dataset_name wikitext \
            --dataset_config_name wikitext-2-raw-v1 \
            --save_strategy no \
            --logging_steps 1 \
            --report_to tensorboard \
            --output_dir clm
```

CLI options:

- `--gpus '"device=0,1"'` – limit the number of devices used by each host or set
  to `all` to use all available devices.
- `--fsdp_ya_fsdp_enabled true` – toggle between FSDP and YaFSDP
- `--(config_name|tokenizer_name|model_name) meta-llama/Meta-Llama-3-8B`  –
  specify any model available at 🤗 hub or provide a path to you local model
  folder.
- `--max_steps 5` — specify number of training steps.
- `--block_size 2048` – specify input sequence length.
- `--per_device_(train|eval)_batch_size 1` – specify train/eval batch size
- `--dataset_name wikitext` – specify any publicly available dataset from the 🤗
  dataset library.
- `--save_strategy no` – specify saving strategy `(no|steps)`.

`fsdp_config.yaml` options:

- `fsdp_state_dict_type` — choose between `FULL_STATE_DICT` and
  `LOCAL_STATE_DICT` to save a global gathered state or local sharded states.
- `fsdp_activation_checkpointing` — toggle activation checkpointing.
- `fsdp_num_layers_to_checkpoint` — specify number of layers to checkpoint.
- `num_processes` — specify total number of training processes (`number or hosts
  x number of devices on each host`)
