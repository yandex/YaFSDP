# FSDP to YaFSDP migration guide

Sharding a model with FSDP might look similar to the example below:

```python
model: LlamaForCausalLM = ...

FSDP(
    model,
    sharding_strategy=sharding_strategy,
    auto_wrap_policy=partial(
        lambda_auto_wrap_policy,
        lambda_fn=lambda m: (
            m is model.model.embed_tokens
            or m in model.model.layers
            or m is model.lm_head
        ),
    ),
    mixed_precision=MixedPrecision(param_dtype=param_dtype, reduce_dtype=torch.float32),
    sync_module_states=sync_module_states,
    param_init_fn=param_init_fn,
    device_id=device,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    forward_prefetch=True,
    use_orig_params=True,
)
```

An equivalent YaFSDP sharding looks like this:

```python
model: LlamaForCausalLM = ...

YaFSDP(
    model,
    zero_stage={
        "ShardingStrategy.FULL_SHARD": 3,
        "ShardingStrategy.SHARD_GRAD_OP": 2
    }[sharding_strategy],
    modules_to_wrap_with_names=[
        (model.model.embed_tokens, "model.embed_tokens"),
        *((m, f"model.layers.{i}") for i, m in enumerate(model.model.layers)),
        (model.lm_head, "lm_head")
    ],
    rogue_layer_norm_modules_with_names={model.norm: "model.norm"}
    layer_norm_module_cls=LlamaRMSNorm,
    param_dtype=param_dtype,
    sync_module_states=sync_module_states,
    param_init_fn=param_init_fn,
    device_id=device,
    gradient_accumulation_steps=gradient_accumulation_steps,
)
```

A major interface difference is in `auto_wrap_policy` as it is replaced with 3
arguments:

- `modules_to_wrap_with_names` — an explicit list of modules to shard (with
  their names to be used in state dict)
- `rogue_layer_norm_modules_with_names` — the first layer after all transformer
  blocks, which contains only layer norm parameters (it typically does) (with
  its name)
- `layer_norm_module_cls` — type of layer norm layers

Also YaFSDP requires the number of gradient accumulation steps to be explicitly
provided.
