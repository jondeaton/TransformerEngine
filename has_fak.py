import os

os.environ["NVTE_FUSED_ATTN"] = "1"  # make sure it uses fused attn

import jax
import jax.numpy as jnp
import transformer_engine.jax as te

print(jax.devices())


rng = jax.random.PRNGKey(0)
B, L, H, D = 16, 512, 8, 32

dtype = jnp.bfloat16
# dtype = jnp.float32

with te.fp8_autocast(enabled=True):

    q = jax.random.normal(rng, shape=(B, L, H, D), dtype=dtype)
    k = jax.random.normal(rng, shape=(B, L, H, D), dtype=dtype)
    v = jax.random.normal(rng, shape=(B, L, H, D), dtype=dtype)

    attn = te.flax.transformer.DotProductAttention(
        num_attention_heads=H,
        head_dim=D,
        attn_mask_type="no_mask",  # type: ignore
        dtype=q.dtype,
        transpose_batch_sequence=False,
        num_gqa_groups=H,
    )
    o = attn.apply({}, q, k, v)
