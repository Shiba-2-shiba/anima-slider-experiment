# Local Training Models

This directory is for local model weights used by the trainer. Do not commit
the weight files to GitHub.

Expected default layout:

```text
学習用モデル/
  Diffusion_model/
    copycatAnima_20260425.safetensors
  Textencorder/
    qwen_3_06b_base.safetensors
  VAE/
    qwen_image_vae.safetensors
```

The default config resolves these paths from the repository root through the
`{repo_root}` token.
