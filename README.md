# anima-slider

Anima/Cosmos RFlow 向けの text-only Slider LoRA 学習を試すための実験リポジトリです。

このリポジトリは研究・検証目的の実験コードです。学習 recipe、prompt、loss、評価方法は今後変更される可能性があります。

ComfyUI の Anima 実装を使って diffusion model を読み込み、target prompt の denoising trajectory 上で FLUX-style の teacher direction を学習します。現在の主な学習入口は `scripts/anima_train_lora_flow_slider.py` です。

## 参考にしたリポジトリ

この実装は、以下の GitHub リポジトリを参考にしています。

- [rohitgandikota/sliders](https://github.com/rohitgandikota/sliders): concept slider / text slider の考え方、positive と unconditional の差分方向を使う設計の参考
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts): LoRA 学習実装、target module の扱い、学習スクリプト構成の参考

本リポジトリのコードは Anima/Cosmos RFlow と ComfyUI の実装に合わせて独自に組み直しています。

## 構成

```text
ComfyUI/       ComfyUI 本体。Git submodule
configs/       モデルパスと LoRA target preset
prompts/       Slider 学習用 prompt YAML
scripts/       cache 作成、LoRA 学習、ComfyUI 推論、比較用スクリプト
src/           trainer の実装
tests/         unit tests
workflows/     ComfyUI API workflow
学習用モデル/  手動配置するローカルモデル置き場
cache/         conditioning cache 出力先
models/        学習済み LoRA 出力先
reports/       学習 report 出力先
```

`cache/`, `models/`, `reports/` はフォルダだけ GitHub に含め、生成物は `.gitignore` で除外します。`学習用モデル/` のモデル重みも GitHub には含めません。

## 必要条件

- Python 3.10 以上
- CUDA 対応 GPU 推奨
- Git submodule が使える Git 環境
- 手動で入手した Anima diffusion model、Qwen text encoder、VAE

`ComfyUI/` は必須です。cache 作成は ComfyUI の text encoder loader を使い、LoRA 学習は ComfyUI の diffusion model 実装を使うため、`ComfyUI/` がない状態では学習できません。

## クローン

submodule も同時に取得する場合:

```powershell
git clone --recurse-submodules https://github.com/Shiba-2-shiba/anima-slider-experiment.git
cd anima-slider-experiment
```

通常 clone 済みの場合:

```powershell
cd anima-slider-experiment
git submodule update --init --recursive
```

ComfyUI の依存関係を入れます。

```powershell
python -m pip install -r ComfyUI\requirements.txt
python -m pip install -e .
```

## モデル配置

次の場所にモデルファイルを手動で配置してください。

```text
学習用モデル/
  Diffusion_model/
    copycatAnima_20260425.safetensors
  Textencorder/
    qwen_3_06b_base.safetensors
  VAE/
    qwen_image_vae.safetensors
```

default config は以下を参照します。

```text
{repo_root}/学習用モデル/Diffusion_model/copycatAnima_20260425.safetensors
{repo_root}/学習用モデル/Textencorder/qwen_3_06b_base.safetensors
{repo_root}/学習用モデル/VAE/qwen_image_vae.safetensors
{repo_root}/ComfyUI
```

配置確認:

```powershell
python scripts\check_local_assets.py
```

別の diffusion model を使う場合は、`configs/config-anima-slider.yaml` を編集するか、学習時に `--model` を指定してください。`--model other.safetensors` のようなファイル名だけの指定は、config の diffusion model と同じディレクトリから解決されます。

## テスト

```powershell
python -m unittest discover -s tests
```

## cache 作成

Age slider v2 prompt の conditioning cache を作成します。

```powershell
python scripts\cache_anima_conditioning.py `
  --config_file configs\config-anima-slider.yaml `
  --prompts_file prompts\prompts-anima-age_slider_v2.yaml `
  --output cache\anima-age-conditioning-v2.pt `
  --allow_unsafe_age_terms
```

実際に cache を作らず prompt 設定だけ確認する場合:

```powershell
python scripts\cache_anima_conditioning.py `
  --config_file configs\config-anima-slider.yaml `
  --prompts_file prompts\prompts-anima-age_slider_v2.yaml `
  --dry_run `
  --allow_unsafe_age_terms
```

## LoRA 学習

現在の推奨 recipe は `run04` 相当です。

```powershell
python scripts\anima_train_lora_flow_slider.py `
  --config_file configs\config-anima-slider.yaml `
  --cache cache\anima-age-conditioning-v2.pt `
  --prompt_indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 `
  --eval_prompt_indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 `
  --steps 600 `
  --lr 0.000005 `
  --width 512 `
  --height 512 `
  --num_inference_steps 20 `
  --timestep_sampling shift `
  --sigmoid_scale 1.0 `
  --discrete_flow_shift 3.0 `
  --loss_weighting_scheme none `
  --eta 1.0 `
  --vary_seed `
  --output_lora models\age_slider_flow_run04_shift3_eta10\anima_age_slider_flow_run04_shift3_eta10.safetensors `
  --output_report reports\age-slider-flow-run04-shift3-eta10.json
```

出力:

```text
models/age_slider_flow_run04_shift3_eta10/anima_age_slider_flow_run04_shift3_eta10.safetensors
reports/age-slider-flow-run04-shift3-eta10.json
```

別モデルで学習する例:

```powershell
python scripts\anima_train_lora_flow_slider.py `
  --config_file configs\config-anima-slider.yaml `
  --model otherAnimaModel.safetensors `
  --cache cache\anima-age-conditioning-v2.pt `
  --output_lora models\other_model_age_slider\anima_age_slider.safetensors `
  --output_report reports\other-model-age-slider.json
```

`--diffusion_model` も `--model` の alias として使えます。

## ComfyUI で確認

学習した LoRA を local ComfyUI の LoRA フォルダに置きます。

```powershell
New-Item -ItemType Directory -Force ComfyUI\models\loras
Copy-Item `
  models\age_slider_flow_run04_shift3_eta10\anima_age_slider_flow_run04_shift3_eta10.safetensors `
  ComfyUI\models\loras\anima_age_slider_flow_run04_shift3_eta10.safetensors `
  -Force
```

ComfyUI API を起動します。

```powershell
python ComfyUI\main.py --listen 127.0.0.1 --port 8000
```

別 terminal から strength grid を実行します。

```powershell
python scripts\anima_lora_strength_grid.py `
  --workflow workflows\anima_lora_model_only.json `
  --url http://localhost:8000 `
  --lora_name anima_age_slider_flow_run04_shift3_eta10.safetensors `
  --strengths=-1,-0.5,-0.25,0,0.25,0.5,0.75,1 `
  --seed 961218314523996 `
  --steps 20 `
  --width 1024 `
  --height 1024 `
  --filename_prefix Anima\age_slider_flow_run04_shift3_eta10_grid_mild
```

PowerShell では negative strength が option として解釈されることを避けるため、`--strengths=...` のように `=` 付きで指定してください。

## 学習の概要

flow slider trainer は次の teacher direction を学習します。

```text
teacher = target_base + eta * (positive_base - unconditional_base)
teacher = normalize_like(teacher, positive_base)
loss = MSE(model_pred_with_lora, teacher)
```

保存される LoRA は ComfyUI の `LoraLoaderModelOnly` で読み込める `safetensors` 形式です。

## ライセンスと公開時の注意

`ComfyUI/` は Git submodule です。ComfyUI 本体は GPL-3.0 です。

このリポジトリで GitHub に含める想定のもの:

- trainer code
- configs
- prompts
- scripts
- tests
- workflows
- ComfyUI submodule reference
- `cache/`, `models/`, `reports/` の空フォルダ

GitHub に含めないもの:

- `学習用モデル/` 配下のモデル重み
- `cache/` 内の `.pt`
- `models/` 内の LoRA `.safetensors`
- `reports/` 内の生成 report や画像
- `.omx/`
- `docs/`
