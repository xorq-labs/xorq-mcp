# Fine-Tuning Plan

## Goal

Fine-tune an open-weight LLM to improve task-specific performance (e.g., code generation, domain Q&A, structured output). This document covers the workflow on **Together.ai** as the primary provider, with alternatives compared below.

---

## 1. Data Preparation

### Format

Together.ai expects **JSONL** with a `messages` array (chat format):

```jsonl
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

For **DPO** (preference tuning), each line also needs `chosen` and `rejected` responses.

Parquet is also supported (pre-tokenized, skips tokenization step, useful for large datasets).

### Best Practices

- **Volume**: 500-10,000 examples is the sweet spot for LoRA; more for full fine-tuning
- **Quality over quantity**: curate aggressively, remove duplicates, fix formatting
- **Split**: 90/10 train/validation
- **Validate**: run `json.loads()` on every line, check for required keys
- **Strip unused fields**: Together packs samples for efficiency; extra fields slow upload
- **Context length**: most models now support 2-4x longer contexts than before (up to 131k tokens for some)

### Data Generation Strategy

If you lack training data:

1. Use a strong model (Claude, GPT-4o) to generate seed examples
2. Have domain experts review/correct a subset
3. Use the corrected subset as few-shot examples to generate more
4. Final human review pass

---

## 2. Together.ai: Model Selection

| Model | Params | LoRA Train $/1M tok | Full Train $/1M tok | Good For |
|-------|--------|---------------------|----------------------|----------|
| Llama 3.1 8B | 8B | $0.48 | $0.54 | Fast iteration, cost-sensitive |
| Qwen 2.5 14B | 14B | $0.48 | $0.54 | Strong multilingual, coding |
| Llama 3.1 70B | 70B | $2.90 | $3.20 | Maximum quality, complex tasks |
| DeepSeek-R1 | 671B (MoE) | varies | varies | Reasoning-heavy tasks |
| Qwen3-235B | 235B | varies | varies | Large-scale, multilingual |

**Recommendation**: Start with **Llama 3.1 8B LoRA** for fast/cheap iteration, then scale to 70B once your data pipeline is proven.

---

## 3. Together.ai: Training Workflow

### Step 1: Upload Data

```python
import together

# Upload training file
resp = together.Files.upload(file="train.jsonl")
training_file_id = resp["id"]

# Upload validation file
resp = together.Files.upload(file="val.jsonl")
val_file_id = resp["id"]
```

### Step 2: Launch Fine-Tuning Job

```python
resp = together.Fine_tuning.create(
    training_file=training_file_id,
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    # LoRA config
    training_type={"type": "lora", "lora_r": 16, "lora_alpha": 32},
    n_epochs=3,
    learning_rate=1e-5,
    batch_size=4,
    validation_file=val_file_id,
    suffix="my-task-v1",
)
job_id = resp["id"]
```

### Step 3: Monitor

```python
status = together.Fine_tuning.retrieve(job_id)
# or via CLI: together fine-tuning retrieve <job_id>
```

### Step 4: Inference

```python
resp = together.Complete.create(
    model=f"<your-org>/{status['output_name']}",
    prompt="...",
)
```

### DPO (Preference Tuning)

- Costs **2.5x** the SFT rate
- Supports variants: standard DPO, LN-DPO, DPO+NLL, SimPO
- Use when you have preference data (chosen vs. rejected outputs)
- Often run as a second stage after initial SFT

### Cost Estimation

For a 5,000-example dataset (~2M tokens), 3 epochs on Llama 3.1 8B LoRA:
- Training tokens: 2M x 3 = 6M
- Cost: 6 x $0.48 = **~$2.88**

Same dataset on Llama 3.1 70B LoRA:
- Cost: 6 x $2.90 = **~$17.40**

---

## 4. Together.ai: Strengths & Weaknesses

**Strengths**
- No minimum spend; pay only for tokens processed
- Wide model catalog including 100B+ models
- LoRA + full fine-tuning + DPO all supported
- Serverless Multi-LoRA: deploy hundreds of adapters on one base model
- Hugging Face Hub integration (bring your own model)
- Long context support (up to 131k tokens)

**Weaknesses**
- Less control over training infrastructure vs. self-hosted
- DPO at 2.5x markup can get expensive
- No custom training loop / callback support
- Proprietary models (GPT, Claude) not available for fine-tuning here

---

## 5. Alternative Providers

### Fireworks.ai

| Aspect | Details |
|--------|---------|
| **Pricing** | ~$0.50/1M tokens (up to 16B), competitive with Together |
| **Methods** | LoRA, full fine-tuning, reinforcement fine-tuning, quantization-aware training |
| **Key advantage** | Fine-tuned models served at **base model inference prices** |
| **Models** | Llama 3.1, DeepSeek R1, 1T+ parameter support |
| **Best for** | Teams that want fine-tune + deploy on the same platform cheaply |

**Verdict**: Strong alternative to Together. The same-price inference for fine-tuned models is a real cost saver in production.

### OpenAI

| Aspect | Details |
|--------|---------|
| **Pricing** | GPT-4o-mini: $3.00/1M training tokens; GPT-4o: $25.00/1M |
| **Methods** | SFT only (no LoRA/DPO exposed) |
| **Key advantage** | Access to GPT-4o family; best if you're already in the OpenAI ecosystem |
| **Limitations** | Proprietary, no model weights, vendor lock-in, higher price |
| **Best for** | Teams committed to OpenAI that need incremental improvement on GPT-4o-mini |

**Verdict**: Expensive and locked-in compared to open-weight alternatives. GPT-4o-mini fine-tuning at $3/1M tokens is 6x the cost of Together's 8B LoRA. Only choose this if you specifically need a GPT model.

### Modal

| Aspect | Details |
|--------|---------|
| **Pricing** | Per-second GPU billing: H100 ~$3.95/hr, A100 80GB ~$2.50/hr |
| **Methods** | Full control: any framework (Axolotl, HF Trainer, PyTorch) |
| **Key advantage** | Maximum flexibility; bring your own training code |
| **Limitations** | You manage the training loop, hyperparams, everything |
| **Best for** | Teams with ML engineering capacity who want full control |

**Verdict**: Best "DIY with guardrails" option. You write the training script, Modal handles infra. More work but total control over the process. Cost-effective for large jobs since you pay for raw GPU time.

### Anyscale (Ray)

| Aspect | Details |
|--------|---------|
| **Pricing** | Enterprise pricing, typically negotiated |
| **Methods** | Ray Train, DeepSpeed, FSDP |
| **Key advantage** | Scales to massive multi-node training |
| **Limitations** | Enterprise-oriented, overkill for small jobs |
| **Best for** | Large orgs doing multi-node fine-tuning at scale |

**Verdict**: Skip unless you're training at enterprise scale with a dedicated ML team.

---

## 6. Provider Comparison Matrix

| | Together.ai | Fireworks.ai | OpenAI | Modal | Anyscale |
|--|-------------|--------------|--------|-------|----------|
| **Ease of use** | High | High | Highest | Medium | Low |
| **Cost (small job)** | $$ | $$ | $$$$ | $$ | $$$$$ |
| **Cost (large job)** | $$$ | $$$ | $$$$$ | $$ | $$$ |
| **Model freedom** | Open-weight | Open-weight | GPT only | Any | Any |
| **DPO support** | Yes | Yes | No | DIY | DIY |
| **Inference bundled** | Yes | Yes (same price!) | Yes | No (separate) | No |
| **Control** | Low | Low | Lowest | Highest | High |

---

## 7. Recommended Approach

### Phase 1: Validate the Idea (budget: <$10)

1. Prepare 500-1,000 high-quality examples in JSONL chat format
2. Fine-tune **Llama 3.1 8B LoRA** on **Together.ai**
3. Evaluate against base model on a held-out test set
4. If improvement is marginal, invest in better data before scaling up

### Phase 2: Scale Quality (budget: <$50)

1. Expand dataset to 3,000-5,000 examples
2. Try **Qwen 2.5 14B** or **Llama 3.1 70B** LoRA on Together
3. Experiment with DPO if you have preference data
4. A/B test fine-tuned vs. base model in real usage

### Phase 3: Production (budget: varies)

1. If deploying via API: stay on Together or move to **Fireworks** (free inference upgrade for fine-tuned models)
2. If deploying self-hosted: export weights and serve with vLLM/TGI
3. If you need full control of training: move to **Modal** with Axolotl/HF Trainer
4. Set up continuous fine-tuning pipeline as new data comes in

---

## 8. Sources

- [Together.ai Pricing](https://www.together.ai/pricing)
- [Together.ai Fine-Tuning Pricing Docs](https://docs.together.ai/docs/fine-tuning-pricing)
- [Together.ai Fine-Tuning Platform Updates (Sept 2025)](https://www.together.ai/blog/fine-tuning-updates-sept-2025)
- [Together.ai Data Preparation Docs](https://docs.together.ai/docs/fine-tuning-data-preparation)
- [Together.ai Fine-Tuning Product Page](https://www.together.ai/fine-tuning)
- [Fireworks.ai Pricing](https://fireworks.ai/pricing)
- [Fireworks.ai Fine-Tuning Blog](https://fireworks.ai/blog/fine-tune-launch)
- [Modal Pricing](https://modal.com/pricing)
- [OpenAI Pricing](https://platform.openai.com/docs/pricing)
- [Fine-Tuning Landscape 2025 (Medium)](https://medium.com/@pradeepdas/the-fine-tuning-landscape-in-2025-a-comprehensive-analysis-d650d24bed97)
- [Top Fine-Tuning Platforms for Enterprises 2026](https://www.siliconflow.com/articles/en/the-top-fine-tuning-platforms-for-enterprises)
