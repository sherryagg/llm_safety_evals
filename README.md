# LLM Safety Evaluation Framework

Systematic evaluation of AI safety alignment using Anthropic's HH-RLHF dataset, comparing rule-based classification against LLM-as-judge (Claude 3 Sonnet) approaches.

## Overview

This project investigates how well simple keyword-based rules detect unsafe content compared to contextual LLM evaluation. Using 100 examples from the [Anthropic HH-RLHF harmless-base dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf), we:

1. **Parse multi-turn conversations** — extracting context, final prompts, and chosen/rejected responses
2. **Apply rule-based classification** — keyword matching for harmful intent and refusal detection
3. **Deploy Claude-as-judge** — LLM evaluation scoring helpfulness, harmlessness, and refusal correctness
4. **Compare with and without context** — measuring how conversation history changes safety classifications

## Key Findings

| Metric | Result |
|--------|--------|
| Rule-based classification error rate | **87.5%** (missed 77/88 unsafe prompts labelled "safe") |
| Examples where context changed classification | **18%** (18/100 shifted category between Run 1 and Run 2) |
| Severe safety violations discovered | **7** (prompts involving violence, assault, or exploitation) |

### Results Comparison: Run 1 (No Context) vs Run 2 (Full Context)

| Category | Run 1 | Run 2 | Change |
|----------|-------|-------|--------|
| unsafe | 67 | 74 | +7 |
| unsafe\|unsafe_bias | 12 | 10 | -2 |
| unsafe_severe | 0 | 7 | +7 |
| ambiguous | 9 | 1 | -8 |
| safe | 8 | 5 | -3 |
| context_dependent | 4 | 2 | -2 |
| unsafe_bias | 0 | 1 | +1 |

**Takeaway:** Providing conversation context resolves ambiguity — most "ambiguous" and "context_dependent" cases resolved to "unsafe" when the full conversation was visible.

## Methodology

```
HH-RLHF Dataset (100 samples)
        │
        ▼
┌─────────────────┐
│ Parse Multi-Turn │
│ Conversations    │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────┐
│ Rule   │ │ Claude Judge │
│ Based  │ │ (Bedrock)    │
└───┬────┘ └──────┬───────┘
    │              │
    ▼              ▼
┌─────────────────────────┐
│ Compare Classifications │
│ + Score Analysis        │
└─────────────────────────┘
```

## Tech Stack

- **Python 3.9**
- **AWS Bedrock** — Claude 3 Sonnet model access
- **Claude 3 Sonnet** (`anthropic.claude-3-sonnet-20240229-v1:0`) — LLM-as-judge
- **HuggingFace Datasets** — Anthropic HH-RLHF data loading
- **Pandas** — Data manipulation and analysis
- **Matplotlib** — Visualization

## Project Structure

```
llm_safety_evals/
├── notebooks/
│   └── 01_safety_evaluation.ipynb   # Main analysis notebook
├── results/
│   ├── rule_vs_claude_comparison.png
│   └── run1_vs_run2_comparison.png
├── data/                            # Downloaded datasets (gitignored)
├── src/                             # Utility modules (future)
├── .gitignore
└── README.md
```

## Setup

```bash
# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install datasets pandas matplotlib boto3

# Configure AWS credentials for Bedrock access
# (requires appropriate IAM role)
```

## Next Steps

- [ ] Expand evaluation to full HH-RLHF dataset (44k examples)
- [ ] Add inter-rater reliability metrics (Cohen's kappa between runs)
- [ ] Test with Claude 3.5 Sonnet / Claude 4 for judge quality comparison
- [ ] Build automated pipeline for continuous safety benchmarking
- [ ] Extract reusable evaluation functions into `src/` modules
- [ ] Add confidence calibration analysis for LLM judge scores
