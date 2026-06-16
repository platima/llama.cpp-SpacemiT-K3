# Small Multimodal GGUF Models: Local Inference Test Set

A working reference for a local inference comparison of small multimodal GGUF models on RISC-V (SpacemiT) hardware. It covers the Qwen3.5 and Gemma 4 families, plus two hosted models included only as quality reference points. Throughput is measured locally; the benchmark accuracy figures in this document are published vendor or third-party numbers, collated and sourced below, not results produced by this test set.

Read the Disclaimers section before quoting any accuracy number from here.

## Scope and intent

The set is built to answer practical questions for edge deployment, not to crown a winner:

- How much real throughput does Multi-Token Prediction (MTP) buy on these CPUs, isolated by testing the same model and quant with and without the MTP build.
- How Google's official QAT Q4_0 compares against Unsloth's dynamic QAT on the same Gemma weights.
- How QAT versus non-QAT compares at the 12B size.
- Where the small open models actually sit against two hosted reference models on published reasoning and knowledge benchmarks.

## Models under test

All files are GGUF. "Vision (mmproj)" notes whether an mmproj projector is present so vision can be tested. "MTP / draft" notes whether a draft head or drafter file is present in the folder. In current llama.cpp, vision and MTP/draft cannot be active in the same run, so the two are tested separately.

| File | Base model | Vendor | Quant | Params (nominal / llama.cpp) | QAT | Vision (mmproj) | MTP / draft | Reason for selection |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-2B-Q4_1.gguf | Qwen3.5-2B | Unsloth | Q4_1 | 2B / 1.88B | No | Yes | No | Smallest dense model; Q4_1 chosen to match SpacemiT's official benchmark quant; non-MTP baseline |
| Qwen3.5-2B-Q4_1.gguf (MTP repo) | Qwen3.5-2B | Unsloth | Q4_1 | 2B / 1.88B | No | Present (not usable with MTP) | Yes | Same model and quant with MTP enabled; isolates draft-mtp speedup on a 2B dense model |
| Qwen3.5-4B-Q4_K_M.gguf | Qwen3.5-4B | Unsloth | Q4_K_M | 4B | No | Present (not usable with MTP) | No | Mid dense model, non-MTP baseline |
| Qwen3.5-4B-Q4_K_M.gguf (MTP repo) | Qwen3.5-4B | Unsloth | Q4_K_M | 4B | No | Present (not usable with MTP) | Yes | Mid dense model, MTP build; perf and size sweet spot with speculative decode |
| Qwen3.5-9B-Q4_K_M.gguf | Qwen3.5-9B | Unsloth | Q4_K_M | 9B | No | Yes | No | Largest edge-feasible dense Qwen; standard-decode baseline |
| Qwen3.5-9B-Q4_K_M.gguf (MTP repo) | Qwen3.5-9B | Unsloth | Q4_K_M | 9B | No | Present (not usable with MTP) | Yes | Same model and quant with MTP; isolates draft-mtp speedup at 9B |
| gemma-4-E2B_q4_0-it.gguf | Gemma 4 E2B | Google | Q4_0 (QAT) | E2B (eff. 2B) | Yes | Yes | None published | Google's official QAT reference quant; pairs with the Unsloth E2B QAT to compare quantisation method |
| gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf | Gemma 4 E2B | Unsloth | UD-Q4_K_XL (QAT) | E2B (eff. 2B) | Yes | Yes | Yes (mtp file present) | Unsloth dynamic QAT versus Google's official QAT on the same base weights |
| gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf | Gemma 4 E4B | Unsloth | UD-Q4_K_XL (QAT) | E4B (eff. 4B) / 7.52B | Yes | Yes | Yes (mtp file present) | Larger edge Gemma, audio-capable; Unsloth dynamic QAT |
| gemma-4-12b-it-Q4_K_M.gguf | Gemma 4 12B | Unsloth | Q4_K_M | 12B | No | Yes | Yes (mtp file present) | Largest Gemma here; non-QAT quality and speed reference against the QAT 12B |
| gemma-4-12B-it-qat-UD-Q4_K_XL.gguf | Gemma 4 12B | Unsloth | UD-Q4_K_XL (QAT) | 12B | Yes | Yes | None published | QAT versus non-QAT 12B at low bpw (KLD, quality and speed) |
| Haiku 4.5 | Haiku 4.5 | Anthropic (hosted) | n/a | undisclosed | n/a | n/a | n/a | Reference only: hosted small-tier quality ceiling |
| Gemini 3.1 Flash Lite | Gemini 3.1 Flash Lite | Google (hosted) | n/a | undisclosed | n/a | n/a | n/a | Reference only: hosted efficiency-tier multimodal, 1M context |

Notes:
- A UD-Q2_K_XL build flagged as crashing. This is excluded from automated testing.
- "None published" means no MTP head or drafter file has been released for that checkpoint in its Hugging Face repo, so speculative decoding cannot be tested for it. This applies to the Google E2B Q4_0 QAT and the Unsloth 12B QAT builds in some circumstances..

## Published reference benchmarks

These are published figures, not measured here. They are sensitive to thinking mode and evaluation harness, so the same model can appear with very different numbers across sources. Treat them as indicative, not definitive, and see per-row notes.

| Base model | MMLU Pro | GPQA Diamond | Notes |
|---|---|---|---|
| Qwen3.5-2B | 55.3% | 45.0% | Official Hugging Face evaluation widget (TIGER-Lab/MMLU-Pro, Idavidrein/gpqa). Likely thinking-enabled |
| Qwen3.5-4B | 79.1% | 76.2% | Official Hugging Face evaluation widget. Likely thinking-enabled |
| Qwen3.5-9B | 82.5% | 81.7% | Qwen reported headline figures (thinking-enabled / max) |
| Gemma 4 E2B | 60.0% | 43.4% | Google official, Gemma 4 E2B model card |
| Gemma 4 E4B | 69.4% | 58.6% | Google official, Gemma 4 E2B model card |
| Gemma 4 12B | not published | not published | Google did not include the 12B in its published benchmark table |
| Haiku 4.5 | 76% (80% thinking) | 62.6% (67.2% reasoning) | MMLU Pro from pricepertoken leaderboard; GPQA Diamond from Artificial Analysis (standard vs reasoning) |
| Gemini 3.1 Flash Lite | 86.2% (self-reported) | 82.2% | MMLU Pro self-reported on TIGER-Lab MMLU-Pro; GPQA Diamond from Artificial Analysis (independent). Google's own max-thinking GPQA Diamond figure is 86.9% |

Important reading note on Qwen3.5 thinking mode: the four small Qwen3.5 models ship with reasoning OFF by default. The published figures above are almost certainly thinking-enabled, so default-configuration local runs will score lower unless thinking is explicitly turned on. Where accuracy is being compared, run both modes.

## Capability and context reference

| Model family | Image | Video | Audio | Native context | Extended context | Thinking | Function calling | Licence |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5 (2B / 4B / 9B) | Yes | Yes | No | 262K | ~1M (YaRN) | Yes (off by default) | Yes | Apache 2.0 |
| Gemma 4 E2B / E4B | Yes | Yes | Yes | 128K | — | Yes | Yes | Apache 2.0 |
| Gemma 4 12B | Yes | Yes | Yes | 256K | — | Yes | Yes | Apache 2.0 |
| Haiku 4.5 (hosted) | Yes | No | No | 200K | — | Yes | Yes | Proprietary |
| Gemini 3.1 Flash Lite (hosted) | Yes | Yes | Yes | 1M | — | Yes | Yes | Proprietary |

Architecture notes: Qwen3.5 uses a hybrid Gated DeltaNet plus Gated Attention layout, which needs a recent llama.cpp and changes the KV cache and long-context memory profile relative to a pure-attention model. Gemma 4 is described as encoder-free / native multimodal, though an mmproj file is still loaded for vision in llama.cpp. Gemma 4 has no intrinsic MTP head; its speculative speedup comes from a separate drafter model used via the same draft path.

## Target hardware throughput reference (SpacemiT, A100 cores)

Reference figures on the target platform (CPU backend, 8 threads, n_ubatch 128, flash attention on), from SpacemiT's own benchmark data. Useful as a ballpark for what to expect before the full run.

| Model | Size | Params | Test | Tokens/sec |
|---|---|---|---|---|
| Qwen3.5 2B Q4_1 | 1.19 GiB | 1.88B | pp128 (prompt) | 115.23 ± 0.04 |
| Qwen3.5 2B Q4_1 | 1.19 GiB | 1.88B | tg128 (generation) | 16.49 ± 0.01 |
| Gemma 4 E4B Q4_K_M | 4.76 GiB | 7.52B | pp128 (prompt) | 21.13 ± 0.01 |
| Gemma 4 E4B Q4_K_M | 4.76 GiB | 7.52B | tg128 (generation) | 5.66 ± 0.00 |

Note the Gemma E4B "effective 4B" reports 7.52B parameters to llama.cpp, which is why its footprint and throughput sit well above the nominally similar Qwen3.5 4B.

## Disclaimers

- Accuracy figures here are published vendor or third-party numbers, collated for context. They were not produced by this test set and are not directly comparable to the local throughput results.
- Benchmark scores shift with thinking mode, evaluation harness, prompt format and sampling. Single-figure comparisons across vendors are rough at best. Gemini 3.1 Flash Lite's GPQA Diamond, for example, ranges from about 72% to 86.9% depending on thinking level and harness.
- Quantisation affects quality. The published scores reflect the full-precision model; a Q4 or Q2 GGUF will generally score lower, and no per-quant accuracy figures are published for these files.
- Gemma 4 12B MMLU Pro and GPQA Diamond are not published by Google. A GPQA Diamond figure near 78.8% has circulated in coverage but is unconfirmed and is deliberately not used here.
- Hosted reference models (Haiku 4.5, Gemini 3.1 Flash Lite) have undisclosed parameter counts and architectures and run server-side, so their throughput is not comparable to the local GGUF runs.
- llama.cpp support for the Qwen3.5 hybrid architecture and for MTP is still evolving. Results depend on the build; pin and record the llama.cpp commit used.

## Sources

- Qwen3.5 2B and 4B accuracy: official Hugging Face model card evaluation widgets (Qwen/Qwen3.5-2B, Qwen/Qwen3.5-4B).
- Qwen3.5 9B accuracy: Qwen reported figures, corroborated across coverage.
- Gemma 4 E2B and E4B accuracy and the full Gemma 4 benchmark appendix: Google Gemma 4 E2B model card on Hugging Face.
- Haiku 4.5: Artificial Analysis (GPQA Diamond, standard and reasoning), pricepertoken MMLU Pro leaderboard.
- Gemini 3.1 Flash Lite: Artificial Analysis (GPQA Diamond), TIGER-Lab MMLU-Pro leaderboard (self-reported), Google blog (max-thinking GPQA Diamond 86.9%).
- Capability, licence, MTP, mmproj and architecture details: Qwen3.5 and Gemma 4 Hugging Face model cards and Unsloth GGUF repositories and run guides.
- Target hardware throughput: SpacemiT official benchmark data.

## Appendix: Google's published Gemma 4 benchmark table

Reproduced from the Gemma 4 E2B model card for reference. Note the 12B is not included in Google's table.

| | Gemma 4 31B | Gemma 4 26B A4B | Gemma 4 E4B | Gemma 4 E2B | Gemma 3 27B (no think) |
|---|---|---|---|---|---|
| MMLU Pro | 85.2% | 82.6% | 69.4% | 60.0% | 67.6% |
| AIME 2026 no tools | 89.2% | 88.3% | 42.5% | 37.5% | 20.8% |
| LiveCodeBench v6 | 80.0% | 77.1% | 52.0% | 44.0% | 29.1% |
| Codeforces ELO | 2150 | 1718 | 940 | 633 | 110 |
| GPQA Diamond | 84.3% | 82.3% | 58.6% | 43.4% | 42.4% |
| Tau2 (average over 3) | 76.9% | 68.2% | 42.2% | 24.5% | 16.2% |
| HLE no tools | 19.5% | 8.7% | - | - | - |
| HLE with search | 26.5% | 17.2% | - | - | - |
| BigBench Extra Hard | 74.4% | 64.8% | 33.1% | 21.9% | 19.3% |
| MMMLU | 88.4% | 86.3% | 76.6% | 67.4% | 70.7% |
| MMMU Pro (vision) | 76.9% | 73.8% | 52.6% | 44.2% | 49.7% |
| OmniDocBench 1.5 (avg edit distance, lower is better) | 0.131 | 0.149 | 0.181 | 0.290 | 0.365 |
| MATH-Vision | 85.6% | 82.4% | 59.5% | 52.4% | 46.0% |
| MedXPertQA MM | 61.3% | 58.1% | 28.7% | 23.5% | - |
| CoVoST (audio) | - | - | 35.54 | 33.47 | - |
| FLEURS (audio, lower is better) | - | - | 0.08 | 0.09 | - |
| MRCR v2 8 needle 128k (avg) | 66.4% | 44.1% | 25.4% | 19.1% | 13.5% |

## References
- https://huggingface.co/google/gemma-4-E2B-it
- https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro 
- https://pricepertoken.com/leaderboards/benchmark/mmlu-pro 