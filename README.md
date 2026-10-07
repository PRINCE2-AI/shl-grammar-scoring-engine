# Grammar Scoring Engine for Spoken English

Solution for the **SHL Hiring Assessment 2026** Kaggle challenge: predict a 1–5 grammar score (MOS Likert rubric) for a 45–60 second spoken-English recording.

**Public leaderboard: 0.3262 (rank 5 at submission time)** · Kaggle user: `primce1`

> The competition data is **not** included (the competition rules do not allow sharing it). Attach the competition dataset in Kaggle to run the notebook.

## Approach

Each clip is described by several complementary views. Every view gets a small regularised model (Ridge / RBF-SVR, or a fine-tuned regressor) trained with **speaker-grouped cross-validation**, and the out-of-fold predictions are combined with a **non-negative least-squares (NNLS) stack**.

```
45–60 s WAV (16 kHz)
 ├─ what was said
 │    ├─ Whisper large-v3-turbo: normal transcript + "verbatim" transcript (keeps the speaker's own errors)
 │    ├─ RoBERTa-large fine-tuned as a regressor on the transcripts
 │    ├─ CoEdIT-large grammar correction → edit rate per 100 words
 │    └─ Qwen2.5-7B: rubric grade probabilities, token surprisal, hidden states
 ├─ how it sounded
 │    ├─ fluency statistics (pauses, speech rate, loudness)
 │    ├─ Whisper encoder, WavLM-base/large, HuBERT-large: per-layer mean-pooled states
 │    └─ audio language models: Voxtral-Mini-3B and Qwen2-Audio-7B hidden states over the audio tokens
 └─ Ridge / SVR per view and per layer  →  NNLS stack (separate stacker for the 45 s recording batch)
                                         →  ×1.15 range expansion → clip [2, 5] → zero gate for noise-masked clips
```

Key decisions:

| Decision | Why |
|---|---|
| Speaker-grouped CV (WavLM speaker clusters + `StratifiedGroupKFold`) | The same speakers appear in several training clips; random folds leak identity and overstate accuracy. |
| Verbatim transcript | Whisper's language model silently "fixes" learner grammar; a disfluent prompt makes it keep the errors that the score is about. |
| Audio-LLM hidden states | Voxtral-Mini-3B audio-token states are the strongest single view (OOF RMSE 0.540). |
| Separate stacker for the 45.06 s batch | That recording batch is ~14 % of train but ~47 % of test and has lower grades. |
| Zero gate | The 37 training clips graded 0 are noise-masked recordings; a frame-energy rule separates them perfectly and flags one test clip. |

## Results

| | RMSE | Pearson r |
|---|---|---|
| Training data (models applied to the clips they were fit on) | 0.297 | 0.956 |
| Speaker-grouped 5-fold CV, out-of-fold (732 scored clips) | 0.495 | 0.873 |
| Kaggle public leaderboard | **0.3262** | – |

Leaderboard history (each step kept only if it helped):

| Change | Public LB |
|---|---|
| Whisper + CoEdIT + WavLM baseline | 0.4617 |
| + layer embeddings, speaker-grouped CV | 0.4031 |
| + RoBERTa-large on verbatim transcripts | 0.3678 |
| + Qwen2-Audio / HuBERT / Qwen2.5 text views | 0.3568 |
| + zero gate, clip to [2, 5] | 0.3540 |
| + Voxtral-Mini-3B audio-token views | 0.3390 |
| + separate stacker for the 45 s batch | **0.3262** |

Not kept: a Qwen2.5-3B LoRA regressor (better out-of-fold score but worse public score) and a constant offset for the 45 s batch. The public leaderboard has only ~130 clips, so differences below ~0.01 are within noise; the average of the global and per-batch stackers (public 0.3297) is the safer second choice.

## How to run

1. Import `shl-grammar-scoring.ipynb` into Kaggle and attach the competition dataset.
2. Settings: **Accelerator = GPU T4 ×2**, **Internet = On** (open-source models are downloaded from Hugging Face).
3. Run all. From scratch the run takes about 4 hours; every expensive step caches its result to `/kaggle/working`, so re-runs only recompute what changed.
4. Output files: `submission.csv` (global stack) and the variants `sub_noql_batch_zc.csv` (final submission), `sub_noql_avgbatch_zc.csv`, `sub_noql_zc.csv`, `sub_full_zc.csv`, `sub_frozen_zc.csv`.

## Models used (all open-source, no paid APIs)

`openai/whisper-large-v3-turbo` · `grammarly/coedit-large` · `roberta-large` · `Qwen/Qwen2.5-7B-Instruct` · `Qwen/Qwen2.5-3B` · `sentence-transformers/all-mpnet-base-v2` · `microsoft/wavlm-base-plus` · `microsoft/wavlm-large` · `facebook/hubert-large-ll60k` · `Qwen/Qwen2-Audio-7B-Instruct` · `mistralai/Voxtral-Mini-3B-2507` · scikit-learn

## Author

Prince Maurya · [GitHub](https://github.com/PRINCE2-AI)

AI coding assistants were used during development (allowed by SHL); all design choices are explained above and in the notebook.
