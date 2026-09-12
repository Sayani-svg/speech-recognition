# Speech Recognition: CNN-LSTM-Attention-CTC

An end-to-end Automatic Speech Recognition (ASR) pipeline built from scratch in TensorFlow/Keras, trained and evaluated on the LJSpeech dataset.

## Overview

Raw audio is converted into Log-Mel Spectrograms, passed through a CNN for local acoustic feature extraction, then a **unidirectional LSTM** (chosen specifically for real-time streaming compatibility, unlike bidirectional variants), refined with an **Attention** mechanism, and decoded using **CTC (Connectionist Temporal Classification)** — avoiding the need for frame-level alignment labels.

```
Audio → Log-Mel Spectrogram → CNN → LSTM → Attention → CTC Decoding → Transcript
```

## Results

| Metric | Value |
|---|---|
| Word Error Rate (WER) | **34.75%** |
| Dataset | LJSpeech |
| Framework | TensorFlow / Keras |

## Why unidirectional LSTM?

Most ASR architectures use bidirectional LSTMs for higher accuracy, but they require the full audio sequence before producing output — unusable for streaming. This pipeline uses a unidirectional LSTM to keep the architecture compatible with real-time/streaming inference, at a deliberate accuracy trade-off.

## Tech Stack

- **Language/Framework:** Python, TensorFlow, Keras
- **Environment:** Google Colab
- **Dataset:** [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)

## Status

Prototype complete with a full training run. Write-up in progress for formal (IEEE-style) publication.

## Author

Sayani Chatterjee — [GitHub](https://github.com/sayanichatterjee) · [LinkedIn](https://linkedin.com/in/sayani--chatterjee)
