# Cross-Regime Generalization in Cryptocurrency Direction Prediction

Investigates how a deep learning model's predictive edge collapses when the market regime shifts, emphasizing the dangers of backtest overfitting and distribution shifts in production environments.

## Motivation
Financial machine learning models carry substantial risks. Models that demonstrate high accuracy in backtesting frequently collapse in production due to distribution shifts, creating a false sense of security that can mislead investors. Rather than treating this degradation as a tragic footnote, this project elevates it to the primary object of study. By explicitly measuring how a model's edge degrades across drastically different market regimes, this study highlights the imperative of honest evaluation and cross-regime stress-testing.

## Approach
This project reformulates cryptocurrency forecasting as a multi-horizon directional movement classification task (Up, Down, or Flat) for BTC and ETH. 
* **Architecture:** A hybrid deep learning architecture implemented in PyTorch. A 1D CNN extracts local temporal patterns from recent price windows, which are then fed into an LSTM to capture long-range temporal dependencies. 
* **Evaluation:** The model is trained in one market regime and evaluated in a drastically different regime. Performance is benchmarked against memory-free baselines, including logistic regression and naive momentum heuristics.

## Data Pipeline
* **Source:** High-frequency OHLCV data extracted via the public Binance API. 
* **Feature Engineering:** Enriched with standard technical indicators (RSI, MACD, Bollinger Bands). 
* **Integrity:** To strictly prevent look-ahead bias, all continuous features are normalized using z-scores calculated exclusively from the training set distribution. The dataset is partitioned chronologically into distinct, volatility-defined market phases to test cross-regime generalization.

## Repository Contents
* `proposal.pdf` — Project proposal
* `proposal.tex`, `refs.bib`, `APS360.sty` — LaTeX source files
* *(Code implementation files to be added)*

## Status
Proposal stage — implementation in progress. 

## Context
