# Sentiment Analysis: BERT vs Classical ML

Comparative study of transformer models and traditional machine learning for movie review sentiment classification, demonstrating when simpler models outperform complex deep learning.

##  Key Finding

**Classical ML baseline achieved equivalent performance to BERT (88.84% vs 89.16%) while being significantly faster and avoiding severe overfitting.**

This project demonstrates that complex deep learning models aren't always superior - model selection should be driven by dataset characteristics and task requirements.

## Quick Results

| Model | Test Accuracy | Train Accuracy | Overfitting Gap | Training Time |
|-------|--------------|----------------|-----------------|---------------|
| **TF-IDF + LogReg** | **88.84%** | 88.9% | 0.06%  | <1 min |
| **BERT-base** | 89.16% | 97.13% | 7.96% | 40m 55s|

*Training time includes Epoch 3 which took significantly longer due to system performance

**Key Insight:** BERT's 109M parameters caused severe overfitting with only 25k training examples (4,379 params/example ratio).

## Project Structure

- **[RESULTS.md](RESULTS.md)** - Detailed experimental results, training curves, and analysis
- **[notebooks/sentiment_analysis_enhanced.ipynb](notebooks/)** - Complete implementation with theory explanations
- **[models/](models/)** - Saved model artifacts

## Tech Stack

- **Deep Learning:** PyTorch, HuggingFace Transformers
- **Classical ML:** scikit-learn, TF-IDF
- **Training:** CUDA (NVIDIA GPU)
- **Dataset:** IMDb 50k Movie Reviews

## What This Demonstrates

### Technical Skills
-Transfer learning with pre-trained transformers  
-PyTorch training pipelines and GPU acceleration  
-Overfitting detection and analysis  
-Model comparison methodology  
-Classical machine learning fundamentals  

### ML Engineering Judgment
-When NOT to use deep learning  
-Cost-benefit analysis (performance vs complexity)  
-Production-readiness assessment  
-Data-driven model selection  
-Understanding dataset size requirements  

## Key Insights

### 1. Severe Overfitting in BERT

**Training progression:**
- **Epoch 1:** Train 85.52%, Test 88.56% (Healthy)
- **Epoch 2:** Train 92.87%, Test 88.69% (Slight decline)
- **Epoch 3:** Train 97.13%, Test 89.16% (Best performance)

Train accuracy climbed 11.6% while test accuracy stayed essentially flat - classic overfitting pattern.

### 2. Dataset Size Requirements

With 25,000 training examples:
- **BERT:** 4,379 parameters per example (excessive → memorization)
- **Baseline:** 0.4 parameters per example (appropriate → generalization)

**Lesson:** Transformers need 100k+ examples to justify their 100M+ parameter complexity.

### 3. Simplicity Often Wins

Despite being 10,000x larger, BERT provided:
- Only 0.32% accuracy gain over baseline
- Significantly longer training time (40+ minutes vs <1 minute)
- Severe overfitting risk (7.96% train-test gap)
- Complex deployment requirements

**Conclusion:** Simple baseline was the optimal choice for this task.

### 4. Production Considerations

| Criterion | TF-IDF | BERT | Winner |
|-----------|--------|------|--------|
| Accuracy | 88.84% | 89.16% | Tie (~0.32% diff) |
| Training Speed | <1 min | 40m 55s | TF-IDF |
| Inference Speed | ~1ms | ~50ms | TF-IDF |
| Overfitting Risk | None | Severe | TF-IDF |
| Infrastructure | Simple | Complex (GPU) | TF-IDF |
| Interpretability | High | Low | TF-IDF |

**Production Recommendation:** Deploy TF-IDF baseline for this use case.

## Detailed Documentation

- **[RESULTS.md](RESULTS.md)** - Complete experimental results with epoch-by-epoch analysis
- **[notebooks/sentiment_analysis_enhanced.ipynb](notebooks/)** - Implementation with theory explanations covering:
  - TF-IDF and how it works
  - BERT architecture and attention mechanisms
  - Transfer learning concepts
  - Overfitting detection and analysis
  - Model selection principles

## When to Use Each Approach

### Use Classical ML (TF-IDF + Logistic Regression) When:
-Dataset < 100k examples  
-Binary or simple multi-class classification  
-Speed and simplicity are valued  
-Model interpretability is required  
-Limited compute resources  

### Use Transformers (BERT) When:
-Dataset > 100k examples  
-Complex NLP tasks (NER, QA, summarization)  
-Context-dependent understanding is critical  
-Multilingual requirements  
-You have compute budget and time  

## Skills Demonstrated

This project showcases understanding of:
- **Transformer architecture** - Attention mechanisms, BERT internals, layer-wise representations
- **Overfitting causes** - Parameter-to-data ratio, memorization vs generalization
- **Model selection** - Matching complexity to task and data availability
- **Experimental methodology** - Baseline comparison, ablation analysis
- **Production thinking** - Cost-benefit tradeoffs, deployment considerations
- **Critical evaluation** - Questioning state-of-the-art when appropriate

##  Reproducibility

All experiments are fully reproducible using:
- Fixed random seeds
- Documented hyperparameters
- Complete code in notebooks
- Detailed results in RESULTS.md

### Environment
```
Python: 3.10+
PyTorch: 2.0+
Transformers: 4.30+
scikit-learn: 1.3+
CUDA: 12.x
```

See `requirements.txt` for complete dependencies.

## Future Work

### To Improve BERT Performance:
1. Increase dataset size to 100k+ examples
2. Implement early stopping (stop at Epoch 1)
3. Try DistilBERT (66M params, less prone to overfitting)
4. Add stronger regularization (higher dropout)
5. Implement data augmentation (paraphrasing, back-translation)

### Additional Experiments:
- Multi-class sentiment (5-star ratings)
- Cross-domain evaluation (train on movies, test on products)
- Attention visualization to interpret BERT's decisions
- Model compression (quantization, pruning)
- Compare against other architectures (RoBERTa, ELECTRA)

## 📫 Contact

**Prahalad M** | Current Student @ Georgia Imstitute of Technology  
[LinkedIn]() | prahalad@gatech.edu

**Built as part of portfolio demonstrating modern NLP techniques and ML engineering best practices.**

*This project validates the principle: Intelligent solutions / models arent about how complex the solution is rather how fast , simple and effecient one is*
