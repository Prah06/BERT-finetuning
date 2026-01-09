# Sentiment Analysis: BERT vs Classical ML Comparative study of transformer models and traditional machine learning for movie review sentiment classification, demonstrating when simpler models outperform complex deep learning. ## Key Finding

**Classical ML baseline achieved equivalent performance to BERT (88.84% vs 89.00%) while being significantly faster and avoiding severe overfitting.** This project demonstrates that complex deep learning models aren't always superior - model selection should be driven by dataset characteristics and task requirements. ## Quick Results
| Model | Test Accuracy | Train Accuracy | Overfitting Gap | Training Time |
|-------|--------------|----------------|-----------------|---------------|
| **TF-IDF + LogReg** | **88.84%** | 88.90% | 0.06% | <1 min |
| **BERT-base** | 89.00% | 97.24% | 8.30% | 42m 31s|

*Best BERT performance at Epoch 2: 89.00% test accuracy*

**Key Insight:** BERT's 109M parameters caused severe overfitting with only 25k training examples (4,379 params/example ratio). ## Project Structure - **[RESULTS.md](RESULTS.md)** - Detailed experimental results, training curves, and analysis - **[notebooks/sentiment_analysis_enhanced.ipynb](notebooks/)** - Complete implementation with theory explanations - **[models/](models/)** - Saved model artifacts ## Tech Stack - **Deep Learning:** PyTorch, HuggingFace Transformers - **Classical ML:** scikit-learn, TF-IDF - **Training:** CUDA (NVIDIA GPU) - **Dataset:** IMDb k Movie Reviews ## What This Demonstrates ### Technical Skills -Transfer learning with pre-trained transformers -PyTorch training pipelines and GPU acceleration -Overfitting detection and analysis -Model comparison methodology -Classical machine learning fundamentals ### ML Engineering Judgment -When NOT to use deep learning -Cost-benefit analysis (performance vs complexity) -Production-readiness assessment -Data-driven model selection -Understanding dataset size requirements ## Key Insights ### 1. Severe Overfitting in BERT

**Training progression:**
- **Epoch 1:** Train 85.78%, Test 88.89% (Healthy)
- **Epoch 2:** Train 93.11%, Test 89.00% (Best performance)
- **Epoch 3:** Train 97.24%, Test 88.95% (Overfitting)

Train accuracy climbed 11.46% while test accuracy stayed essentially flat - classic overfitting pattern. ### 2. Dataset Size Requirements

With 25,000 training examples:
- **BERT:** 4,379 parameters per example (excessive → memorization)
- **Baseline:** 0.8 parameters per example (appropriate → generalization)

**Lesson:** Transformers need 100k+ examples to justify their 100M+ parameter complexity. ### . Simplicity Often Wins Despite being ,x larger, BERT provided: - Only .% accuracy gain over baseline - Significantly longer training time (+ minutes vs < minute) - Severe overfitting risk (.% train-test gap) - Complex deployment requirements **Conclusion:** Simple baseline was the optimal choice for this task. ### 4. Production Considerations

| Criterion | TF-IDF | BERT | Winner |
|-----------|--------|------|--------|
| Accuracy | 88.84% | 89.00% | Tie (~0.16% diff) |
| Training Speed | <1 min | 42m 31s | TF-IDF |
| Inference Speed | ~1ms | ~50ms | TF-IDF |
| Overfitting Risk | None | Severe | TF-IDF |
| Infrastructure | Simple | Complex (GPU) | TF-IDF |
| Interpretability | High | Low | TF-IDF | **Production Recommendation:** Deploy TF-IDF baseline for this use case. ## Detailed Documentation - **[RESULTS.md](RESULTS.md)** - Complete experimental results with epoch-by-epoch analysis - **[notebooks/sentiment_analysis_enhanced.ipynb](notebooks/)** - Implementation with theory explanations covering: - TF-IDF and how it works - BERT architecture and attention mechanisms - Transfer learning concepts - Overfitting detection and analysis - Model selection principles ## When to Use Each Approach ### Use Classical ML (TF-IDF + Logistic Regression) When: -Dataset < k examples -Binary or simple multi-class classification -Speed and simplicity are valued -Model interpretability is required -Limited compute resources ### Use Transformers (BERT) When: -Dataset > k examples -Complex NLP tasks -Context-dependent understanding is critical -Multilingual requirements -You have compute budget and time ## Skills Demonstrated This project showcases understanding of: - **Transformer architecture** - Attention mechanisms, BERT internals, layer-wise representations - **Overfitting causes** - Parameter-to-data ratio, memorization vs generalization - **Model selection** - Matching complexity to task and data availability - **Experimental methodology** - Baseline comparison, ablation analysis - **Production thinking** - Cost-benefit tradeoffs, deployment considerations - **Critical evaluation** - Questioning state-of-the-art when appropriate ## Reproducibility All experiments are fully reproducible using: - Fixed random seeds - Documented hyperparameters - Complete code in notebooks - Detailed results in RESULTS.md ### Environment ``` Python: .+ PyTorch: .+ Transformers: .+ scikit-learn: .+ CUDA: .x ``` See `requirements.txt` for complete dependencies. ## Future Work ### To Improve BERT Performance: . Increase dataset size to k+ examples . Implement early stopping (stop at Epoch ) . Try DistilBERT (M params, less prone to overfitting) . Add stronger regularization (higher dropout) . Implement data augmentation (paraphrasing, back-translation) ### Additional Experiments: - Multi-class sentiment (-star ratings) - Cross-domain evaluation (train on movies, test on products) - Attention visualization to interpret BERT's decisions - Model compression (quantization, pruning) - Compare against other architectures (RoBERTa, ELECTRA) ## 📫 Contact **Prahalad M** | Current Student @ Georgia Institute of Technology [LinkedIn](https://www.linkedin.com/in/prahalad-muralidharan-baa/) | pmuralitharan@gatech.edu **Built as part of portfolio demonstrating modern NLP techniques and ML engineering best practices.** *This project validates the principle: Intelligent solutions / models arent about how complex the solution is rather how fast , simple and effecient one is*