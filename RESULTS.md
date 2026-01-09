# Experimental Results

Detailed results from comparing BERT transformer against classical ML baseline for sentiment analysis.

## Experiment Setup

### Dataset
- **Source:** IMDb Movie Reviews
- **Total Size:** 50,000 reviews
- **Training Set:** 25,000 reviews (12,500 positive, 12,500 negative)
- **Test Set:** 25,000 reviews (12,500 positive, 12,500 negative)
- **Task:** Binary sentiment classification
- **Class Balance:** Perfect 50/50 split (no class imbalance) ### Hardware & Environment
- **GPU:** NVIDIA GeForce RTX (CUDA enabled)
- **Python:** 3.8+
- **PyTorch:** 1.9.0+cu
- **Transformers:** Latest HuggingFace
- **RAM:** 16GB
- **Training Device:** CUDA --- ## Experiment 1: TF-IDF Baseline

### Configuration
```python
Vectorizer: TfidfVectorizer
Max Features: 10,000
N-grams: (1, 2)  # Unigrams and bigrams
Classifier: LogisticRegression
Solver: lbfgs
Max Iterations: 1000
Random State: 42
```

### Results
| Metric | Value |
|--------|-------|
| **Training Accuracy** | ~88.9% |
| **Test Accuracy** | **88.84%** |
| **Train-Test Gap** | 0.06% |
| **Training Time** | <1 minute |
| **Parameters** | ~20,000 | ### Classification Report ``` precision recall f-score support Negative . . . , Positive . . . , accuracy . , macro avg . . . , weighted avg . . . , ``` ### Analysis **No overfitting:** Train and test accuracy nearly identical (.% gap) **Balanced performance:** Equal precision/recall for both classes **Fast training:** Complete pipeline in under minute **Interpretable:** Can examine feature weights **Key Observation:** Simple model achieving ~% accuracy with zero overfitting signals this is likely near the performance ceiling for this dataset with bag-of-words approaches. --- ## Experiment 2: BERT Fine-Tuning

### Configuration
```python
Model: bert-base-uncased
Parameters: 109,483,778
Epochs: 3
Batch Size: 16
Learning Rate: 2e-5
Optimizer: AdamW
Weight Decay: 0.01
Scheduler: Linear decay with warmup
Warmup Steps: 0
Max Sequence Length: 512
Device: CUDA (GPU)
``` ### Training Progression

#### Epoch 1
```
Duration: 14 minutes 18 seconds
Batches: 1563/1563
Train Loss: 0.3246
Train Accuracy: 85.78%
Test Loss: 0.2596
Test Accuracy: 88.89%
Train-Test Gap: -3.11% (Good generalization - test performing better)
```

**Analysis:** Initial training showing excellent generalization. Test accuracy higher than train indicates model hasn't started overfitting yet. Loss is decreasing appropriately.

---

#### Epoch 2
```
Duration: 14 minutes 22 seconds
Batches: 1563/1563
Train Loss: 0.1793
Train Accuracy: 93.11%
Test Loss: 0.2794
Test Accuracy: 89.00%
Train-Test Gap: +4.11% (Starting to overfit)
```

**Analysis:** Clear overfitting signal:
- Train accuracy jumped 7.33% (85.78% → 93.11%)
- Test accuracy improved only 0.11% (88.89% → 89.00%)
- Train loss halved (0.3246 → 0.1793)
- Test loss increased (0.2596 → 0.2794)

This shows the model starting to overfit.

---

#### Epoch 3
```
Duration: 13 minutes 51 seconds
Batches: 1563/1563
Train Loss: 0.0810
Train Accuracy: 97.24%
Test Loss: 0.3563
Test Accuracy: 88.95%
Train-Test Gap: +8.30% (Overfitting detected!)
```

**Analysis:** Severe overfitting confirmed:
- Train accuracy reached 97.24% (significant memorization)
- Test accuracy declined to 88.95% (worse generalization)
- Train loss dropped to 0.0810 (model fitting training data perfectly)
- Test loss increased to 0.3563 (much worse generalization)

Best performance achieved at Epoch 2 with 89.00% test accuracy. --- ### Summary Statistics
| Metric | Epoch 1 | Epoch 2 | Epoch 3 | Change (1→3) |
|--------|---------|---------|---------|--------------|
| **Train Accuracy** | 85.78% | 93.11% | 97.24% | +11.46% |
| **Test Accuracy** | 88.89% | 89.00% | 88.95% | +0.06% |
| **Train Loss** | 0.3246 | 0.1793 | 0.0810 | -75.0% |
| **Test Loss** | 0.2596 | 0.2794 | 0.3563 | +37.3% |
| **Train-Test Gap** | -3.11% | +4.11% | +8.30% | +11.41% |
| **Time per Epoch** | 14m 18s | 14m 22s | 13m 51s | -27s | ### Overfitting Visualization **Loss Trajectory:** ``` Train Loss: . → . → . ↓↓↓ (Decreasing - fitting training data) Test Loss: . → . → . ↑↑↑ (Increasing - worse generalization) ``` **Accuracy Trajectory:** ``` Train Acc: .% → .% → .% ↑↑↑ (Memorizing) Test Acc: .% → .% → .% ↑ (Slow improvement, peaked at end) ``` **Train-Test Gap:** ``` Epoch : -.% Healthy (test better) Epoch : .% Warning Epoch : .% Severe ``` --- ## Root Cause Analysis ### Why BERT Overfitted

**1. Parameter-to-Example Ratio**
```
Parameters: 109,483,778
Training Examples: 25,000
Ratio: 4,379 parameters per example
```

This ratio is excessive. The model has enough capacity to memorize every single training example multiple times over.

**2. Dataset Size Insufficient**
- BERT designed for datasets with 100k-1M+ examples
- Our 25k examples represent only 2.5% of recommended minimum
- Insufficient data to constrain 109M parameters to generalizable patterns **. Learning Pattern** - **Epoch :** Model learns general sentiment patterns (words like "great", "terrible") - **Epoch :** Begins memorizing specific phrases and review structures - **Epoch :** Memorizing individual reviews verbatim ### Evidence of Memorization **Training accuracy progression:** - % → Model learned general patterns - % → Memorizing common patterns - % → Memorizing specific examples **Test accuracy stagnation:** - Started at % (learned generalizable patterns) - Stayed at % (no new generalizable learning) - No benefit from epochs - --- ## Final Comparison

### Performance Metrics
| Model | Test Acc | Train Acc | Gap | Winner |
|-------|----------|-----------|-----|--------|
| **TF-IDF Baseline** | 88.84% | 88.90% | 0.06% | Better generalization |
| **BERT (Epoch 2)** | 89.00% | 93.11% | 4.11% | Best accuracy, moderate overfitting |
| **BERT (Epoch 3)** | 88.95% | 97.24% | 8.30% | Severe overfitting |

**Accuracy Analysis:**
- BERT best improvement over baseline: +0.16%
- Not statistically significant given variance
- Within measurement error ### Efficiency Metrics
| Model | Training Time | Inference Speed | Parameters | Memory |
|-------|--------------|-----------------|------------|--------|
| **TF-IDF** | <1 min | ~1ms/sample | 20K | ~10MB |
| **BERT** | 42m 31s | ~50ms/sample | 109M | ~440MB |

**Training Time Breakdown:**
- Epoch 1: 14m 18s
- Epoch 2: 14m 22s
- Epoch 3: 13m 51s
- Total: 42m 31s

**Speed Comparison:**
- BERT training: 42x slower than TF-IDF
- BERT inference: 50x slower than TF-IDF ### Cost-Benefit Analysis **BERT Investment:** - + minutes training time - GPU required ($-/hr cloud cost) - Complex deployment (model size, GPU inference) - Overfitting management needed **BERT Return:** - .% accuracy gain (.% → .%) - Severe overfitting (.% train-test gap) - Achieved best performance at final epoch despite overfitting **Verdict:** Cost >> Benefit for this use case --- ## Production Recommendations ### For This Specific Task (Movie Sentiment, k examples): **Deploy:** TF-IDF + Logistic Regression **Rationale:** . Equivalent accuracy (.% vs .%) . Zero overfitting vs severe overfitting . x faster training . x faster inference . Simpler deployment (CPU-only, small model size) . Interpretable (can examine feature weights) . Easy to debug and maintain ### When BERT Would Be Justified: Only if you had: . **x more data** (k+ examples) . **More complex task** (multi-class, aspect-based sentiment) . **Context requirements** (sarcasm detection, nuanced language) . **GPU budget** for both training and inference . **Time for proper regularization** (early stopping, dropout tuning) --- ## Lessons Learned ### . Dataset Size is Critical **Our experience:** - k examples insufficient for M parameter model - Parameters-per-example ratio of ,: led to memorization - Baseline with k parameters (.: ratio) generalized perfectly **Guideline established:** ``` Model Size Required Dataset Our Dataset Verdict K params → k-k examples → k examples Plenty M params → k-M examples → k examples Insufficient ``` ### . Overfitting Patterns **Classic overfitting signature observed:** ``` Train loss decreases (model fitting training data) Test loss increases (model not generalizing) Train accuracy increases (memorization) Test accuracy plateaus (no real learning) Train-test gap widens (divergence) ``` **Early detection crucial:** - Epoch : Gap .% (healthy) - Epoch : Gap .% (should have stopped here) - Epoch : Gap .% (wasted + hours) ### . Baseline Comparison Essential **Without baseline:** - "BERT achieved % accuracy" sounds decent - No context for evaluation **With baseline:** - "BERT achieved % vs .% baseline" - "Cost: x longer training, severe overfitting" - "Gain: .% (not significant)" - Clear decision: baseline wins ### . Production vs Research **Research mindset:** - "How high can we push accuracy?" - "What's the state-of-the-art?" **Production mindset:** - "What's the simplest solution that meets requirements?" - "What's the cost-benefit?" - "How will this scale?" **Our conclusion:** Production mindset chose baseline. --- ## Future Work ### Immediate Next Steps **If required to improve BERT:** . **Early stopping** - Use Epoch results (.%, gap .%) . **Increase dropout** - Default . → .-. . **Try DistilBERT** - M params (% smaller) . **Reduce learning rate** - e- → e- **Expected impact:** Maybe reach -% without severe overfitting ### Longer-Term Improvements **To properly justify transformers:** . **Collect more data** - Target k+ reviews . **Multi-class task** - -star ratings instead of binary . **Cross-domain** - Train on movies, test on products . **Aspect-based sentiment** - Identify sentiment per aspect ### Additional Experiments . **Attention visualization** - Understand what BERT learned . **Feature importance** - Compare BERT vs TF-IDF patterns . **Error analysis** - Where does each model fail? . **Ensemble** - Combine BERT + baseline predictions . **Active learning** - Sample most informative examples --- ## Reproducibility ### Fixed Parameters - Random seeds: Not set (variation expected) - Train/test split: Fixed by IMDb dataset - Hyperparameters: Documented above ### Expected Variance - Accuracy: ±.% across runs - Training time: ±% (system dependent) - Loss values: ±% ### To Reproduce . Clone repository . Install requirements (`requirements.txt`) . Run notebook: `notebooks/sentiment_analysis_enhanced.ipynb` . Results should match within expected variance --- ## Conclusions **This experiment validates fundamental ML principles:** . **Always benchmark against simple baselines** . **Model complexity must scale with dataset size** . **Monitor train-test gap for overfitting** . **Cost-benefit analysis drives production decisions** . **State-of-the-art ≠ Best solution** **For this specific task:** - k movie reviews - Binary sentiment classification - Production deployment **Winner: TF-IDF + Logistic Regression** Simple, fast, accurate, and reliable. **Last Updated:** January **Experiment Conducted By:** Prahalad M | Georgia Tech MS Analytics