
### Model Evolution & Performance Comparison

| Model Version | Kaggle Accuracy | Sunset False Positive Rate | Status |
| :--- | :--- | :--- | :--- |
| **Baseline** (Pre-Adaptation) | 48.4% | N/A | ❌ FAILED |
| **Domain Adapted** (Round 1) | 95.8% | 8.42% | ⚠️ Bias Detected |
| **Hard Negative Mined** (Round 2) | ~96.0% | 3.16% | ⚠️ Improved |
| **Final Expanded** (Round 3) | **97.11%** | **5.79%** | ✅ **PRODUCTION READY** |
