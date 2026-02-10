
# Video Presentation Guide (15 Minutes)

**Topic**: Enhanced UAV Fire Detection System using MobileNetV2-CBAM
**Author**: Prattay Roy Chowdhury

---

## 1. Introduction (2 Minutes)
*   **Hook**: "Forest fires are causing increasing devastation globally. Traditional detection (satellites) is too slow; we need real-time solutions."
*   **Problem**: Existing lightweight models on drones struggle with "false positives" from smoke and fog (as highlighted in the reference paper).
*   **Objective**: To build a deployable, edge-friendly AI system that detects fire *accurately* even in complex, smoky environments.

## 2. Literature Review & Gaps (2 Minutes)
*   **Reference Paper**: Mention "A lightweight CNN model..." (Deng et al., 2025).
*   **The Gap**: Quote page 14 – "Model struggles with heavy smoke."
*   **Our Hypothesis**: Adding an Attention Mechanism (CBAM) + Domain Adaptation (diverse training data) will solve this.

## 3. Methodology & Architecture (4 Minutes)
*   **Base Model**: MobileNetV2. Explain why: "Inverted Residuals" make it fast and lightweight for drones.
*   **The Innovation (CBAM)**:
    *   **Channel Attention**: "Teaches the model *what* is fire (red/yellow heat) vs. what is just background."
    *   **Spatial Attention**: "Teaches the model *where* the fire is, ignoring the smoke around it."
*   **Show Algorithm Steps**: Briefly show the logic (Input -> Features -> Attention -> Classify).

## 4. Technical Implementation (3 Minutes)
*   **Stack**: PyTorch (ML), FastAPI (Backend), Next.js (Dashboard).
*   **Demo**:
    *   Show the **Web Dashboard** running.
    *   **Grad-CAM**: Upload a smoky fire image. Show how the heatmap lights up *only* on the fire, not the smoke. This proves validity.

## 5. Results & Analysis (3 Minutes)
*   **The "Aha!" Moment**:
    *   Show the **Before** result: "On the Kaggle dataset, the original training approach failed (48% accuracy)."
    *   Show the **After** result: "After Domain Adaptation, accuracy jumped to 95.8%."
*   **Comparison**: Show a table comparing your model vs. standard MobileNetV2 or VGG16.

## 6. Conclusion (1 Minute)
*   **Summary**: We successfully built a robust, real-time fire detector.
*   **Impact**: Ready for deployment on NVIDIA Jetson Nano.
*   **Closing**: "This project bridges the gap between theoretical lightweight models and real-world robustness."

---
**Tips for Recording:**
*   Use OBS Studio or Zoom to record your screen.
*   Keep the **Web Dashboard** open and ready to run a live prediction.
*   Speak clearly and confidently!
