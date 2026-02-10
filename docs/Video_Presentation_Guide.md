
# 🎥 Video Presentation Script (15 Minutes)

**Author**: Prattay Roy Chowdhury
**Topic**: Enhanced UAV Fire Detection System using MobileNetV2-CBAM

---

## **0:00 - 2:00 | Introduction & Problem Statement**
*(Visual: Title Slide with Project Name and your Name)*

"Hello everyone, my name is Prattay Roy Chowdhury. Today, I am presenting my solution for the Developer Round 1 task: **'Implementation and Enhancement of MobileNetV2-CBAM for UAV-based Fire Detection'**."

*(Visual: Slide showing Forest Fire devastation stats or images)*

"Forest fires are a growing global crisis. Traditional detection methods like satellites are often too slow, and human patrols are dangerous. Unmanned Aerial Vehicles, or UAVs, offer a rapid solution, but they require Artificial Intelligence that is both **fast** and **accurate**."

*(Visual: Slide with 'The Challenge: False Positives')*

"The core problem we face is **reliability**. Lightweight models deployable on drones often struggle to distinguish between actual **fire** and **smoke** or **fog**, leading to high false alarm rates. My objective for this project was to build a system that solves this specific limitation."

---

## **2:00 - 4:00 | Literature Review & Research Gap**
*(Visual: Slide showing the Reference Paper Title)*

"I selected the research paper titled **'A lightweight CNN model for UAV-based image classification'** published in *Soft Computing (2025)*. This paper proposed using MobileNetV2 for its efficiency."

*(Visual: Slide highlighting the 'Research Gap' text)*

"However, upon critical review, I identified a significant **Research Gap**. On Page 14, the authors explicitly admit: *'Most error-prone fire images contain strong smoke... it is necessary to further improve the model for complex scenes.'*"

"This became the foundation of my work. My Research Questions were:
1. Can we teach the model to ignore smoke and focus only on the flame?
2. Can we bridge the domain gap between clean training data and real-world smoky fires?"

---

## **4:00 - 7:00 | Methodology & Proposed Algorithm**
*(Visual: Architecture Diagram showing MobileNetV2 + CBAM)*

"To answer these questions, I proposed a unique architecture: **MobileNetV2-CBAM**. Let me explain how it works."

**1. Data Preprocessing**
"I utilized the FLAME dataset for aerial imagery. I applied:
*   **Resizing** to 224x224 for consistency.
*   **Normalization** using ImageNet standards.
*   **Augmentation** like rotation and flipping to improve robustness."

**2. The Model (MobileNetV2 + CBAM)**
"I chose **MobileNetV2** as the backbone because of its 'Inverted Residual Blocks', which are perfect for edge devices like drones. However, I enhanced it with **CBAM**—the Convolutional Block Attention Module."

*(Visual: Point to CBAM diagram)*

"CBAM works in two steps:
*   First, **Channel Attention**: It asks 'WHAT is this feature?' helping the model prioritize 'fire texture' over 'leaf texture'.
*   Second, **Spatial Attention**: It asks 'WHERE is the fire?', effectively creating a spotlight on the flame and suppressing the surrounding smoke."

---

## **7:00 - 10:00 | Technical Implementation (Code Walkthrough)**
*(Visual: Switch to Screen Share of VS Code or PyCharm)*

"Now, let's look at the implementation. I built this end-to-end system using **PyTorch** for the model, **FastAPI** for the backend, and **Next.js** for the dashboard."

*(Visual: Scroll through `ml_core/model.py`)*

"Here in `model.py`, you can see the `MobileNetV2_CBAM` class. I insert the attention module right after the feature extraction capability, ensuring we refine the features *before* classification."

*(Visual: Show the Dashboard running in Browser)*

"I also built a real-time dashboard. Let me demonstrate.
I will upload an image with heavy smoke."

*(Action: Upload a test image)*

"As you can see, the model predicts 'Fire' with **high confidence**.
More importantly, look at this **Grad-CAM Heatmap**.
The red 'hot' spots are strictly on the fire itself. The model is completely ignoring the smoke clouds above. This visual proof confirms that our Attention mechanism is working correctly."

---

## **10:00 - 13:00 | Results & Comparative Analysis**
*(Visual: Slide showing the Comparative Table)*

"To rigorously test my solution, I performed a **Comparative Analysis** using a secondary dataset from Kaggle, representing a 'Domain Shift'."

"**The Baseline:**
When I ran the standard MobileNetV2, it failed catastrophically on the smoky dataset, achieving only **48.4% accuracy**. It was guessing."

"**The Proposed Solution:**
After implementing my Domain Adaptation strategy—where I merged diverse, smoky samples into the training pipeline—the accuracy jumped to **95.8%**.
Comparison:
*   **Baseline**: 48.4%
*   **Ours**: 95.8%"

"This +47% improvement validates that my approach successfully bridged the generalization gap found in the literature."

---

## **13:00 - 15:00 | Conclusion & Recommendations**
*(Visual: Conclusion Slide)*

"In conclusion, I have successfully:
1.  **Reproduced** a state-of-the-art lightweight model.
2.  **Identified** a critical flaw regarding smoke.
3.  **Solved** it using Attention Mechanisms and Domain Adaptation."

"**Recommendations:**
For future work, I recommend porting this model to an **NVIDIA Jetson Nano** using TensorRT for fully autonomous drone deployment. I also suggest expanding the dataset to include night-time fire imagery."

"Thank you for your time. I am now open to any questions."

---
**Suggested Journals for Publication**
*(Keep this slide visible at the very end or mention briefly)*
1. *International Journal of Machine Learning and Cybernetics (Springer)* - Q2
2. *Journal of the Optical Society of America A* - Q2
3. *International Journal of Computer Networks and Applications* - Q3
