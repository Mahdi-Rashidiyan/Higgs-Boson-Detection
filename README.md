---

# 🌌 Higgs Boson Detection

This repository contains a machine learning solution for the Higgs Boson Machine Learning Challenge. The objective is to classify particle collision events as either a "signal" (producing a Higgs boson) or "background" noise, based on data from particle physics experiments.

## 📊 Model Performance

The model was trained over several epochs, and its performance was monitored using training and validation loss and AUC (Area Under the ROC Curve) metrics. The training curves below illustrate the learning process and the model's ability to generalize to unseen data.

![Training Curves](https://z-cdn-media.chatglm.cn/files/20c767ed-6300-4a2d-803b-21cf103c435c.png?auth_key=1867276116-450c095a8470453086c8c24ac16a1b04-0-a315ebbcb01fd76d201328a2a26706ec)

### Key Observations

*   **Training & Validation Loss (Left)**: Both training and validation loss decrease sharply in the initial epochs, indicating that the model is learning effectively from the data. The validation loss begins to plateau after approximately 20 epochs, suggesting the model is converging. The small, consistent gap between the training and validation loss indicates a minor degree of overfitting, which is typical and suggests the model has good generalization performance.

*   **Training & Validation AUC (Right)**: The model's discriminative power, measured by AUC, improves rapidly for both sets. The validation AUC plateaus at a high value of approximately **0.89**, which signifies excellent performance in distinguishing between signal (Higgs boson) and background events. An AUC of 0.89 is considered a strong result for this classification task.

Overall, the curves suggest a well-trained model that has successfully learned the underlying patterns in the data without severe overfitting.

## 🛠️ Tech Stack

-   **Python 3**
-   **Pandas** for data manipulation.
-   **Scikit-learn** for data preprocessing and model evaluation.
-   **PyTorch** for building the classification model.
-   **Matplotlib** / **Seaborn** for visualization.

## 🚀 Getting Started

To run this project locally, follow these steps:

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Mahdi-Rashidiyan/Higgs-Boson-Detection.git
    cd Higgs-Boson-Detection
    ```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
