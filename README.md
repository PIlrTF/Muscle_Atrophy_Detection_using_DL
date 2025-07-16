Project Title:  Muscle Atrophy Detection Using Deep Learning

Objective:

To develop an intelligent, interpretable deep learning model capable of detecting muscle atrophy early by analyzing physiological, demographic, and lifestyle parameters.

Key Contributions:

* Problem: Muscle atrophy is hard to detect early using traditional methods like MRI/CT and lacks standardized diagnostic frameworks.

* Solution: A Feed Forward Neural Network (FFNN)** model was trained on a custom synthetic dataset incorporating EMG signals, muscle mass, density, age, gender, physical activity, and other variables.

Dataset:

  * Created due to limitations in public datasets (e.g., Kaggle).
  * Included 13+ features like age, gender, muscle area, texture, fiber integrity, smoking, alcohol, and past muscle measurements.
  * Labeled as: 0 = No Atrophy, 1 = Atrophy.

Methodology:

  1. Data Cleaning: Outlier removal, normalization, encoding.
  2. Visualization: Heatmaps, violin plots, UMAP for dimensionality reduction.
  3. Statistical Validation: KS test to ensure train-test distribution consistency.
  4. Model Architecture: A Feed Forward Neural Network (FFNN) with two hidden layers was designed using ReLU activations and a sigmoid output layer for binary classification. Hyperparameters like layer size and learning rate were optimized using Keras Tuner (Bayesian Optimization).

Training & Evaluation:
The model was trained with SMOTE to address class imbalance, and performance was evaluated using metrics like accuracy, F1-score, ROC-AUC, and confusion matrix. The model achieved high validation accuracy and generalization was confirmed through 5-fold stratified cross-validation.

Interpretability :
Violin plots, correlation heatmaps, and UMAP projections provided deep insights into feature behavior and class separation. Statistical tests like the Kolmogorov-Smirnov test validated the consistency of feature distributions across splits.

Future Scope:
The system demonstrates the potential for clinical integration by offering a scalable, interpretable, and cost-effective solution for early muscle atrophy detection. Future directions include real-time monitoring via wearable sensors, multimodal data fusion, and deployment in telemedicine platforms to support personalized preventive care.


