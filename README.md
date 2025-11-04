<p align="center">
  <img src="https://github.com/Zahra58/data-mining-project--wholesale-customer-clustering/blob/main/images/project_banner.png" 
       alt="Wholesale Customer Segmentation Banner" width="1000">
</p>

<h4 align="center">
  <em>End-to-End Data Mining • Customer Segmentation • Machine Learning • Visualization</em>
</h4>


# data-mining-project--wholesale-customer-clustering
End-to-end data mining and clustering project
#  Wholesale Customer Segmentation – End-to-End Data Mining Project

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Scikit-learn](https://img.shields.io/badge/ML-ScikitLearn-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> **End-to-end customer segmentation and classification project using data mining techniques.**

---

##  Project Overview

This project performs **customer segmentation** on the **Wholesale Customers Dataset** using both **unsupervised and supervised learning** approaches.  
It demonstrates a full data mining pipeline: **data preprocessing, clustering, visualization, and model evaluation**.

---

##  Key Objectives
- Identify distinct customer groups using **K-Means** and **Hierarchical Clustering**
- Analyze cluster characteristics for business insights
- Compare performance of **Logistic Regression** and **SVM** on original vs. resampled data
- Visualize metrics such as precision, recall, and F1-score

---

##  Workflow
| Step | Description |
|------|--------------|
| 1️⃣ | **Data Cleaning & Preprocessing** – handled missing values, normalization |
| 2️⃣ | **K-Means Clustering** – optimal K selection, silhouette analysis |
| 3️⃣ | **Hierarchical Clustering** – dendrogram visualization |
| 4️⃣ | **Cluster Profiling** – interpret customer segments |
| 5️⃣ | **Supervised Models** – compare SVM vs Logistic Regression |
| 6️⃣ | **Visualization & Reporting** – matplotlib + seaborn visual summaries |

---

## 🛠️ Tech Stack
| Category | Tools Used |
|-----------|-------------|
| Language | Python |
| Libraries | `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn` |
| Environment | Google Colab |
| Version Control | GitHub |

---

##  Example Visualizations

| K-Means Clusters | Hierarchical Dendrogram | Model Performance |
|------------------|--------------------------|-------------------|
| ![Clusters](images/kmeans_clusters_visualization.png) | ![Dendrogram](images/dendrogram.png) | ![Comparison](images/clustering_vs_supervised.png) |

---

##  Insights & Findings
- **Optimal K (K-Means):** 3 clusters  
- **Cluster 0:** High spending on grocery/detergents → likely retail shops  
- **Cluster 1:** Moderate spenders → balanced customers  
- **Cluster 2:** Low spenders → small or niche buyers  
- **SVM (resampled)** achieved slightly better F1-score compared to Logistic Regression

---

##  How to Run

Open this notebook directly in Google Colab:
 [**Run in Colab**](https://colab.research.google.com/github/Zahra58/data-mining-project--wholesale-customer-clustering/blob/main/data_mining.ipynb)

Or clone locally:
```bash
git clone https://github.com/Zahra58/data-mining-project--wholesale-customer-clustering.git
cd data-mining-project--wholesale-customer-clustering
pip install -r requirements.txt


## Model Artifacts Summary
##  Trained Models
| Model Type           | File Path | Description |
|----------------------|------------|--------------|
| SVM Classifier       | `models/svm_model.pkl` | Support Vector Machine used for classification |
| Decision Tree        | `models/decision_tree_model.pkl` | Tree-based model for interpretability |
| Random Forest        | `models/random_forest_model.pkl` | Ensemble learning for better generalization |
| Logistic Regression  | `models/log_reg_model.pkl` | Baseline linear classifier |
| KNN (k=3)            | `models/knn_k3_model.pkl` | Instance-based learner |

<p align="center">
  <img src="images/banner.png" alt="Wholesale Customer Segmentation Project Banner">
</p>

<p align="center">
  <a href="https://github.com/Zahra58"><img src="https://img.shields.io/badge/GitHub-Zahra58-181717?style=for-the-badge&logo=github"></a>
  <a href="https://www.linkedin.com/in/zahraetebari/"><img src="https://img.shields.io/badge/LinkedIn-Zahra%20Etebari-blue?style=for-the-badge&logo=linkedin"></a>
  <img src="https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/scikit--learn-1.5+-orange?style=for-the-badge&logo=scikit-learn">
</p>

