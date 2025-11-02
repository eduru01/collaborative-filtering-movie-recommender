# MovieLens Recommendation System — Matrix Factorization & SVD

This project builds a movie recommendation system using collaborative filtering techniques:

✅ **Problem 1:** Custom Latent Factor Model (Matrix Factorization)

* Tested different numbers of latent features (K = 2, 10, 50)
* Demonstrated how increasing model complexity without regularization leads to overfitting
* Added regularization to improve validation and test performance

✅ **Problem 2:** SVD Model using Surprise Library

* Performed hyperparameter search to optimize model performance
* Achieved strong generalization on leaderboard data

### 🔍 Key Results

| Model                                  | Validation MAE |  Test MAE |
| -------------------------------------- | -------------: | --------: |
| Latent Factor (K=2)                    |          0.751 |     0.742 |
| Latent Factor (K=50, + regularization) |          0.749 |     0.740 |
| **SVD (Best Model)**                   |      **0.729** | **0.716** |

### 🚀 What I Learned

* How to diagnose and prevent overfitting using regularization
* The importance of tuning hyperparameters for real performance gains
* How recommender systems capture patterns in user behavior

### 🛠 Tech Used

* Python, NumPy, Pandas, Matplotlib
* Surprise library for recommendation models

---

This project strengthened my experience in machine learning, model tuning, and delivering data-driven insights that improve recommendation accuracy.
