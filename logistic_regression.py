import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("abs_data.tsv", sep="\t").fillna(0)
df = df.drop_duplicates(subset=["absID"])
df_train, df_test_sup = sk.model_selection.train_test_split(
    df, test_size=0.3, stratify=df["response"], random_state=50
)
df_val, df_test = sk.model_selection.train_test_split(
    df_test_sup, test_size=0.5, stratify=df_test_sup["response"], random_state=50
)
X_train, y_train, X_val, y_val, X_test, y_test = (
    df_train["abs"],
    df_train["response"],
    df_val["abs"],
    df_val["response"],
    df_test["abs"],
    df_test["response"],
)
pipe = sk.pipeline.Pipeline(
    steps=[
        ("vectorizer", sk.feature_extraction.text.TfidfVectorizer()),
        ("dim_red", sk.decomposition.PCA()),
        ("clf", sk.linear_model.LogisticRegression()),
    ]
)
pipe.set_params(dim_red__n_components=1000).fit(X_train, y_train)

pipe.score(X_val, y_val)
sk.metrics.fbeta_score(pipe.predict(X_val), y_val, beta=10)

pca = pipe["dim_red"]
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, "ro-", linewidth=2)
plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Proportion of Variance Explained")
plt.show()
