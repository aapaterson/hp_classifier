import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Data
df = pd.read_csv("abs_data.tsv", sep="\t").drop_duplicates(subset=["absID"])
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


# Pipes
pipe = sk.pipeline.Pipeline(
    steps=[
        ("vectorizer", sk.feature_extraction.text.TfidfVectorizer()),
        ("dim_red", sk.decomposition.PCA()),
        ("clf", sk.linear_model.LogisticRegression()),
    ]
)

# Scoring
## Logistic Regresssion
pipe.set_params(
    vectorizer__stop_words="english",
    dim_red__n_components=1000,
).fit(X_train, y_train)
print(pipe.score(X_val, y_val))
print(sk.metrics.fbeta_score(pipe.predict(X_val), y_val, beta=10))
print(sk.metrics.f1_score(pipe.predict(X_val), y_val))
print(sk.metrics.matthews_corrcoef(pipe.predict(X_val), y_val))

# Scoring
# Logistic Lasso
pipe.set_params(
    vectorizer__stop_words="english",
    # vectorizer__token_pattern="[a-z]+\\w*",
    clf__l1_ratio=1,
    clf__solver="liblinear",
).fit(X_train, y_train)
print(pipe.score(X_val, y_val))
print(sk.metrics.fbeta_score(pipe.predict(X_val), y_val, beta=10))
print(sk.metrics.f1_score(pipe.predict(X_val), y_val))
print(sk.metrics.matthews_corrcoef(pipe.predict(X_val), y_val))

# PCA
pca = pipe["dim_red"]
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, "ro-", linewidth=2)
plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Proportion of Variance Explained")
plt.show()

# Word Cloud
vectorizer = sk.feature_extraction.text.TfidfVectorizer(
    stop_words="english",
    token_pattern="[a-z]+\\w*",
    # strip_accents="ascii",
)
tfidf_matrix = vectorizer.fit_transform(X_train)
wordcloud = WordCloud(
    width=800, height=400, background_color="white"
).generate_from_frequencies(vectorizer.vocabulary_)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
