import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import math


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
pca_pipe = sk.pipeline.Pipeline(
    steps=[
        ("vectorizer", sk.feature_extraction.text.TfidfVectorizer()),
        ("dim_red", sk.decomposition.PCA()),
        ("clf", sk.linear_model.LogisticRegression()),
    ]
)
non_pca_pipe = sk.pipeline.Pipeline(
    steps=[
        ("vectorizer", sk.feature_extraction.text.TfidfVectorizer()),
        ("transformer", sk.preprocessing.FunctionTransformer(np.log1p)),
        ("clf", sk.linear_model.LogisticRegression()),
    ]
)


# Scoring
## Logistic Regresssion
pca_pipe.set_params(
    vectorizer__stop_words="english",
    # vectorizer__token_pattern="[a-z]+\\w*",
    vectorizer__strip_accents="ascii",
    vectorizer__min_df=0.01,
    dim_red__n_components=100,
).fit(X_train, y_train)
print(pca_pipe.score(X_val, y_val))
print(sk.metrics.fbeta_score(pca_pipe.predict(X_val), y_val, beta=10))
print(sk.metrics.f1_score(pca_pipe.predict(X_val), y_val))
print(sk.metrics.matthews_corrcoef(pca_pipe.predict(X_val), y_val))

# Scoring
# Logistic Lasso
non_pca_pipe.set_params(
    vectorizer__stop_words="english",
    # vectorizer__token_pattern="[a-z]+\\w*",
    # vectorizer__strip_accents="ascii",
    vectorizer__min_df=(10 / df.shape[0]),
    clf__l1_ratio=1,
    clf__solver="liblinear",
).fit(X_train, y_train)
print(non_pca_pipe.score(X_val, y_val))
print(sk.metrics.fbeta_score(non_pca_pipe.predict(X_val), y_val, beta=10))
print(sk.metrics.f1_score(non_pca_pipe.predict(X_val), y_val))
print(sk.metrics.matthews_corrcoef(non_pca_pipe.predict(X_val), y_val))

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
    # token_pattern="[a-z]+\\w*",
    strip_accents="ascii",
    min_df=(10 / df.shape[0]),
)
tfidf_matrix = vectorizer.fit_transform(X_train)
wordcloud = WordCloud(
    width=800, height=400, background_color="white"
).generate_from_frequencies(vectorizer.vocabulary_)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


corpus1 = vectorizer.fit_transform(X_train)
corpus2 = vectorizer.transform(X_test)


def tfidf_compare(corpus1, corpus2):
    mat1 = pd.DataFrame.sparse.from_spmatrix(corpus1)
    mat2 = pd.DataFrame.sparse.from_spmatrix(corpus2)
    return math.dist(pd.DataFrame.mean(mat1), pd.DataFrame.mean(mat2))


tfidf_compare(corpus1, corpus2)
