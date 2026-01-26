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
def logistic_scoring(accents, min_df, pca_comp):
    pca_pipe.set_params(
        vectorizer__stop_words="english",
        vectorizer__strip_accents=accents,
        vectorizer__min_df=min_df,
        dim_red__n_components=pca_comp,
    ).fit(X_train, y_train)
    return (
        pca_pipe.score(X_test, y_test),
        sk.metrics.matthews_corrcoef(pca_pipe.predict(X_test), y_test),
    )


## Logistic Lasso
def lasso_scoring(accents, min_df):
    non_pca_pipe.set_params(
        vectorizer__stop_words="english",
        vectorizer__strip_accents=accents,
        vectorizer__min_df=min_df,
        clf__l1_ratio=1,
        clf__solver="liblinear",
    ).fit(X_train, y_train)
    return (
        non_pca_pipe.score(X_test, y_test),
        sk.metrics.matthews_corrcoef(non_pca_pipe.predict(X_test), y_test),
    )


logistic_acc, logistic_mcc = logistic_scoring(None, 0.0, 1000)
lasso_none_acc, lasso_none_mcc = lasso_scoring(None, 0.0)
lasso_unicode_acc, lasso_unicode_mcc = lasso_scoring("unicode", 0.0)
lasso_ascii_acc, lasso_ascii_mcc = lasso_scoring("ascii", 0.0)
lasso_mindf_acc, lasso_mindf_mcc = lasso_scoring("unicode", 0.1)


# Plotting
## Scores
colors = ["#1b9e77", "#a9f971", "#fdaa48", "#6890F0", "#A890F0"]
fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
ax[0].bar(
    x=["Logistic", "\n Lasso", "Lasso Unicode", "\n Lasso Ascii", "Lasso Min_DF"],
    height=[
        logistic_acc,
        lasso_none_acc,
        lasso_unicode_acc,
        lasso_ascii_acc,
        lasso_mindf_acc,
    ],
    color=colors,
)
ax[0].set_ylim(top=1)
ax[0].set_title("Acccuracy")
ax[1].bar(
    x=["Logistic", "\n Lasso", "Lasso Unicode", "\n Lasso Ascii", "Lasso Min_DF"],
    height=[
        logistic_mcc,
        lasso_none_mcc,
        lasso_unicode_mcc,
        lasso_ascii_mcc,
        lasso_mindf_mcc,
    ],
    color=colors,
)
ax[1].set_ylim(top=1, bottom=-1)
ax[1].set_title("Matthew's Correlation Coefficient (MCC)")
plt.show()

plt.clf()
plt.close()

## Word Cloud
vectorizer = sk.feature_extraction.text.TfidfVectorizer(
    stop_words="english",
    strip_accents="ascii",
    # min_df=0.1,
)
tfidf_matrix = vectorizer.fit_transform(X_train)
wordcloud = WordCloud(
    width=3200, height=1600, background_color="white"
).generate_from_frequencies(vectorizer.vocabulary_)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Only Ascii", fontsize=40)
plt.show()

plt.clf()
plt.close()

## PCA
pca = pca_pipe["dim_red"]
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, "ro-", linewidth=2)
plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Proportion of Variance Explained")
plt.show()

plt.clf()
plt.close()


# Comparing Corpora
def tfidf_compare(corpus1, corpus2):
    vectorizer = sk.feature_extraction.text.TfidfVectorizer(
        stop_words="english",
    )
    mat1 = vectorizer.fit_transform(corpus1)
    mat2_hat = vectorizer.transform(corpus2)
    mat2 = vectorizer.fit_transform(corpus2)
    mat1_hat = vectorizer.transform(corpus1)

    tfidf1 = pd.DataFrame.sparse.from_spmatrix(mat1)
    tfidf2_hat = pd.DataFrame.sparse.from_spmatrix(mat2_hat)
    tfidf2 = pd.DataFrame.sparse.from_spmatrix(mat2)
    tfidf1_hat = pd.DataFrame.sparse.from_spmatrix(mat1_hat)

    d1 = math.dist(pd.DataFrame.mean(tfidf1), pd.DataFrame.mean(tfidf2_hat))
    d2 = math.dist(pd.DataFrame.mean(tfidf2), pd.DataFrame.mean(tfidf1_hat))

    return (d1 + 1) / (d2 + 2)


tfidf_compare(X_train, X_test)
