import pandas as pd
import numpy as np
from numpy.linalg import norm  # para normalizar datos
from tqdm.notebook import tqdm
from sklearn.preprocessing import Normalizer
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from tabulate import tabulate
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn")


def leer_dpto(Departamento, drop_duplicates=False, delimit_min_count=1):
    df = pd.read_csv(
        "../../../../data/processed/internal/admisiones/chat_reports/clean_logs/logs_"
        + Departamento.capitalize()
        + "_sec_No.csv",
        encoding="utf-8",
        keep_default_na=False,
        na_values=["", " "],
    )
    df = df[
        (
            (df["IniciaTecBotBool"] == False)
            & (df["HuboTransferBool"] == False)
            & (df["Emisor"] == "PROSPECTO")
        )
    ]
    df = df[["OracionLematizada"]]
    df.dropna(inplace=True)
    if drop_duplicates:
        df.drop_duplicates(subset="OracionLematizada", keep="first", inplace=True)
    df["count"] = [len(x.split()) for x in df["OracionLematizada"]]
    df = df[df["count"] > delimit_min_count]
    df = df.reset_index()
    del df["index"]
    del df["count"]
    df["OracionLematizada"] = df["OracionLematizada"].replace(np.NaN, "")
    df["OracionLematizada"] = [str(text) for text in df["OracionLematizada"]]
    return df


def leer_menus_labels(name="menus-labels", delimit_min_count=1):
    df_menus = pd.read_csv(name + ".csv")
    df_menus.dropna(inplace=True)
    df_menus["count"] = [len(x.split()) for x in df_menus["OracionLematizada"]]
    df_menus = df_menus[df_menus["count"] > delimit_min_count]
    df_menus = df_menus.reset_index()
    del df_menus["index"]
    del df_menus["count"]
    return df_menus


def leer_menus(drop_duplicates=False, delimit_min_count=1):
    df_menus = pd.read_csv("menus.csv")
    df_menus.dropna(inplace=True)
    if drop_duplicates:
        df_menus.drop_duplicates(subset="OracionLematizada", keep="first", inplace=True)
    df_menus = df_menus[df_menus["prospecto"] == 1]
    df_menus["count"] = [len(x.split()) for x in df_menus["OracionLematizada"]]
    df_menus = df_menus[df_menus["count"] > delimit_min_count]
    df_menus = df_menus.reset_index()
    del df_menus["index"]
    del df_menus["count"]
    return df_menus


def cosine_similarity_top_k(vect, alpha=1, k_primeros=3, idx=54):
    recommended = []
    matrix = []
    similarity_threshold = 0
    matrix = cosine_similarity(vect)
    matrix_arr = matrix[idx]
    matrix_arr = np.delete(matrix_arr, idx)
    similarity_threshold = np.median(matrix_arr) + alpha * matrix_arr.std()
    score_series = pd.Series(matrix[idx]).sort_values(ascending=False)
    score_series = score_series.drop(idx)
    top_k_indices = list(score_series.iloc[0:k_primeros].index)
    recommended = [i for i in top_k_indices if score_series[i] >= similarity_threshold]
    return recommended, list(score_series[0:k_primeros])


def distance_similarity_top_k(vect, metric="minkowski", alpha=1, k_primeros=3, idx=54):
    recommended = []
    matrix = []
    similarity_threshold = 0
    matrix = pairwise_distances(vect.todense(), metric=metric, n_jobs=-1)
    matrix = Normalizer().fit_transform(matrix)
    matrix_arr = matrix[idx]
    matrix_arr = np.delete(matrix_arr, idx)
    similarity_threshold = np.median(matrix_arr) + alpha * matrix_arr.std()
    score_series = pd.Series(matrix[idx]).sort_values(ascending=True)
    score_series = score_series.drop(idx)
    top_k_indices = list(score_series.iloc[0:k_primeros].index)
    recommended = [i for i in top_k_indices if score_series[i] < similarity_threshold]
    return recommended, list(score_series[0:k_primeros])


def recommend(df_main, df_ref, column_name, k_primeros, alpha, metric, idx=54):
    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    list_message_index = []
    list_message = []
    list_tuples = []
    total = len(df_main)
    result = pd.DataFrame()
    if metric == "cosine":
        with tqdm(
            total=total,
            bar_format="{bar}|{desc}{percentage:3.0f}% {r_bar}",
            leave=False,
        ) as pbar:
            for index, row in enumerate(df_main.itertuples(), 1):
                list_message_index.append(index)
                row_interest = str(getattr(row, column_name))
                list_message.append(row_interest)
                list_ref = list(df_ref[column_name].values)
                list_ref.append(row_interest)
                X_2D_list_ref = tfidf.fit_transform(pd.Series(list_ref))
                recommendations_idx, scores = cosine_similarity_top_k(
                    X_2D_list_ref, alpha, k_primeros, idx
                )
                recommendations = df_ref.iloc[recommendations_idx][column_name].values
                list_tuples.append([recommendations_idx, list(recommendations), scores])
                pbar.update(1)
    else:
        with tqdm(
            total=total,
            bar_format="{bar}|{desc}{percentage:3.0f}% {r_bar}",
            leave=False,
        ) as pbar:
            for index, row in enumerate(df_main.itertuples(), 1):
                list_message_index.append(index)
                row_interest = str(getattr(row, column_name))
                list_message.append(row_interest)
                list_ref = list(df_ref[column_name].values)
                list_ref.append(row_interest)
                X_2D_list_ref = tfidf.fit_transform(pd.Series(list_ref))
                recommendations_idx, scores = distance_similarity_top_k(
                    list_ref, metric, alpha, k_primeros, idx
                )
                recommendations = df_ref.iloc[recommendations_idx][column_name].values
                list_tuples.append([recommendations_idx, list(recommendations), scores])
                pbar.update(1)

    result["message_index"] = list_message_index
    result["message"] = list_message
    temp = pd.DataFrame(data=[[x] for x in np.array(list_tuples)])
    temp.columns = ["x"]
    temp["idx"], temp["rec"], temp["score"] = zip(*temp.pop("x"))
    result["idx"] = temp["idx"]
    result["idx"] = [int(x[0]) for x in result["idx"]]
    result["idx"] = [int(x) if len(str(x).split()) > 0 else -1 for x in result["idx"]]
    result["idx"] = result["idx"].astype("int")
    result["recommendation"] = temp["rec"]
    result["recommendation"] = [str(x[0]) for x in result["recommendation"]]
    result["recommendation"] = [
        str(x) if len(str(x).split()) > 0 else -1 for x in result["recommendation"]
    ]
    result["recommendation"] = result["recommendation"].astype("object")
    result["score"] = temp["score"]
    result["score"] = [float(x[0]) for x in result["score"]]
    result["score"] = [
        float(x) if len(str(x).split()) > 0 else -1 for x in result["score"]
    ]
    result["score"] = result["score"].astype("float")
    return result


# Create a function
def select_n_components(var_ratio, goal_var: float) -> int:
    # Set initial variance explained so far
    total_variance = 0.0

    # Set initial number of features
    n_components = 0

    # For the explained variance of each feature:
    for explained_variance in var_ratio:

        # Add the explained variance to the total
        total_variance += explained_variance

        # Add one to the number of components
        n_components += 1

        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break

    # Return the number of components
    return n_components


def cluster_vs_intents(vect, idx, filter_zeros=False):
    matrix = []
    matrix = cosine_similarity(vect)
    matrix_arr = matrix[idx]
    matrix_arr = np.delete(matrix_arr, idx)
    matrix_arr_filtered = matrix_arr
    if filter_zeros:
        matrix_arr_filtered = matrix_arr[np.where(matrix_arr > 0)]
    return [1 if x > 0 else 0 for x in matrix_arr], np.mean(matrix_arr_filtered)


def run_cluster_analysis(
    df, cluster_label, text_label, menus, tfidf, total, filter_zeros
):
    # MAIN FUNCTION
    with tqdm(
        total=total, bar_format="{bar}|{desc}{percentage:3.0f}% {r_bar}", leave=False
    ) as pbar:
        list_cluster = []
        list_intent = []
        list_intent_text = []
        list_mean = []
        list_count = []
        for cluster in range(0, total):
            list_documents = menus["OracionLematizada"].values
            list_query = df[df[cluster_label] == cluster][text_label].values
            docs_tfidf = tfidf.fit_transform(list_documents)
            query_tfidf = tfidf.transform(list_query)
            cosineSimilarities = cosine_similarity(docs_tfidf, query_tfidf)
            list_intents_means = [np.mean(sims) for sims in cosineSimilarities]
            list_cluster.append(cluster)
            list_intent.append(np.argmax(list_intents_means))
            list_intent_text.append(
                menus[menus["idx"] == np.argmax(list_intents_means)][
                    "OracionLematizada"
                ].values[0]
            )
            list_mean.append(np.max(list_intents_means))
            list_count.append(len(list_query))
            pbar.update(1)

    df_sim = pd.DataFrame(
        list(zip(list_cluster, list_count, list_intent, list_intent_text, list_mean)),
        columns=["cluster", "count", "intent", "intent text", "mean score"],
    )

    # TABULATE
    print("mean score on cluster argmax")
    print(
        tabulate(
            df_sim,
            showindex=False,
            headers=["cluster", "count", "intent", "intent text", "mean score"],
            tablefmt="pretty",
        )
    )
    print()

    # PLOT AXES
    fig = plt.gcf()
    fig.set_size_inches(20, 13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=16)
    plt.xlabel("cluster", fontsize=18)
    plt.ylabel("mean score", fontsize=16)
    df_to_plot = df_sim.copy()
    ax = sns.barplot(data=df_to_plot, x="cluster", y="mean score", linewidth=3)
    rango = range(0, total + 10, 10)
    ax.set_xticks(rango)
    for x in rango:
        ax.axvline(x, linestyle="-", color="#7f7f7f", linewidth=0.5)
    plt.show()

    # PLOT HIST
    fig = plt.gcf()
    fig.set_size_inches(20, 13)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("cluster", fontsize=18)
    plt.ylabel("mean score", fontsize=16)
    ax = sns.histplot(data=df_sim, x="mean score")
    plt.show()

    # PLOT HIST zoomed
    fig = plt.gcf()
    fig.set_size_inches(20, 13)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("cluster", fontsize=18)
    plt.ylabel("mean score", fontsize=16)
    ax = sns.histplot(data=df_sim[df_sim["mean score"] > 0], x="mean score")
    plt.show()

    # DF INFO
    print()
    print("DF INFO:")
    df_sim.info()
    print("")

    return df_sim


def to_vector(texto, we):
    tokens = texto.split()
    vec = np.zeros(300)
    for word in tokens:
        # si la palabra está la acumulamos
        if word in we:
            vec += we[word]
    return vec / norm(vec)


def similarity(texto_1, texto_2):
    vec_1 = to_vector(texto_1)
    vec_2 = to_vector(texto_2)
    sim = cosine_similarity([vec_1], [vec_2])[0][0]
    return sim


def clasifica(texto, clases):
    sims = [similarity(texto, clase) for clase in clases]
    indice = np.argmax(np.array(sims))
    valor = np.max(np.array(sims))
    return indice, valor


def run_cluster_analysis_w2v(
    we, model_name, df, cluster_label, text_label, menus, total
):
    # MAIN FUNCTION
    with tqdm(
        total=total, bar_format="{bar}|{desc}{percentage:3.0f}% {r_bar}", leave=False
    ) as pbar:
        list_cluster = []
        list_intent = []
        list_intent_text = []
        list_mean = []
        list_count = []

        vectores_tec = [
            to_vector(texto, we) for texto in menus["OracionLematizada"].values
        ]
        Y = np.vstack(np.array(vectores_tec))
        for cluster in range(0, total):
            vectores_prospecto = [
                to_vector(texto, we)
                for texto in df[df[cluster_label] == cluster][text_label].values
            ]
            X_ = np.vstack(np.array(vectores_prospecto))
            cosineSimilarities = cosine_similarity(Y, X_)
            list_intents_means = [np.mean(sims) for sims in cosineSimilarities]
            list_cluster.append(cluster)
            list_intent.append(np.argmax(list_intents_means))
            list_intent_text.append(
                menus[menus["idx"] == np.argmax(list_intents_means)][
                    "OracionLematizada"
                ].values[0]
            )
            list_mean.append(np.max(list_intents_means))
            list_count.append(len(vectores_prospecto))
            pbar.update(1)

    df_sim = pd.DataFrame(
        list(zip(list_cluster, list_count, list_intent, list_intent_text, list_mean)),
        columns=["cluster", "count", "intent", "intent text", "mean score"],
    )
    df_sim.to_csv(
        "datasets/" + str(model_name) + "_" + str(total) + "_meanscores.csv",
        index=False,
    )
    # TABULATE
    # print("mean score on cluster argmax")
    # print(
    #    tabulate(
    #        df_sim,
    #        showindex=False,
    #        headers=["cluster", "count", "intent", "intent text", "mean score"],
    #        tablefmt="pretty",
    #    )
    # )
    # print()

    # PLOT AXES
    fig = plt.gcf()
    fig.set_size_inches(20, 13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=16)
    plt.xlabel("cluster", fontsize=18)
    plt.ylabel("mean score", fontsize=16)
    df_to_plot = df_sim.copy()
    ax = sns.barplot(data=df_to_plot, x="cluster", y="mean score", linewidth=3)
    rango = range(0, total + 10, 10)
    ax.set_xticks(rango)
    for x in rango:
        ax.axvline(x, linestyle="-", color="#7f7f7f", linewidth=0.5)
    plt.savefig(
        "plots/" + str(model_name) + "_" + str(total) + "_meanscores_bar.png",
        facecolor=fig.get_facecolor(),
    )
    # plt.show()
    plt.close()

    # PLOT HIST
    fig = plt.gcf()
    fig.set_size_inches(20, 13)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("cluster", fontsize=18)
    plt.ylabel("mean score", fontsize=16)
    ax = sns.histplot(data=df_sim, x="mean score")
    plt.savefig(
        "plots/" + str(model_name) + "_" + str(total) + "_meanscores_hist.png",
        facecolor=fig.get_facecolor(),
    )
    # plt.show()
    plt.close()

    # PLOT HIST zoomed
    fig = plt.gcf()
    fig.set_size_inches(20, 13)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("cluster", fontsize=18)
    plt.ylabel("mean score", fontsize=16)
    ax = sns.histplot(data=df_sim[df_sim["mean score"] > 0], x="mean score")
    plt.savefig(
        "plots/"
        + str(model_name)
        + "_"
        + str(total)
        + "_meanscores_hist_biggerthanzero.png",
        facecolor=fig.get_facecolor(),
    )
    # plt.show()
    plt.close()

    # DF INFO
    # print()
    # print("DF INFO:")
    # df_sim.info()
    # print("")

    return df_sim


def run_precision(df, menus, threshold, show_table=False):
    # PRECISION OF CLUSTERING
    df_to_plot_grouped = (
        df[df["mean score"] > threshold].groupby(by="intent").count().reset_index()
    )
    idxs = df_to_plot_grouped["intent"].values
    list_intent = []
    list_count = []
    for i in range(0, len(menus)):
        if i in idxs:
            list_intent.append(i)
            list_count.append(
                df_to_plot_grouped.loc[df_to_plot_grouped["intent"] == i][
                    "cluster"
                ].values[0]
            )
        else:
            list_intent.append(i)
            list_count.append(0)

    df_to_plot_grouped = pd.DataFrame(
        list(zip(list_intent, list_count)), columns=["intent", "count"]
    )
    df_to_plot_grouped["count_binary"] = [
        1 if value != 0 else 0 for value in df_to_plot_grouped["count"]
    ]
    count_sum = np.sum(df_to_plot_grouped["count"].values)
    count_binary_sum = np.sum(df_to_plot_grouped["count_binary"].values)
    if show_table:
        print(f"cosine_similarity_threshold: {threshold}")
        print(f"all intents present in dataframe: {count_sum}")
        print(f"distinct intents present in dataframe: {count_binary_sum}")
        print(
            f"precision of intents catched: {round(count_binary_sum/len(menus), 4)}\n"
        )
        print(
            tabulate(
                df_to_plot_grouped,
                showindex=False,
                headers=["intent", "count", "count_binary"],
                tablefmt="pretty",
            )
        )
    return (
        threshold,
        count_sum,
        count_binary_sum,
        round(count_binary_sum / len(menus), 4),
    )


def get_tf_idf_query_similarity_1(vectorizer, docs_tfidf, query, k=1):
    query_tfidf = vectorizer.transform(query)
    cosineSimilarities = cosine_similarity(query_tfidf, docs_tfidf)
    return np.argmax(cosineSimilarities, axis=1), np.max(cosineSimilarities, axis=1)


def get_tf_idf_query_similarity(vectorizer, docs_tfidf, query, k):
    query_tfidf = vectorizer.transform(query)
    cosineSimilarities = cosine_similarity(query_tfidf, docs_tfidf)
    index_top_k = [list(arr.argsort()[-k:][::-1]) for arr in cosineSimilarities]
    value_top_k = []
    [value_top_k.append(sims[k]) for k, sims in zip(index_top_k, cosineSimilarities)]
    return index_top_k, value_top_k


def run_w2v(df, df_menus, id_model, n_clusters_kmeans):
    # Load model. References:
    # https://github.com/dccuchile/spanish-word-embeddings
    # https://github.com/dccuchile/spanish-word-embeddings/blob/master/examples/Ejemplo_WordVectors.md
    # https://github.com/mquezada/starsconf2018-word-embeddings/blob/master/code/Workshop.ipynb
    models = [
        "embeddings-new_large-general_3B_fasttext.vec",
        "fasttext-sbwc.3.6.e20.vec",
        "glove-sbwc.i25.vec",
        "SBW-vectors-300-min5.txt",
        "wiki.es.vec",
    ]
    model = models[id_model]
    we = KeyedVectors.load_word2vec_format(
        "../../../../ignore/models/" + model, limit=100000
    )
    # Load vectors of input text
    samples = df["OracionLematizada"].values
    vectores = [to_vector(texto, we) for texto in samples]
    mask = np.all(np.isnan(vectores) | np.equal(vectores, 0), axis=1)
    X = np.vstack(np.array(vectores)[~mask])
    # Run kmeans with vectors that are not nan
    kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42)
    intents = kmeans.fit_transform(X)
    labels_ = kmeans.labels_
    print(
        f"silhouette score: {silhouette_score(X, labels_, sample_size=1000, random_state=42)}"
    )
    # Filter dataframe with valid vectors (no nan's)
    temp = pd.DataFrame(list(zip(samples[~mask], labels_)), columns=["sample", "label"])
    temp.to_csv(
        "datasets/" + str(model) + "_" + str(n_clusters_kmeans) + ".csv", index=False
    )
    # Run analysis of clusters
    df_sim = run_cluster_analysis_w2v(
        we, model, temp, "label", "sample", df_menus, n_clusters_kmeans
    )
    # Run precisions at different similarity thresholds
    list_threshold = []
    list_precision = []
    list_count = []
    list_count_binary = []
    for threshold in np.linspace(0, 1, 50):
        val, count_sum, count_binary_sum, precision = run_precision(
            df_sim, df_menus, threshold, show_table=False
        )
        list_threshold.append(val)
        list_precision.append(precision)
        list_count.append(count_sum)
        list_count_binary.append(count_binary_sum)
    df_precision = pd.DataFrame(
        list(zip(list_threshold, list_precision, list_count, list_count_binary)),
        columns=["threshold", "precision", "sum count", "binary count"],
    )
    df_precision.to_csv(
        "datasets/" + str(model) + "_" + str(n_clusters_kmeans) + "_precision.csv",
        index=False,
    )


# def run_precision(df, menus, threshold, show_table=False):
#    # PRECISION OF CLUSTERING
#    df_to_plot_grouped = (
#        df[df["mean score"] > threshold].groupby(by="intent").count().reset_index()
#    )
#    idxs = df_to_plot_grouped["intent"].values
#    list_intent = []
#    list_count = []
#    for i in range(0, len(menus)):
#        if i in idxs:
#            list_intent.append(i)
#            list_count.append(
#                df_to_plot_grouped.loc[df_to_plot_grouped["intent"] == i][
#                    "cluster"
#                ].values[0]
#            )
#        else:
#            list_intent.append(i)
#            list_count.append(0)
#
#    df_to_plot_grouped = pd.DataFrame(
#        list(zip(list_intent, list_count)), columns=["intent", "count"]
#    )
#    df_to_plot_grouped["count_binary"] = [
#        1 if value is not 0 else 0 for value in df_to_plot_grouped["count"]
#    ]
#    count_sum = np.sum(df_to_plot_grouped["count"].values)
#    count_binary_sum = np.sum(df_to_plot_grouped["count_binary"].values)
#    print(f"cosine_similarity_threshold: {threshold}")
#    print(f"all intents present in dataframe: {count_sum}")
#    print(f"distinct intents present in dataframe: {count_binary_sum}")
#    print(f"precision of intents catched: {round(count_binary_sum/len(menus), 4)}\n")
#    if show_table:
#        print(
#            tabulate(
#                df_to_plot_grouped,
#                showindex=False,
#                headers=["intent", "count", "count_binary"],
#                tablefmt="pretty",
#            )
#        )
