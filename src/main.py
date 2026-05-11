import os
import sys
import random
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from turftopic import KeyNMF


def parse_args():
    parser = argparse.ArgumentParser(description="KeyNMF topic modelling on StorySeeker corpus")
    parser.add_argument("--n_topics",   type=int,   default=10,    help="Number of topics")
    parser.add_argument("--top_n",      type=int,   default=15,    help="Keywords extracted per document")
    parser.add_argument("--encoder",    type=str,   default="paraphrase-MiniLM-L3-v2", help="Sentence-transformer encoder")
    parser.add_argument("--min_df",     type=int,   default=5,     help="Min document frequency for vectorizer")
    parser.add_argument("--max_df",     type=float, default=0.5,   help="Max document frequency threshold (Schofield et al., 2017)")
    parser.add_argument("--seed",       type=int,   default=42,    help="Random seed")
    parser.add_argument("--classify",   action="store_true",       help="Run logistic regression classifier on topic distributions")
    parser.add_argument("--downsample", action="store_true",       help="Downsample majority class before classification")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(out_dir: str, n_topics: int) -> str:
    log_path = os.path.join(out_dir, f"run_n{n_topics}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path


def log(msg: str = ""):
    logging.info(msg)


def format_topic_words(topic_entry, top_k: int = 10) -> str:
    _, word_weights = topic_entry
    return ", ".join(str(w) for w, _ in word_weights[:top_k])


def topic_label(topic_entry, top_k: int = 4) -> str:
    tid, word_weights = topic_entry
    words = ", ".join(str(w) for w, _ in word_weights[:top_k])
    return f"T{tid}: {words}"


def log_all_topics(model: KeyNMF, n_topics: int, top_k: int = 10):
    topics = model.get_topics()
    log("\n--- Topics (top words) ---")
    for i in range(n_topics):
        log(f"Topic {i}: {format_topic_words(topics[i], top_k)}")


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    n_before = len(df)
    df = df.drop_duplicates(subset="text")
    log(f"Removed {n_before - len(df)} duplicate documents ({len(df)} remaining)")
    log(f"Label distribution:\n{df['gold_consensus'].value_counts()}\n")
    return df


def build_model(args) -> KeyNMF:
    vectorizer = CountVectorizer(min_df=args.min_df, max_df=args.max_df)
    return KeyNMF(
        n_components=args.n_topics,
        encoder=args.encoder,
        vectorizer=vectorizer,
        top_n=args.top_n,
        random_state=args.seed,
    )


def print_top_docs(topic_idx: int, topic_dists: np.ndarray, data: list[str], n: int = 2):
    doc_indices = np.argsort(topic_dists[:, topic_idx])[::-1][:n]
    log(f"\n--- Top {n} documents for Topic {topic_idx} ---")
    for rank, i in enumerate(doc_indices, 1):
        log(f"\n[{rank}] (score={topic_dists[i, topic_idx]:.3f})\n{data[i][:400]}...")


def compute_story_skew(topic_dists: np.ndarray, labels: pd.Series) -> np.ndarray:
    return np.array([
        np.average(labels, weights=topic_dists[:, k])
        for k in range(topic_dists.shape[1])
    ])


def plot_topic_lexical_similarity(model: KeyNMF, out_dir: str, n_topics: int):
    #Lexical similarity: cosine over NMF component vectors in vocabulary space.
    #Always low for KeyNMF because keyword extraction + NMF both produce sparse term distributions.
    #Useful as a diagnostic for catching duplicates, NOT for claiming topics are well-separated.
    sim = cosine_similarity(model.components_)
    np.fill_diagonal(sim, 0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1)
    plt.title(f"Lexical Topic Similarity (n={n_topics}) — sparse by construction")
    plt.tight_layout()
    path = os.path.join(out_dir, f"topic_similarity_lexical_n{n_topics}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log(f"Saved: {path}")


def plot_topic_semantic_similarity(model: KeyNMF, out_dir: str, n_topics: int, top_k: int = 10):
    #Semantic similarity: each topic represented by the mean embedding of its top-k words,
    #computed with the same sentence-transformer KeyNMF used. Reveals semantic overlap that
    #lexical similarity misses.
    topics = model.get_topics()
    encoder = model.encoder_

    centroids = []
    for tid in range(n_topics):
        top_words = [str(w) for w, _ in topics[tid][1][:top_k]]
        embeddings = encoder.encode(top_words)
        centroids.append(np.asarray(embeddings).mean(axis=0))
    centroids = np.vstack(centroids)

    sim = cosine_similarity(centroids)
    np.fill_diagonal(sim, 0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(sim, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1)
    plt.title(f"Semantic Topic Similarity (n={n_topics}) — embedding centroids")
    plt.tight_layout()
    path = os.path.join(out_dir, f"topic_similarity_semantic_n{n_topics}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log(f"Saved: {path}")

    i, j = np.unravel_index(np.argmax(sim), sim.shape)
    log(f"Highest semantic similarity: T{i} <-> T{j} = {sim[i, j]:.3f}")


def plot_skew_ranked(story_skew: np.ndarray, topics_out, global_mean: float, out_dir: str, n_topics: int):
    sorted_idx = np.argsort(story_skew)
    skew_sorted = story_skew[sorted_idx]
    labels = [topic_label(topics_out[i], top_k=4) for i in sorted_idx]
    colors = ["#c0504d" if s < global_mean else "#4f81bd" for s in skew_sorted]

    fig_height = max(4, n_topics * 0.45)
    plt.figure(figsize=(10, fig_height))
    plt.barh(range(n_topics), skew_sorted, color=colors, alpha=0.85, edgecolor="white")
    plt.yticks(range(n_topics), labels, fontsize=10)
    plt.axvline(global_mean, color="black", linestyle="--", linewidth=1.2,
                label=f"Global mean ({global_mean:.2f})")
    plt.xlabel("Story skew (weighted mean narrativity)")
    plt.title(f"Topics ranked by narrativity (n={n_topics})")
    plt.xlim(0, 1)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, f"skew_ranked_n{n_topics}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log(f"Saved: {path}")


def plot_topic_by_label(topic_dists: np.ndarray, labels: pd.Series, topics_out, out_dir: str, n_topics: int):
    avg_story    = topic_dists[labels.values == 1].mean(axis=0)
    avg_no_story = topic_dists[labels.values == 0].mean(axis=0)
    diff         = avg_story - avg_no_story
    sorted_idx   = np.argsort(diff)

    topic_labels = [topic_label(topics_out[i], top_k=3) for i in sorted_idx]
    y = np.arange(n_topics)
    width = 0.4

    fig_height = max(4, n_topics * 0.45)
    plt.figure(figsize=(10, fig_height))
    plt.barh(y - width/2, avg_no_story[sorted_idx], width,
             label="no story (label=0)", color="#c0504d", alpha=0.85, edgecolor="white")
    plt.barh(y + width/2, avg_story[sorted_idx], width,
             label="story (label=1)",    color="#4f81bd", alpha=0.85, edgecolor="white")
    plt.yticks(y, topic_labels, fontsize=10)
    plt.xlabel("Mean topic loading within label group")
    plt.title(f"Topic prevalence by label (n={n_topics})")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, f"topic_by_label_n{n_topics}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log(f"Saved: {path}")


def run_classifier(topic_dists: np.ndarray, labels: pd.Series, downsample: bool, seed: int):
    from sklearn.utils import resample

    X, y = topic_dists, labels

    if downsample:
        df_tmp = pd.DataFrame(X)
        df_tmp["label"] = y.values
        majority = df_tmp[df_tmp["label"] == 0]
        minority = df_tmp[df_tmp["label"] == 1]
        majority = resample(majority, replace=False, n_samples=len(minority), random_state=seed)
        df_tmp = pd.concat([majority, minority])
        X = df_tmp.drop("label", axis=1).values
        y = df_tmp["label"]
        log(f"Downsampled to {len(df_tmp)} documents")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    clf = LogisticRegression(random_state=seed, max_iter=1000)
    clf.fit(X_train, y_train)
    log("\n--- Classification report (topic dists -> story/no-story) ---")
    log(classification_report(y_test, clf.predict(X_test)))


def main():
    args = parse_args()
    set_seed(args.seed)

    src_dir   = os.path.dirname(os.path.abspath(__file__))
    root_dir  = os.path.dirname(src_dir)
    data_path = os.path.join(root_dir, "in", "storyseeker.csv")
    out_dir   = os.path.join(root_dir, "out")
    os.makedirs(out_dir, exist_ok=True)

    log_path = setup_logging(out_dir, args.n_topics)

    log(f"=== Run config ===")
    log(f"n_topics={args.n_topics} | encoder={args.encoder} | top_n={args.top_n}")
    log(f"min_df={args.min_df} | max_df={args.max_df} | seed={args.seed}")
    log(f"classify={args.classify} | downsample={args.downsample}")
    log(f"Log saved to: {log_path}\n")

    df   = load_data(data_path)
    data = df["text"].tolist()

    model       = build_model(args)
    topic_dists = model.fit_transform(data)

    log_all_topics(model, args.n_topics)

    for i in range(args.n_topics):
        print_top_docs(i, topic_dists, data, n=2)

    story_skew  = compute_story_skew(topic_dists, df["gold_consensus"])
    global_mean = df["gold_consensus"].mean()
    topics_out  = model.get_topics()

    log("\n\n--- Most story-like topics ---")
    for i in np.argsort(story_skew)[::-1][:5]:
        log(f"Topic {i} | skew={story_skew[i]:.4f} | {format_topic_words(topics_out[i])}")

    log("\n--- Least story-like topics ---")
    for i in np.argsort(story_skew)[:5]:
        log(f"Topic {i} | skew={story_skew[i]:.4f} | {format_topic_words(topics_out[i])}")

    plot_topic_lexical_similarity(model, out_dir, args.n_topics)
    plot_topic_semantic_similarity(model, out_dir, args.n_topics)
    plot_skew_ranked(story_skew, topics_out, global_mean, out_dir, args.n_topics)
    plot_topic_by_label(topic_dists, df["gold_consensus"], topics_out, out_dir, args.n_topics)

    if args.classify:
        run_classifier(topic_dists, df["gold_consensus"], args.downsample, args.seed)


if __name__ == "__main__":
    main()