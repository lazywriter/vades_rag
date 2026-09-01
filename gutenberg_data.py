# -*- coding: utf-8 -*-
"""
gutenberg_data.py

Loads, cleans, and chunks the raw Project Gutenberg book HTML files that
train_vades.py trains the VADES style model on. This is the SAME corpus that
ends up in the FAISS "database" the recommender queries — one pass over the
books produces both the trained model and the retrieval index, per your
requirement that training and storage share the same source data.

Expected on-disk layout (unchanged from the original notebook):
    BOOKS_DIR/
        <author_1>/
            some_book.html
            another_book.html
        <author_2>/
            ...

book_id is built the same way as in the original vades_main.py:
    book_id = f"{author}___{file_name}"
"""

import os
import re

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def extract_text_from_html(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
        for tag in soup(["table", "script", "style"]):
            tag.extract()
        return soup.get_text(separator=" ", strip=True)


def clean_gutenberg_text(text: str) -> str:
    text = re.sub(r'(?is).*?(the project gutenberg ebook of )', '', text, count=1)
    text = re.sub(r'(?is)end of (the )?project gutenberg ebook.*', '', text)
    text = re.sub(r'(?is)\*\*\*.*?project gutenberg.*?\*\*\*', '', text)
    text = re.sub(r'(?i)(produced by|transcribed from).*?\n', '', text)
    text = re.sub(r'\b(CHAPTER|Chapter)\s+[A-Z0-9IVX]+\b', '', text)
    text = re.sub(r'[_*#~-]{2,}', ' ', text)
    text = re.sub(r'http\S+', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def chunk_text(text: str, size: int = 4000, overlap: int = 500):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def load_or_build_chunks_df(books_dir: str, output_csv: str) -> pd.DataFrame:
    """
    Walks books_dir/<author>/*.html, cleans + chunks each book, and returns a
    DataFrame with columns: author, file_name, text, book_id.
    Cached to output_csv after the first build (delete the file to rebuild).
    """
    if os.path.exists(output_csv):
        print(f"Loading cached chunk dataset from {output_csv}")
        df = pd.read_csv(output_csv)
        if "book_id" not in df.columns:
            df["book_id"] = df["author"] + "___" + df["file_name"]
        return df

    if not os.path.isdir(books_dir):
        raise FileNotFoundError(
            f"books_dir '{books_dir}' does not exist. Expected <books_dir>/<author>/*.html"
        )

    data = []
    authors = [a for a in os.listdir(books_dir) if os.path.isdir(os.path.join(books_dir, a))]
    if not authors:
        raise ValueError(
            f"No author subfolders found under '{books_dir}'. Expected one folder per "
            f"author, each containing that author's .html book files."
        )

    for author in tqdm(authors, desc="Authors"):
        folder = os.path.join(books_dir, author)
        for fname in os.listdir(folder):
            if not fname.endswith(".html"):
                continue
            try:
                raw = extract_text_from_html(os.path.join(folder, fname))
                clean = clean_gutenberg_text(raw)
                for chunk in chunk_text(clean):
                    if chunk.strip():
                        data.append({"author": author, "file_name": fname, "text": chunk})
            except Exception as e:
                print(f"  skipped {author}/{fname}: {e}")

    if not data:
        raise ValueError(f"No usable text extracted from any .html file under '{books_dir}'.")

    df = pd.DataFrame(data)
    df["book_id"] = df["author"] + "___" + df["file_name"]
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Built {len(df)} chunks from {df['book_id'].nunique()} books, "
          f"{df['author'].nunique()} authors")
    return df


def book_level_split(df: pd.DataFrame, split_dir: str, test_size: float = 0.3,
                      val_fraction_of_temp: float = 0.5, random_state: int = 42):
    """
    Splits by BOOK (not chunk) so no book's chunks appear in more than one
    split — a chunk-level split would leak the same book's style signature
    across train/val/test and inflate every downstream metric.
    """
    train_csv, val_csv, test_csv = [os.path.join(split_dir, f"{s}.csv") for s in ["train", "val", "test"]]

    if all(os.path.exists(f) for f in [train_csv, val_csv, test_csv]):
        print(f"Loading cached splits from {split_dir}")
        return (pd.read_csv(train_csv), pd.read_csv(val_csv), pd.read_csv(test_csv))

    books = df[["author", "book_id"]].drop_duplicates()
    counts = books["author"].value_counts()
    # Stratifying twice (train/temp, then val/test within temp) needs each
    # author's book count to survive both splits with >=2 books left for the
    # second stratification. Derived from the actual ratios rather than a
    # fixed constant, since it depends on test_size.
    min_required = int(-(-2 // test_size))  # ceil(2 / test_size)
    if (counts < min_required).any():
        thin = counts[counts < min_required].to_dict()
        raise ValueError(
            f"book_level_split stratifies by author twice (train/temp, then val/test "
            f"within temp), which needs at least {min_required} books per author at "
            f"test_size={test_size}. These authors have too few books: {thin}. Either "
            f"add more books for them, drop them, or raise test_size / lower the "
            f"per-author minimum by accepting a non-stratified split instead."
        )

    train_books, temp = train_test_split(books, test_size=test_size,
                                          stratify=books["author"], random_state=random_state)
    val_books, test_books = train_test_split(temp, test_size=val_fraction_of_temp,
                                              stratify=temp["author"], random_state=random_state)

    train_df = df[df["book_id"].isin(train_books["book_id"])].reset_index(drop=True)
    val_df = df[df["book_id"].isin(val_books["book_id"])].reset_index(drop=True)
    test_df = df[df["book_id"].isin(test_books["book_id"])].reset_index(drop=True)

    assert len(set(train_df["book_id"]) & set(test_df["book_id"])) == 0
    assert len(set(train_df["book_id"]) & set(val_df["book_id"])) == 0
    assert len(set(val_df["book_id"]) & set(test_df["book_id"])) == 0

    os.makedirs(split_dir, exist_ok=True)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)} chunks")
    return train_df, val_df, test_df
