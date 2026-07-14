"""Tests for the frozen sentence-embedding baseline."""

import numpy as np

from src.models.sentence_embedding_baseline import (
    build_classifier,
    encode_texts,
)


class FakeEncoder:
    """Deterministic encoder used without downloading a model."""

    def encode(
        self,
        texts,
        batch_size,
        show_progress_bar,
        convert_to_numpy,
        normalize_embeddings,
    ):
        """Create small vectors from text properties."""
        vectors = []

        for text in texts:
            length = float(len(text))
            word_count = float(len(text.split()))

            vector = np.array(
                [
                    length,
                    word_count,
                    length + word_count,
                ],
                dtype=np.float32,
            )

            norm = np.linalg.norm(vector)

            if normalize_embeddings and norm > 0:
                vector = vector / norm

            vectors.append(vector)

        return np.vstack(vectors)


def test_encode_texts_returns_expected_shape():
    encoder = FakeEncoder()

    embeddings = encode_texts(
        encoder,
        ["hello", "a longer sentence"],
        batch_size=2,
    )

    assert embeddings.shape == (2, 3)
    assert embeddings.dtype == np.float32


def test_embeddings_are_normalized():
    encoder = FakeEncoder()

    embeddings = encode_texts(
        encoder,
        ["hello world"],
        batch_size=1,
    )

    assert np.isclose(
        np.linalg.norm(embeddings[0]),
        1.0,
    )


def test_classifier_can_fit_embedding_vectors():
    features = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )

    labels = [
        "joy",
        "joy",
        "sadness",
        "sadness",
    ]

    classifier = build_classifier()
    classifier.fit(features, labels)

    predictions = classifier.predict(features)

    assert len(predictions) == 4
    assert set(predictions).issubset(
        {"joy", "sadness"}
    )
