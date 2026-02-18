"""OpenAI embedding client for generating text embeddings."""

import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


class EmbeddingClient:
    """Client for generating embeddings using OpenAI's API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-large",
        dimensions: int = 3072,
    ):
        """
        Initialize the embedding client.

        Args:
            api_key: OpenAI API key. If None, loads from environment.
            model: Embedding model to use.
            dimensions: Output embedding dimensions.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY or pass api_key.")

        self.model = model
        self.dimensions = dimensions
        self.client = OpenAI(api_key=self.api_key)

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.02  # 50 requests per second max

    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the embedding.
        """
        self._rate_limit()

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
        )

        return response.data[0].embedding

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per API call (max 2048).
            show_progress: Whether to print progress.

        Returns:
            List of embeddings in the same order as input texts.
        """
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1

            if show_progress:
                print(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} texts)...")

            self._rate_limit()

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
            )

            # Sort by index to maintain order
            batch_embeddings = [None] * len(batch)
            for item in response.data:
                batch_embeddings[item.index] = item.embedding

            all_embeddings.extend(batch_embeddings)

        return all_embeddings


def get_embedding_client(env_path: Optional[Path] = None) -> EmbeddingClient:
    """
    Create an embedding client, loading API key from environment.

    Args:
        env_path: Path to .env file. Defaults to keyholder.env in project root.

    Returns:
        Configured EmbeddingClient instance.
    """
    if env_path:
        load_dotenv(env_path)
    else:
        # Try common locations
        for path in [Path("keyholder.env"), Path("../.env"), Path(".env")]:
            if path.exists():
                load_dotenv(path)
                break

    return EmbeddingClient()
