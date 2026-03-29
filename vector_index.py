import os
import pickle

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


class LSIFaissIndex:
    """
    Vector-space index using:
    1) TF-IDF -> LSI (TruncatedSVD)
    2) FAISS top-k retrieval on LSI vectors
    """

    META_FILENAME = "vector_lsi.meta"
    FAISS_FILENAME = "vector_faiss.index"

    def __init__(self, data_dir="collection", output_dir="index", n_components=128):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_components = n_components

        self.vectorizer = None
        self.svd = None
        self.doc_paths = []
        self.lsi_matrix = None
        self.faiss_index = None

    @property
    def meta_path(self):
        return os.path.join(self.output_dir, self.META_FILENAME)

    @property
    def faiss_path(self):
        return os.path.join(self.output_dir, self.FAISS_FILENAME)

    def has_artifacts(self):
        return os.path.exists(self.meta_path)

    def build(self):
        docs, doc_paths = self._collect_documents()
        if not docs:
            raise ValueError("Tidak ada dokumen yang ditemukan untuk membangun vector index")

        self.vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\b\w+\b")
        tfidf = self.vectorizer.fit_transform(docs)

        max_components = min(tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        if max_components < 1:
            raise ValueError("Koleksi terlalu kecil untuk membangun LSI")

        actual_components = min(self.n_components, max_components)
        self.svd = TruncatedSVD(n_components=actual_components, random_state=42)
        lsi_matrix = self.svd.fit_transform(tfidf)

        self.doc_paths = doc_paths
        self.lsi_matrix = lsi_matrix.astype("float32")
        self._build_faiss_index()

    def save(self):
        if self.vectorizer is None or self.svd is None or self.lsi_matrix is None:
            raise ValueError("Index belum dibangun. Jalankan build() dulu")

        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.meta_path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "svd": self.svd,
                    "doc_paths": self.doc_paths,
                    "lsi_matrix": self.lsi_matrix,
                    "n_components": self.n_components,
                },
                f,
            )

        if self.faiss_index is None:
            raise ValueError("FAISS index tidak tersedia. Pastikan faiss-cpu terinstall")
        faiss.write_index(self.faiss_index, self.faiss_path)

    def load(self):
        with open(self.meta_path, "rb") as f:
            metadata = pickle.load(f)

        self.vectorizer = metadata["vectorizer"]
        self.svd = metadata["svd"]
        self.doc_paths = metadata["doc_paths"]
        self.lsi_matrix = metadata["lsi_matrix"].astype("float32")
        self.n_components = metadata.get("n_components", self.n_components)

        if faiss is None:
            raise ImportError("faiss belum terpasang. Install package faiss-cpu")

        if os.path.exists(self.faiss_path):
            self.faiss_index = faiss.read_index(self.faiss_path)
        else:
            self._build_faiss_index()

    def query_faiss(self, query, k=10):
        query_vec = self._encode_query(query)
        if query_vec is None:
            return []

        if self.faiss_index is None:
            raise ValueError("FAISS index tidak tersedia. Jalankan build() lalu save()")

        q = query_vec.reshape(1, -1).astype("float32")
        distances, indices = self.faiss_index.search(q, k)

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            # Similarity proxy agar konsisten dengan sistem ranking "lebih besar lebih baik"
            results.append((float(-distance), self.doc_paths[idx]))
        return results

    def _encode_query(self, query):
        if self.vectorizer is None or self.svd is None or self.lsi_matrix is None:
            raise ValueError("Index belum dimuat. Jalankan load() atau build() dulu")

        tfidf_query = self.vectorizer.transform([query])
        lsi_query = self.svd.transform(tfidf_query).astype("float32")
        return lsi_query[0] if lsi_query.shape[0] > 0 else None

    def _build_faiss_index(self):
        self.faiss_index = None
        if self.lsi_matrix is None:
            return
        if faiss is None:
            raise ImportError("faiss belum terpasang. Install package faiss-cpu")

        dim = self.lsi_matrix.shape[1]
        # Gunakan exact top-k L2 search (tanpa cosine normalization)
        index = faiss.IndexFlatL2(dim)
        index.add(self.lsi_matrix)
        self.faiss_index = index

    def _collect_documents(self):
        docs = []
        doc_paths = []

        for block in sorted(next(os.walk(self.data_dir))[1]):
            block_path = os.path.join(self.data_dir, block)
            for filename in sorted(next(os.walk(block_path))[2]):
                path = os.path.join(block_path, filename)
                with open(path, "r", encoding="utf8", errors="surrogateescape") as f:
                    docs.append(f.read())
                    doc_paths.append(path)

        return docs, doc_paths
