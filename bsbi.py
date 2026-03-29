import os
import pickle
import contextlib
import heapq
import argparse

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, PatriciaTermDict, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings, EliasGammaPostings
from retrieval import retrieve_tfidf_taat, retrieve_bm25_taat, retrieve_wand
from tqdm import tqdm

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index", index_mode="bsbi", term_dict_mode="idmap"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        if index_mode not in ("bsbi", "spimi"):
            raise ValueError("index_mode harus 'bsbi' atau 'spimi'")
        if term_dict_mode not in ("idmap", "patricia"):
            raise ValueError("term_dict_mode harus 'idmap' atau 'patricia'")
        self.index_mode = index_mode
        self.term_dict_mode = term_dict_mode
        self.term_patricia = None

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

        if self.term_dict_mode == "patricia":
            patricia = PatriciaTermDict()
            patricia.build_from_terms(self.term_id_map.id_to_str)
            patricia_path = os.path.join(self.output_dir, 'terms.patricia')
            patricia.save(patricia_path)
            self.term_patricia = patricia

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

        self.term_patricia = None
        if self.term_dict_mode == "patricia":
            patricia_path = os.path.join(self.output_dir, 'terms.patricia')
            if os.path.exists(patricia_path):
                self.term_patricia = PatriciaTermDict.load(patricia_path)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding = "utf8", errors = "surrogateescape") as f:
                for token in f.read().split():
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname]))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def spimi_invert_write_block(self, block_dir_relative, index):
        """
        SPIMI (Single Pass In-Memory Indexing): inversion langsung saat parsing block
        tanpa membentuk td_pairs global terlebih dahulu.

        Alur:
        1. Scan dokumen di block, update hashtable term_postings_tf (map term_id -> {doc_id -> tf})
        2. Setelah semua dokumen di block selesai, sort term IDs dan write ke index

        Parameters
        ----------
        block_dir_relative: str
            Relative path ke direktori block (misal '1' untuk collection/1/)
        index: InvertedIndexWriter
            Writer untuk menyimpan intermediate index file block ini
        """
        block_path = os.path.join('.', self.data_dir, block_dir_relative)
        term_postings_tf = {}

        for filename in next(os.walk(block_path))[2]:
            docname = os.path.join(block_path, filename)
            doc_id = self.doc_id_map[docname]
            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                for token in f.read().split():
                    term_id = self.term_id_map[token]
                    if term_id not in term_postings_tf:
                        term_postings_tf[term_id] = {}
                    postings_tf = term_postings_tf[term_id]
                    postings_tf[doc_id] = postings_tf.get(doc_id, 0) + 1

        for term_id in sorted(term_postings_tf.keys()):
            postings_tf = term_postings_tf[term_id]
            sorted_doc_id = sorted(postings_tf.keys())
            assoc_tf = [postings_tf[doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        term_ids = self._get_existing_query_term_ids(query)
        if not term_ids:
            return []

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            return retrieve_tfidf_taat(term_ids, merged_index, self.doc_id_map, k=k)

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Ranked retrieval BM25 dengan skema TaaT.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi
        k: int
            Banyak dokumen yang dikembalikan
        k1: float
            Parameter BM25 untuk kontrol pengaruh TF
        b: float
            Parameter BM25 untuk normalisasi panjang dokumen
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        term_ids = self._get_existing_query_term_ids(query)
        if not term_ids:
            return []

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            return retrieve_bm25_taat(term_ids, merged_index, self.doc_id_map, k=k, k1=k1, b=b)

    def retrieve_wand(self, query, k=10, scoring='bm25', k1=1.2, b=0.75):
        """
        WAND top-k retrieval untuk mode TF-IDF atau BM25.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi
        k: int
            Banyak dokumen yang dikembalikan
        scoring: str
            Mode scoring: 'tfidf' atau 'bm25'
        k1: float
            Parameter BM25 (dipakai saat scoring='bm25')
        b: float
            Parameter BM25 (dipakai saat scoring='bm25')
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        term_ids = self._get_existing_query_term_ids(query)
        if not term_ids:
            return []

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            return retrieve_wand(term_ids, merged_index, self.doc_id_map, k=k, scoring=scoring, k1=k1, b=b)

    def _get_existing_query_term_ids(self, query):
        """Mengembalikan daftar termID query yang memang ada di koleksi."""
        terms = []
        for word in query.split():
            if self.term_patricia is not None:
                term_id = self.term_patricia.lookup(word)
                if term_id is not None:
                    terms.append(term_id)
            elif word in self.term_id_map.str_to_id:
                terms.append(self.term_id_map.str_to_id[word])
        return terms

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI / SPIMI

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        self.intermediate_indices = []
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                if self.index_mode == "spimi":
                    self.spimi_invert_write_block(block_dir_relative, index)
                else:
                    td_pairs = self.parse_block(block_dir_relative)
                    self.invert_write(td_pairs, index)
                    td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Bangun indeks dengan BSBI")
    parser.add_argument("--compression", choices=["standard", "vbe", "elias-gamma"], default="vbe")
    parser.add_argument("--index-mode", choices=["bsbi", "spimi"], default="bsbi")
    parser.add_argument("--term-dict", choices=["idmap", "patricia"], default="idmap")
    parser.add_argument("--data-dir", default="collection")
    parser.add_argument("--output-dir", default="index")
    args = parser.parse_args()

    encoding_map = {
        "standard": StandardPostings,
        "vbe": VBEPostings,
        "elias-gamma": EliasGammaPostings,
    }

    BSBI_instance = BSBIIndex(data_dir=args.data_dir, \
                              postings_encoding=encoding_map[args.compression], \
                              output_dir=args.output_dir,
                              index_mode=args.index_mode,
                              term_dict_mode=args.term_dict)
    BSBI_instance.index() # memulai indexing!
