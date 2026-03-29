import argparse

from vector_index import LSIFaissIndex


def main():
    parser = argparse.ArgumentParser(description="Build vector index (LSI + FAISS)")
    parser.add_argument("--data-dir", default="collection")
    parser.add_argument("--output-dir", default="index")
    parser.add_argument("--components", type=int, default=128)
    args = parser.parse_args()

    indexer = LSIFaissIndex(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_components=args.components,
    )

    print("Building vector index...")
    indexer.build()
    indexer.save()

    print("Vector index built successfully")
    print(f"  - Documents   : {len(indexer.doc_paths)}")
    print(f"  - Components  : {indexer.svd.n_components}")
    print(f"  - Meta file   : {indexer.meta_path}")
    print(f"  - FAISS file  : {indexer.faiss_path}")


if __name__ == "__main__":
    main()
