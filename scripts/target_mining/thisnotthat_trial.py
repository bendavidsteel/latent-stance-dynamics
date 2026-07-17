import polars as pl
import sentence_transformers
import thisnotthat as tnt
import umap

def main():
    doc_df = pl.read_parquet('./data/1month_doc_targets_with_stance.parquet.zstd')

    embedding_model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    model = sentence_transformers.SentenceTransformer(embedding_model_name)
    embeddings = model.encode(doc_df['Document'].to_list())

    umap_model = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine')
    umap_embeddings = umap_model.fit_transform(embeddings)

    basic_plot = tnt.BokehPlotPane(
        umap_embeddings,
        show_legend=False,
    )

    basic_plot.plot()

if __name__ == '__main__':
    main()