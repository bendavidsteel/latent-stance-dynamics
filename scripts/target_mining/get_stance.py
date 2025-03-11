import polars as pl

import stancemining

def main():
    period = '1year'
    document_df = pl.read_parquet(f'./data/stance_targets/{period}_doc_targets.parquet.zstd')

    finetune_kwargs = {
        'model_name': 'HuggingFaceTB/SmolLM2-135M-Instruct',
        'add_system_message': True,
        'save_model_path': '../stancemining/models/stancemining',
        'prompting_method': 'stancemining',
        'classification_method': 'generation',
        'generation_method': 'list',
        'batch_size': 200
    }

    model_kwargs = {
            'device_map': {'': 1},
        'torch_dtype': 'auto',
        'attn_implementation': 'flash_attention_2'
    }

    miner = stancemining.StanceMining(finetune_kwargs=finetune_kwargs, model_kwargs=model_kwargs, verbose=True)
    document_df, _ = miner.get_stance(document_df)
    document_df.write_parquet(f'./data/stance_targets/{period}_doc_targets_with_stance.parquet.zstd')

if __name__ == "__main__":
    main()
