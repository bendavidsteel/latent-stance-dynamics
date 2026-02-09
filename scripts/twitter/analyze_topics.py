import polars as pl

def main():
    df = pl.read_parquet('./data/twitter/topic_info.parquet')

    pass

if __name__ == '__main__':
    main()