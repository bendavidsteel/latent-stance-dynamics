import polars as pl

def main():
    df = pl.DataFrame({
        'val': [None, 0, 1, None, None, None],
        'name': ['a', 'a', 'a', 'a', 'b', 'b']
    })
    df = df.with_columns(
        pl.col('val').forward_fill().over('name')
    )
    df = df.with_columns(
        pl.col('val').backward_fill().over('name')
    )
    df = df.with_columns(
        pl.col('val').fill_null(0).over('name')
    )
    pass

if __name__ == '__main__':
    main()