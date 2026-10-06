import duckdb


# convert X_train_N1UvY30.csv into Xtrain.parquet
duckdb.sql("""
COPY (
    SELECT *
    FROM read_csv_auto('../X_train_N1UvY30.csv')
)
TO 'Xtrain.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD);
""")


# convert y_train_or6m3Ta.csv into ytrain.parquet
duckdb.sql("""
COPY (
    SELECT *
    FROM read_csv_auto('../y_train_or6m3Ta.csv')
)
TO 'ytrain.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD);
""")


# convert X_test_m4HAPAP.csv into Xtrain.parquet
duckdb.sql("""
COPY (
    SELECT *
    FROM read_csv_auto('../X_test_m4HAPAP.csv')
)
TO 'Xtest.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD);
""")
