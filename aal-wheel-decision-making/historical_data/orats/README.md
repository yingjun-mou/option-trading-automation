# historical_data/orats/

Drop real ORATS EOD **strikes** files here (`.csv` or `.parquet`, one big file or
one per day -- all matching files are concatenated).

As soon as any file is present, `default_sources()` uses this folder instead of
the synthetic `historical_data/mock/` chain, and the report header changes to
`OratsFolder`.

Schema: see `../README.md`. Files whose names start with `_` are ignored
(`_normalized.parquet` is a generated cache).
