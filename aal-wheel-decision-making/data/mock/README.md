# data/mock/

Synthetic option chain, written here on first run as
`ORATS_strikes_AAL_synthetic.csv` in the exact ORATS EOD strikes schema.

It is a Black-Scholes chain priced off the real AAL close, with a modelled
mean-reverting IV surface (skew + smile). It exists so the pipeline runs before
real data arrives. Delete the file to regenerate.

Used only when `data/orats/` is empty. IV-timing conclusions from this data are
model-dependent; everything else (price / DTE / delta / laddering) is structural.
