# realtime_data/robinhood_api/

TODO -- not implemented yet.

Will hold Robinhood API credentials / token cache. When present,
`src/realtime.py::default_realtime_source()` should return a `RobinhoodRealtime`
that implements the same `RealtimeSource` interface (`quote()` returning a
`Quote`, `option_chain()` returning a DataFrame with columns
`date, expiration, dte, strike, opt_type, bid, ask, mid, iv, delta, underlying`).

No other code changes are needed -- the advisor and dashboard only see the
interface.
