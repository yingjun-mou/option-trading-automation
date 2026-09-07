# realtime_data/

Live market data for the dashboard advisor. Same idea as `historical_data/`: one
interface, swappable backends.

| path | what | status |
| --- | --- | --- |
| `mock/` | fake feed derived from the last cached historical close (price wanders a few percent so the dashboard ticks) + a Black-Scholes option chain generated on the fly | active |
| `robinhood_api/` | real AAL quote + option chain via the Robinhood API | TODO |

`src/realtime.py` defines `RealtimeSource` (`quote()` + `option_chain()`).
`default_realtime_source()` returns `MockRealtime` today; it will return a
`RobinhoodRealtime` once `robinhood_api/` holds credentials.

## mock/portfolio.json

Your current holdings, read by the advisor to decide whether to recommend
selling, closing, or rolling. Copy `mock/portfolio.example.json` to
`mock/portfolio.json` and edit. If absent, the advisor assumes all cash and no
positions.
