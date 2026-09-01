# Finding the slot count that fits the confidence-filtered signal

**Status: research exercise, not a validated trading strategy, and NOT wired into decide_trades.py or the portal.** `ml/BACKTEST_PIT_LEVER_ISOLATION_REPORT.md` confirmed confidence filtering improves results on its own, but tested it only at v1's original 15-slot limit -- a number chosen before filtering existed and never revisited. Since filtering to the model's top 50% most confident BUY calls roughly halves the signal supply, this sweeps a range of slot counts, always with the confidence filter on, to find the count that actually fits the thinned-out signal flow.

## Results

| Max positions | Ending value | CAGR | Trades | Median invested | Max drawdown |
|---|---|---|---|---|---|
| 10 | $617,412 | +7.08% | 4,978 | 88.2% | -76.9% |
| 15 | $588,236 | +6.88% | 6,327 | 65.4% | -77.0% |
| 20 | $849,954 | +8.37% **<- best** | 7,235 | 49.5% | -66.3% |
| 25 | $817,392 | +8.21% | 7,821 | 39.7% | -60.9% |
| 30 | $668,753 | +7.40% | 8,209 | 33.1% | -57.3% |
| 35 | $564,790 | +6.72% | 8,448 | 28.4% | -55.6% |
| 40 | $537,880 | +6.52% | 8,617 | 24.9% | -49.3% |
| 50 | $422,908 | +5.57% | 8,805 | 19.9% | -43.1% |
| SPY buy-and-hold | $847,851 | +8.36% | - | - | - |

The best slot count tested is **20**, reaching CAGR +8.37% (ending value $849,954) -- this **essentially ties** SPY buy-and-hold (+8.36%), a margin of only +0.01 percentage points. That margin is small enough that it should be read as "finally in the same neighborhood as buy-and-hold" rather than "a proven edge over it" -- see the in-sample-selection caveat below before treating 20 slots as the answer. 
Compared to the 15-slot count used in every earlier confidence-filtered test, this is +1.49 percentage points of CAGR different.

*(A chart of CAGR and capital utilization across all slot counts tested is generated locally as `ml/backtest_pit_slot_sweep_chart.svg` when you run `backtest_pit_slot_sweep.py` -- not committed to the repo, same build-artifact policy as the rest of `ml/`.)*

## Honest caveats -- read before drawing conclusions

- **This slot count was chosen by looking at results on this exact backtest, then reported as "best."** That's a form of in-sample selection -- picking the winner after seeing all the outcomes on the same 27 years of data being reported on. A slot count "optimized" this way is likely to look somewhat better here than it would on genuinely new, unseen years; treat the improvement over 15 slots as an optimistic upper bound, not a guaranteed gain.
- **Only 8 slot counts were tried, on a coarse grid** (10, 15, 20, 25, 30, 35, 40, 50) -- the true optimum could sit between two tested values.
- **The 50% confidence cutoff itself was never re-swept alongside slot count.** A different cutoff paired with a different slot count could do better still; this only searches one dimension at a time.
- **Same caveats as `ml/BACKTEST_PIT_REPORT.md`**: no interim mark-to-market, SELL signals don't trigger early exits, no fractional-share rounding, same survivorship bias, approximate exit dates for multi-stint tickers, transaction-cost assumption unchanged, not integrated into `decide_trades.py` or the portal.
