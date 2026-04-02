# Real Benchmark Data

This directory documents the real-data benchmark workflow.

The benchmark script [`real_data_bench.py`](/home/tret/Code/allcode/rusting_away/uni_diploma/interlib/python/benches/real_data_bench.py)
downloads official public datasets and stores local snapshots under
`python/benches/data_cache/`.

The cache directory is intentionally ignored by git because benchmark timing
should use stable local snapshots, not live API calls inside the timed region.

Current sources:

- NOAA / National Weather Service observations API
  - Documentation: <https://www.weather.gov/documentation/services-web-api>
- NASA / JPL Horizons API
  - Documentation: <https://ssd-api.jpl.nasa.gov/doc/horizons.html>

Example commands:

```bash
python python/benches/real_data_bench.py --dataset noaa --station KSFO --field temperature
python python/benches/real_data_bench.py --dataset nasa --command 499 --axis x
```
