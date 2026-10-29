# Official scored pass

6 candidates scored; all passed. Best: v21_safe_bundle.py at1.158264x, CI[1.140890,1.175638]. Normalized4170.724us. Snapshot rank1039/1354. Remaining scored attempts:4; unscored:0. No fabricated repeat: returned status/duplicate metadata are retained.

| Candidate | Speedup | Interval | Time us | Job |
|---|---:|---|---:|---|
|v21_safe_bundle.py|1.158264x|[1.1408901057139054, 1.175638027715344]|4170.724|[433a8be9f5c14d37](../results/runs/433a8be9f5c14d37/result.json)|
|v34_official_bundle.py|1.156021x|[1.138681026279216, 1.1733616666735067]|4178.816|[764ac132f5224473](../results/runs/764ac132f5224473/result.json)|
|v24_tiny_unpooler.py|1.144293x|[1.1271290622290626, 1.1614578661548207]|4221.644|[bc65b67e2cf64705](../results/runs/bc65b67e2cf64705/result.json)|
|v33_qkv_lt_checked.py|1.141305x|[1.1241854438039678, 1.1584245943766773]|4232.698|[f494abe5f9f94394](../results/runs/f494abe5f9f94394/result.json)|
|v12_wide_native.py|1.139599x|[1.1225053093350128, 1.1566932882995309]|4239.034|[e93e60835ffb4770](../results/runs/e93e60835ffb4770/result.json)|
|v20_router_group4.py|1.132450x|[1.1154636831501603, 1.149437196342551]|4265.794|[14da83f6a21449d5](../results/runs/14da83f6a21449d5/result.json)|

Modal and official timing are separate. The official scorer subtracts replay overhead and uses its own paired estimator; Modal raw timings do not. The combined candidate was built after comparing official per-shape results, then validated locally on H200 before scored submission. Intervals overlap, so highest point estimate is selected provisionally, not claimed statistically superior.
