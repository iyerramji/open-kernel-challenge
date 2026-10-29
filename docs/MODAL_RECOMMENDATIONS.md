# Modal recommendation results

Diagnostic raw graph timings; not official scores. V12 remains selected.



## modal-resume1

v12_wide_native-timings: diagnostic load aggregate 1.087982x; M8 unpooler 11.968 us (baseline 11.968 us).

v24_tiny_unpooler-timings: diagnostic load aggregate 1.092228x; M8 unpooler 10.496 us (baseline 12.192 us).

v25_bmm_explicit-timings: diagnostic load aggregate 1.066986x; M8 unpooler 12.096 us (baseline 12.016 us).

## modal-repeat1

v12_wide_native-timings: diagnostic load aggregate 1.086477x; M8 unpooler 11.968 us (baseline 11.920 us).

v24_tiny_unpooler-timings: diagnostic load aggregate 1.091299x; M8 unpooler 10.224 us (baseline 11.872 us).



All three candidates passed 74 full-dimension cases and 30 fresh-input graph replays. BMM profiling/copy investigation was not run; do not claim copy-free dispatch. No official checks or submissions used. All Modal apps completed and container list was empty.

TinyGEMM small-unpooler gain repeated; preserve v24 as the next official-check candidate. BMM universal split table is not recommended. Combination with older M1/QKV variants deferred because no new robust evidence supports those paths.
