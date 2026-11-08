# Verification summary

- `results/verify-drift2-451532730.json`: seed 451532730 (os.urandom at launch), weights sha256 c9fdfea3a2c35c50..., arch sha256 27d89627a0771ac3..., 9/9 jobs

| kind | n | m | T | halt pass | kmax | problems | fp32 wrong | fp32 halt != | fp32 fixed-pt wrong | fp32 min margin (halt / to kmax) | bf16 wrong | bf16 halt != | bf16 fixed-pt wrong | bf16 min margin (halt / to kmax) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rand | 3 | 3 | 15 | 17 | 60 | 8 | 0 | 0 | 0 | 0.999989 / 0.999985 | 0 | 0 | 0 | 0.999989 / 0.999985 |
| rand | 10 | 10 | 43 | 52 | 172 | 8 | 0 | 0 | 0 | 0.999992 / 0.999985 | 0 | 0 | 0 | 0.999992 / 0.999985 |
| rand | 30 | 30 | 123 | 152 | 492 | 8 | 0 | 0 | 0 | 0.999985 / 0.999981 | 0 | 0 | 0 | 0.999985 / 0.999981 |
| rand | 100 | 100 | 403 | 502 | 1612 | 8 | 0 | 0 | 0 | 0.999981 / 0.999981 | 0 | 0 | 0 | 0.999981 / 0.999981 |
| rand | 200 | 200 | 803 | 1002 | 3212 | 8 | 0 | 0 | 0 | 0.999985 / 0.999981 | 0 | 0 | 0 | 0.999985 / 0.999981 |

Total: 40 problems per weight set. fp32: strict wrong 0, argmax wrong 0, halted at another pass 0, never halted 0, fixed-point wrong 0, min margin at halt 0.999981, min margin halt..kmax 0.999981; bf16: strict wrong 0, argmax wrong 0, halted at another pass 0, never halted 0, fixed-point wrong 0, min margin at halt 0.999981, min margin halt..kmax 0.999981.
float64: 3x3: wrong 0, min 1.000000000, 10x10: wrong 0, min 1.000000000, 30x30: wrong 0, min 1.000000000, 100x100: wrong 0, min 0.999999999, 200x200: wrong 0, min 0.999999999
