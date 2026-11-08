# Verification summary

- `results/verify-top200-53163452.json`: seed 53163452 (os.urandom at launch), weights sha256 c9fdfea3a2c35c50..., arch sha256 27d89627a0771ac3..., 71/71 jobs

| kind | n | m | T | halt pass | kmax | problems | fp32 wrong | fp32 halt != | fp32 fixed-pt wrong | fp32 min margin (halt / to kmax) | bf16 wrong | bf16 halt != | bf16 fixed-pt wrong | bf16 min margin (halt / to kmax) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rand | 200 | 200 | 803 | 1002 | 1607 | 852 | 0 | 0 | 0 | 0.999981 / 0.999981 | 0 | 0 | 0 | 0.999981 / 0.999981 |

Total: 852 problems per weight set. fp32: strict wrong 0, argmax wrong 0, halted at another pass 0, never halted 0, fixed-point wrong 0, min margin at halt 0.999981, min margin halt..kmax 0.999981; bf16: strict wrong 0, argmax wrong 0, halted at another pass 0, never halted 0, fixed-point wrong 0, min margin at halt 0.999981, min margin halt..kmax 0.999981.
