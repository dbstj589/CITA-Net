# realistic-v1 — conditional validity of all five constraint terms, n=5

Data `data/realistic_v1` (161 true identities/sector, 3.99M triples), decoder `num_slots=200`, blocking `dt_max_s=450 / r_err_floor_m=300`, epochs=40, `--full-sectors`, decode_full scoring. Seeds: 19521006, 19521007, 19521008, 19521009, 19521010.

**n=5 → low statistical power. The paired t-test (df=4) is reported for completeness; interpretation leans on per-seed sign consistency (SC) and dev/test agreement, not p alone. Holm-adjusted p is a reference column; the pre-registered verdict uses the uncorrected p together with SC.**

Convention: Δ = variant − m3_full. **Negative ΔF1 ⇒ removing the term HURT ⇒ the term helps.** Where dev and test disagree in sign, the result is marked NOT TRUSTWORTHY.

Analyses reported: **표준 디코드 (사전 등록 주 분석), revive margin 0.7 (2차 분석)**

## 1. [표준 디코드 (사전 등록 주 분석)] 전체 지표 패널 (mean±std, 분모 병기)

### dev

| model | precision | recall | f1 | wrong_merge_rate | fragmentation_rate | impossible_transition_rate | trajectory_consistency_rate | dangling_precision | dangling_recall |
|---|---|---|---|---|---|---|---|---|---|
| m3_full | 0.7013±0.0478 | 0.1658±0.0497 | 0.2644±0.0684 | 0.1877±0.0308 | 0.9859±0.0156 | 0.2664±0.0285 | 0.7336±0.0285 | 0.2595±0.0295 | 0.7680±0.0717 |
| no_motion | 0.7655±0.0303 | 0.1264±0.0692 | 0.2086±0.1003 | 0.1416±0.0260 | 0.9744±0.0113 | 0.2345±0.0186 | 0.7655±0.0186 | 0.2341±0.0335 | 0.8104±0.0875 |
| no_time | 0.6960±0.0344 | 0.1344±0.0595 | 0.2189±0.0913 | 0.1745±0.0612 | 0.9812±0.0231 | 0.2444±0.0274 | 0.7556±0.0274 | 0.2417±0.0253 | 0.8172±0.0783 |
| no_state | 0.6989±0.0379 | 0.1365±0.0478 | 0.2238±0.0676 | 0.1902±0.0363 | 0.9932±0.0346 | 0.2531±0.0264 | 0.7469±0.0264 | 0.2400±0.0160 | 0.7883±0.0714 |
| no_rel | 0.7124±0.0671 | 0.1005±0.0573 | 0.1689±0.0845 | 0.1755±0.0458 | 0.9821±0.0168 | 0.2465±0.0447 | 0.7535±0.0447 | 0.2240±0.0291 | 0.8570±0.0742 |
| no_src | 0.7051±0.0589 | 0.1363±0.0688 | 0.2199±0.0997 | 0.1753±0.0364 | 0.9906±0.0343 | 0.2473±0.0197 | 0.7527±0.0197 | 0.2448±0.0409 | 0.8055±0.0934 |
| *분모(평균)* | 25.8 | 109.9 | — | 72.3 | 109.9 | 1093.3 | 1093.3 | 154.3 | 51.1 |   ← m3_full 기준

### test

| model | precision | recall | f1 | wrong_merge_rate | fragmentation_rate | impossible_transition_rate | trajectory_consistency_rate | dangling_precision | dangling_recall |
|---|---|---|---|---|---|---|---|---|---|
| m3_full | 0.6917±0.0381 | 0.1596±0.0429 | 0.2559±0.0589 | 0.1795±0.0376 | 0.9786±0.0288 | 0.2559±0.0241 | 0.7441±0.0241 | 0.2556±0.0259 | 0.7592±0.0615 |
| no_motion | 0.7510±0.0391 | 0.1206±0.0728 | 0.1975±0.1042 | 0.1486±0.0327 | 0.9804±0.0073 | 0.2320±0.0211 | 0.7680±0.0211 | 0.2402±0.0347 | 0.8226±0.0927 |
| no_time | 0.6940±0.0598 | 0.1293±0.0592 | 0.2125±0.0928 | 0.1600±0.0195 | 0.9687±0.0164 | 0.2409±0.0180 | 0.7591±0.0180 | 0.2359±0.0186 | 0.8079±0.0889 |
| no_state | 0.6939±0.0567 | 0.1294±0.0403 | 0.2140±0.0598 | 0.1723±0.0384 | 0.9872±0.0222 | 0.2431±0.0296 | 0.7569±0.0296 | 0.2385±0.0120 | 0.7889±0.0551 |
| no_rel | 0.7527±0.0613 | 0.1067±0.0556 | 0.1798±0.0811 | 0.1488±0.0485 | 0.9670±0.0114 | 0.2379±0.0379 | 0.7621±0.0379 | 0.2255±0.0237 | 0.8466±0.0762 |
| no_src | 0.7224±0.0504 | 0.1358±0.0645 | 0.2205±0.0937 | 0.1590±0.0363 | 0.9813±0.0161 | 0.2400±0.0151 | 0.7600±0.0151 | 0.2404±0.0305 | 0.7992±0.1005 |
| *분모(평균)* | 24.9 | 108.8 | — | 69.0 | 108.8 | 1044.7 | 1044.7 | 158.5 | 52.2 |   ← m3_full 기준

## 2. [revive margin 0.7 (2차 분석)] 전체 지표 패널 (mean±std, 분모 병기)

### dev

| model | precision | recall | f1 | wrong_merge_rate | fragmentation_rate | impossible_transition_rate | trajectory_consistency_rate | dangling_precision | dangling_recall |
|---|---|---|---|---|---|---|---|---|---|
| m3_full | 0.6471±0.0426 | 0.3404±0.0412 | 0.4451±0.0442 | 0.2713±0.0269 | 1.1058±0.0317 | 0.2921±0.0188 | 0.7079±0.0188 | 0.7411±0.1584 | 0.5248±0.0291 |
| no_motion | 0.6530±0.0315 | 0.3198±0.0712 | 0.4255±0.0731 | 0.2456±0.0130 | 1.0752±0.0144 | 0.2804±0.0056 | 0.7196±0.0056 | 0.6294±0.2486 | 0.5515±0.0554 |
| no_time | 0.6555±0.0423 | 0.3207±0.0808 | 0.4242±0.0837 | 0.2544±0.0330 | 1.0646±0.0568 | 0.2825±0.0097 | 0.7175±0.0097 | 0.6475±0.2354 | 0.5570±0.0923 |
| no_state | 0.6451±0.0313 | 0.3372±0.0436 | 0.4415±0.0445 | 0.2624±0.0233 | 1.0889±0.0395 | 0.2850±0.0143 | 0.7150±0.0143 | 0.6578±0.1305 | 0.5373±0.0265 |
| no_rel | 0.6425±0.0300 | 0.2790±0.0645 | 0.3837±0.0623 | 0.2473±0.0322 | 1.0692±0.0524 | 0.2828±0.0154 | 0.7172±0.0154 | 0.5039±0.2292 | 0.5924±0.0772 |
| no_src | 0.6467±0.0404 | 0.3232±0.0599 | 0.4277±0.0604 | 0.2562±0.0232 | 1.0771±0.0417 | 0.2894±0.0166 | 0.7106±0.0166 | 0.6570±0.2419 | 0.5457±0.0566 |
| *분모(평균)* | 57.7 | 109.9 | — | 126.8 | 109.9 | 2199.1 | 2199.1 | 38.3 | 51.1 |   ← m3_full 기준

### test

| model | precision | recall | f1 | wrong_merge_rate | fragmentation_rate | impossible_transition_rate | trajectory_consistency_rate | dangling_precision | dangling_recall |
|---|---|---|---|---|---|---|---|---|---|
| m3_full | 0.6370±0.0417 | 0.3451±0.0466 | 0.4462±0.0481 | 0.2688±0.0276 | 1.0838±0.0583 | 0.2810±0.0144 | 0.7190±0.0144 | 0.7167±0.1481 | 0.5145±0.0246 |
| no_motion | 0.6582±0.0200 | 0.3140±0.0685 | 0.4211±0.0675 | 0.2438±0.0163 | 1.0710±0.0238 | 0.2691±0.0056 | 0.7309±0.0056 | 0.6282±0.2644 | 0.5541±0.0615 |
| no_time | 0.6617±0.0240 | 0.3223±0.0829 | 0.4277±0.0858 | 0.2452±0.0169 | 1.0600±0.0190 | 0.2723±0.0146 | 0.7277±0.0146 | 0.6240±0.2238 | 0.5493±0.0828 |
| no_state | 0.6431±0.0382 | 0.3283±0.0450 | 0.4327±0.0481 | 0.2589±0.0266 | 1.0880±0.0318 | 0.2741±0.0159 | 0.7259±0.0159 | 0.6348±0.1316 | 0.5268±0.0206 |
| no_rel | 0.6506±0.0426 | 0.2879±0.0793 | 0.3928±0.0789 | 0.2389±0.0346 | 1.0475±0.0443 | 0.2730±0.0148 | 0.7270±0.0148 | 0.5154±0.2430 | 0.5850±0.0783 |
| no_src | 0.6696±0.0261 | 0.3291±0.0644 | 0.4374±0.0631 | 0.2540±0.0183 | 1.0663±0.0456 | 0.2772±0.0132 | 0.7228±0.0132 | 0.6308±0.2242 | 0.5400±0.0565 |
| *분모(평균)* | 58.6 | 108.8 | — | 124.3 | 108.8 | 2173.6 | 2173.6 | 40.1 | 52.2 |   ← m3_full 기준

## 3. (a) ΔF1 vs m3_full — 방향·부호일관성·paired t·Holm

### [표준 디코드 (사전 등록 주 분석)]

**dev**

| variant | per-seed ΔF1 | mean ΔF1 | direction | SC | t(df=4) | p | p(Holm) | reading |
|---|---|---|---|---|---|---|---|---|
| no_motion | +0.0478, -0.0746, -0.0383, -0.1302, -0.0841 | -0.0559 | removal hurts | no | -1.88 | 0.1337 | 0.4222 | term HELPS (removal hurts) |
| no_time | -0.0180, +0.0657, -0.1036, -0.0146, -0.1575 | -0.0456 | removal hurts | no | -1.18 | 0.3044 | 0.6088 | term HELPS (removal hurts) |
| no_state | -0.0590, +0.0600, -0.1376, +0.0118, -0.0785 | -0.0407 | removal hurts | no | -1.17 | 0.3060 | 0.6088 | term HELPS (removal hurts) |
| no_rel | +0.0120, -0.0543, -0.2429, -0.0944, -0.0980 | -0.0955 | removal hurts | no | -2.28 | 0.0844 | 0.4222 | term HELPS (removal hurts) |
| no_src | +0.0295, -0.0840, -0.0406, -0.0485, -0.0791 | -0.0445 | removal hurts | no | -2.19 | 0.0936 | 0.4222 | term HELPS (removal hurts) |

**test**

| variant | per-seed ΔF1 | mean ΔF1 | direction | SC | t(df=4) | p | p(Holm) | reading |
|---|---|---|---|---|---|---|---|---|
| no_motion | +0.0449, -0.0722, -0.0285, -0.1333, -0.1027 | -0.0584 | removal hurts | no | -1.88 | 0.1336 | 0.5343 | term HELPS (removal hurts) |
| no_time | -0.0153, +0.0423, -0.0808, -0.0037, -0.1592 | -0.0433 | removal hurts | no | -1.24 | 0.2834 | 0.5343 | term HELPS (removal hurts) |
| no_state | -0.0633, +0.0266, -0.1065, +0.0135, -0.0794 | -0.0418 | removal hurts | no | -1.59 | 0.1864 | 0.5343 | term HELPS (removal hurts) |
| no_rel | +0.0216, -0.0414, -0.2318, -0.0669, -0.0619 | -0.0761 | removal hurts | no | -1.81 | 0.1443 | 0.5343 | term HELPS (removal hurts) |
| no_src | +0.0049, -0.0724, -0.0084, -0.0235, -0.0773 | -0.0353 | removal hurts | no | -2.11 | 0.1028 | 0.5140 | term HELPS (removal hurts) |

### [revive margin 0.7 (2차 분석)]

**dev**

| variant | per-seed ΔF1 | mean ΔF1 | direction | SC | t(df=4) | p | p(Holm) | reading |
|---|---|---|---|---|---|---|---|---|
| no_motion | +0.0267, -0.0878, +0.0040, -0.0306, -0.0103 | -0.0196 | removal hurts | no | -1.01 | 0.3702 | 1.0000 | term HELPS (removal hurts) |
| no_time | +0.0345, -0.0021, -0.0331, +0.0028, -0.1069 | -0.0209 | removal hurts | no | -0.87 | 0.4322 | 1.0000 | term HELPS (removal hurts) |
| no_state | -0.0237, +0.0221, -0.0370, +0.0189, +0.0015 | -0.0037 | removal hurts | no | -0.31 | 0.7696 | 1.0000 | term HELPS (removal hurts) |
| no_rel | -0.0153, -0.0267, -0.1461, -0.0395, -0.0795 | -0.0614 | removal hurts | YES (5/5) | -2.58 | 0.0612 | 0.3061 | term HELPS (removal hurts) |
| no_src | +0.0337, -0.0398, -0.0491, -0.0069, -0.0248 | -0.0174 | removal hurts | no | -1.19 | 0.3004 | 1.0000 | term HELPS (removal hurts) |

**test**

| variant | per-seed ΔF1 | mean ΔF1 | direction | SC | t(df=4) | p | p(Holm) | reading |
|---|---|---|---|---|---|---|---|---|
| no_motion | +0.0331, -0.0817, -0.0163, -0.0454, -0.0156 | -0.0252 | removal hurts | no | -1.33 | 0.2544 | 1.0000 | term HELPS (removal hurts) |
| no_time | +0.0356, -0.0124, +0.0008, -0.0234, -0.0930 | -0.0185 | removal hurts | no | -0.88 | 0.4310 | 1.0000 | term HELPS (removal hurts) |
| no_state | -0.0392, -0.0168, -0.0397, +0.0151, +0.0130 | -0.0135 | removal hurts | no | -1.13 | 0.3228 | 1.0000 | term HELPS (removal hurts) |
| no_rel | +0.0002, -0.0237, -0.1613, -0.0083, -0.0741 | -0.0535 | removal hurts | no | -1.79 | 0.1482 | 0.7409 | term HELPS (removal hurts) |
| no_src | +0.0535, -0.0529, -0.0312, -0.0139, +0.0005 | -0.0088 | removal hurts | no | -0.49 | 0.6494 | 1.0000 | term HELPS (removal hurts) |

## 4. (b) amb160 대조 — realistic-v1에서 새로 유효해진 항이 있는가

amb160은 불확실성이 깨끗한 벤치마크였고 운동학만 역전했다. 사전 등록 가설이 맞다면 새로 주입한 4종 불확실성(시각 오프셋·보고지연, 상태 동역학, 편제 관계 지문, 출처 오차 이질성)을 겨냥한 항들이 여기서 음수로 돌아서야 한다.

### [표준 디코드 (사전 등록 주 분석)]

| variant | realistic-v1 dev ΔF1 | realistic-v1 test ΔF1 | amb160 test ΔF1 | amb160 SC | 판정 |
|---|---|---|---|---|---|
| no_motion | -0.0559 | -0.0584 | -0.0203 | YES | amb160에서도 이미 유효 |
| no_time | -0.0456 | -0.0433 | -0.0001 | no | YES — 새로 유효 |
| no_state | -0.0407 | -0.0418 | -0.0111 | no | YES — 새로 유효 |
| no_rel | -0.0955 | -0.0761 | -0.0014 | no | YES — 새로 유효 |
| no_src | -0.0445 | -0.0353 | -0.0123 | no | YES — 새로 유효 |

- 새로 유효: **no_time, no_state, no_rel, no_src**
- 신뢰 불가(dev/test 충돌): **none**

### [revive margin 0.7 (2차 분석)]

| variant | realistic-v1 dev ΔF1 | realistic-v1 test ΔF1 | amb160 test ΔF1 | amb160 SC | 판정 |
|---|---|---|---|---|---|
| no_motion | -0.0196 | -0.0252 | -0.0203 | YES | amb160에서도 이미 유효 |
| no_time | -0.0209 | -0.0185 | -0.0001 | no | YES — 새로 유효 |
| no_state | -0.0037 | -0.0135 | -0.0111 | no | YES — 새로 유효 |
| no_rel | -0.0614 | -0.0535 | -0.0014 | no | YES — 새로 유효 |
| no_src | -0.0174 | -0.0088 | -0.0123 | no | YES — 새로 유효 |

- 새로 유효: **no_time, no_state, no_rel, no_src**
- 신뢰 불가(dev/test 충돌): **none**

## 5. (d) 운동학이 realistic-v1에서도 유효한가

- amb160(대조): test ΔF1 -0.0203, SC YES, p 0.0133 → 운동학이 도움이 됐다.
- [표준 디코드 (사전 등록 주 분석)] dev ΔF1 -0.0559 (SC no), test ΔF1 -0.0584 (SC no, t -1.88, p 0.1336) → **amb160에서도 이미 유효**
- [revive margin 0.7 (2차 분석)] dev ΔF1 -0.0196 (SC no), test ΔF1 -0.0252 (SC no, t -1.33, p 0.2544) → **amb160에서도 이미 유효**

## 6. recall·precision 채널 — ΔF1이 어느 경로에서 나오는가

특히 `no_motion`의 ΔF1이 재현율 경로에서 나오는지 확인한다.

### [표준 디코드 (사전 등록 주 분석)]

| variant | split | ΔF1 | ΔRecall | ΔPrecision | 주 경로 |
|---|---|---|---|---|---|
| no_motion | dev | -0.0559 | -0.0394 | +0.0643 | both |
| no_motion | test | -0.0584 | -0.0389 | +0.0594 | both |
| no_time | dev | -0.0456 | -0.0313 | -0.0053 | recall |
| no_time | test | -0.0433 | -0.0303 | +0.0023 | recall |
| no_state | dev | -0.0407 | -0.0292 | -0.0024 | recall |
| no_state | test | -0.0418 | -0.0301 | +0.0023 | recall |
| no_rel | dev | -0.0955 | -0.0653 | +0.0112 | recall |
| no_rel | test | -0.0761 | -0.0528 | +0.0610 | both |
| no_src | dev | -0.0445 | -0.0295 | +0.0039 | recall |
| no_src | test | -0.0353 | -0.0238 | +0.0308 | both |

### [revive margin 0.7 (2차 분석)]

| variant | split | ΔF1 | ΔRecall | ΔPrecision | 주 경로 |
|---|---|---|---|---|---|
| no_motion | dev | -0.0196 | -0.0206 | +0.0059 | recall |
| no_motion | test | -0.0252 | -0.0311 | +0.0212 | both |
| no_time | dev | -0.0209 | -0.0197 | +0.0084 | recall |
| no_time | test | -0.0185 | -0.0228 | +0.0248 | both |
| no_state | dev | -0.0037 | -0.0032 | -0.0020 | both |
| no_state | test | -0.0135 | -0.0168 | +0.0061 | recall |
| no_rel | dev | -0.0614 | -0.0614 | -0.0046 | recall |
| no_rel | test | -0.0535 | -0.0571 | +0.0136 | recall |
| no_src | dev | -0.0174 | -0.0172 | -0.0004 | recall |
| no_src | test | -0.0088 | -0.0160 | +0.0326 | precision |

## 7. (c) 각 항의 겨냥 지표가 특이적으로 나빠지는가 (분모 병기)

### [표준 디코드 (사전 등록 주 분석)]

| variant | 겨냥 지표 | 제거 시 기대 | dev Δ | test Δ | dev SC | test SC | 분모(변형/기준, test) | 일치? |
|---|---|---|---|---|---|---|---|---|
| no_motion | recall | down | -0.0394 | -0.0389 | no | no | 108.8 / 108.8 | YES (양쪽) |
| no_time | wrong_merge_rate | up | -0.0132 | -0.0195 | no | no | 55.1 / 69.0 | NO |
| no_time | impossible_transition_rate | up | -0.0220 | -0.0150 | no | no | 821.2 / 1044.7 | NO |
| no_state | wrong_merge_rate | up | +0.0025 | -0.0072 | no | no | 60.7 / 69.0 | partial (한쪽) |
| no_rel | recall | down | -0.0653 | -0.0528 | no | no | 108.8 / 108.8 | YES (양쪽) |
| no_rel | fragmentation_rate | up | -0.0038 | -0.0115 | no | no | 108.8 / 108.8 | NO |
| no_src | precision | ? | +0.0039 | +0.0308 | no | no | 20.2 / 24.9 | n/a (방향 예측 없음) |

### [revive margin 0.7 (2차 분석)]

| variant | 겨냥 지표 | 제거 시 기대 | dev Δ | test Δ | dev SC | test SC | 분모(변형/기준, test) | 일치? |
|---|---|---|---|---|---|---|---|---|
| no_motion | recall | down | -0.0206 | -0.0311 | no | no | 108.8 / 108.8 | YES (양쪽) |
| no_time | wrong_merge_rate | up | -0.0169 | -0.0235 | no | no | 118.0 / 124.3 | NO |
| no_time | impossible_transition_rate | up | -0.0096 | -0.0087 | no | no | 2008.0 / 2173.6 | NO |
| no_state | wrong_merge_rate | up | -0.0089 | -0.0099 | no | no | 123.3 / 124.3 | NO |
| no_rel | recall | down | -0.0614 | -0.0571 | YES (5/5) | no | 108.8 / 108.8 | YES (양쪽) |
| no_rel | fragmentation_rate | up | -0.0366 | -0.0363 | no | no | 108.8 / 108.8 | NO |
| no_src | precision | ? | -0.0004 | +0.0326 | no | no | 53.4 / 58.6 | n/a (방향 예측 없음) |

- **출처쌍별(per-source-pair) 정밀도는 `metrics.json`에 없다** (보유 키: precision, recall, f1, wrong_merge_rate, fragmentation_rate, impossible_transition_rate, trajectory_consistency_rate, dangling_precision, dangling_recall). `no_src` 행은 전체 정밀도로만 대리하며, 출처쌍 분해 정밀도 점검은 **수행 불가**다.

## 8. 한계

- n=5 → paired t(df=4)는 저검정력. 비유의 p가 효과 부재를 뜻하지 않고, 이 n에서의 유의 p도 취약하다.
- 5개 변형을 하나의 기준선과 비교하며 **다중비교 미보정**(α=0.05에서 FWER ≈23%). Holm 보정 p를 참고 열로 병기했으나 주 판정은 사전 등록대로 미보정 + 부호일관성.
- dev로 선택하고 test를 보고한다. 학습된 모든 변형을 결과와 무관하게 보고했고 제외한 변형은 없다.
- 절대 성능이 모든 모델에서 낮다 (m3_full test F1 ≈ 0.256, recall ≈ 0.160). 디코더가 대부분의 관측을 ∅ 슬롯으로 보내므로 항 효과가 **저재현율 영역**에서 측정된다. 2차(revive) 분석이 이 영역 의존성을 점검한다.
- **`fragmentation_rate`는 비율이 아니다.** 분자가 `fragmented + unrecovered`라 1.0을 넘을 수 있다(실측 최대 1.13). 0~1로 해석하면 안 된다.
- `aggregate`는 섹터별 **비율의 매크로 평균**(카운트 합산 후 재계산이 아님)이라 분모가 작은 섹터도 동일 가중을 받는다. 그래서 분모 평균을 함께 싣는다.
- `wrong_merge_rate`의 분모는 `metrics.json`에 저장되지 않는다(eval.py:136의 `len(result.identities)`). revive 패스의 margin-0 재디코드에서 복구했으며, margin-0은 표준 디코드와 수학적으로 동일하므로 근사가 아니라 정확값이다.
- 40에폭은 amb160 대조와 맞춘 고정 예산이며 수렴점이 아니다(dev F1이 40에폭에서도 상승 중). 변형들은 수렴이 아니라 **동일 예산**에서 비교된다. 80에폭 탐색 런이 이 예산 의존성을 별도로 점검한다(본 집계 제외).
- 모든 수치는 통제된 합성 suite에 대한 것이며 실제 전장 데이터 일반화 성능이 아니다.