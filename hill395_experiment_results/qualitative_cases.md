# 백마고지(Hill 395) 실험 — 정성 결과 사례

두 정보망(KG-A 아군 직접관측 / KG-B 항공·감청)이 같은 전장을 관측한 결과를, 모델이 개체정합한 사례.

## ✅ 올바른 병합 (correct merges) — 예시 25건

동일 실체를 A·B 양쪽에서 정확히 같은 개체로 합친 경우.

- [sec_dev_0000] A=중공군 침투조(CCF_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=중공군 잔존대(CCF_Infantry, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (예측타입 CCF_Infantry, 확신도 1.0)
- [sec_dev_0000] A=UN 차량대(VehicleColumn, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=중공군 차량대(VehicleColumn, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (예측타입 VehicleColumn, 확신도 0.995)
- [sec_dev_0000] A=미군 전차(US_Tank, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=73d Tank Bn 전차(US_Tank, src=AERIAL/HUMINT/SIGINT, 10obs)  (예측타입 US_Tank, 확신도 0.862)
- [sec_dev_0000] A=중공군 돌격조(CCF_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=미상 보병(UNKNOWN, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (예측타입 CCF_Infantry, 확신도 0.919)
- [sec_dev_0000] A=국군 53전차중대 전차(ROK_Tank, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=ROK tank(ROK_Tank, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (예측타입 ROK_Tank, 확신도 0.962)
- [sec_dev_0000] A=국군 전차(ROK_Tank, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=국군 53전차중대 전차(ROK_Tank, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (예측타입 ROK_Tank, 확신도 0.957)
- [sec_dev_0000] A=quad-50 대공포(QuadFifty, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=quad-.50(QuadFifty, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (예측타입 QuadFifty, 확신도 1.0)
- [sec_dev_0000] A=미상 표적(UNKNOWN, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=국군 보병 중대(ROK_Infantry, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (예측타입 ROK_Infantry, 확신도 0.997)
- [sec_dev_0000] A=국군 보병 대대(ROK_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=국군 보병 중대(ROK_Infantry, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (예측타입 ROK_Infantry, 확신도 0.982)
- [sec_dev_0000] A=국군 방어중대(ROK_Infantry, src=RADAR/VISUAL_OBS, 10obs) ↔ B=국군 보병 중대(ROK_Infantry, src=AERIAL/HUMINT/SIGINT, 10obs)  (예측타입 ROK_Infantry, 확신도 0.988)
- [sec_dev_0000] A=국군 소총소대(ROK_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=국군 보병 소대(ROK_Infantry, src=ACOUSTIC/AERIAL/HUMINT, 10obs)  (예측타입 ROK_Infantry, 확신도 0.988)
- [sec_dev_0000] A=quad-50 대공포(QuadFifty, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=quad-.50(QuadFifty, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (예측타입 QuadFifty, 확신도 1.0)

## ❌ 잘못된 병합 (wrong merges) — 예시 25건

서로 다른 실체를 같은 개체로 합친 오류. 동일 type 대량 hard negative에서 주로 발생.

- [sec_dev_0001] A=미군 포병대대(US_Artillery, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=미군 155mm 곡사포(US_Artillery, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (A 정답타입 ROK_Artillery, 확신도 0.831)
- [sec_dev_0001] A=국군 보병 대대(ROK_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=국군 방어중대(ROK_Infantry, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (A 정답타입 ROK_Infantry, 확신도 0.045)
- [sec_dev_0002] A=정체불명 부대(UNKNOWN, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=미군 포병대대(US_Artillery, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (A 정답타입 ?, 확신도 0.487)
- [sec_dev_0002] A=아군 보병(ROK_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=국군 보병 소대(ROK_Infantry, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (A 정답타입 ROK_Infantry, 확신도 0.27)
- [sec_dev_0002] A=중공군 재공격대(CCF_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=적 보병(CCF_Infantry, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (A 정답타입 CCF_Infantry, 확신도 0.678)
- [sec_dev_0003] A=중공군 보병 대대(CCF_Infantry, src=RADAR/VISUAL_OBS, 10obs) ↔ B=중공군 침투조(CCF_Infantry, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (A 정답타입 ?, 확신도 0.721)
- [sec_dev_0004] A=중공군 보병 대대(CCF_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=중공군 돌격조(CCF_Infantry, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (A 정답타입 CCF_Infantry, 확신도 0.194)
- [sec_dev_0004] A=국군 105mm 곡사포(ROK_Artillery, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=아군 포병(ROK_Artillery, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (A 정답타입 ROK_Artillery, 확신도 0.647)
- [sec_dev_0004] A=국군 보병 소대(ROK_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=국군 보병 대대(ROK_Infantry, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (A 정답타입 ROK_Infantry, 확신도 0.961)
- [sec_dev_0004] A=국군 보병 소대(ROK_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=분류불가 기동표적(UNKNOWN, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (A 정답타입 ROK_Infantry, 확신도 0.327)
- [sec_dev_0005] A=중공군 돌격조(CCF_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=중공군 침투조(CCF_Infantry, src=ACOUSTIC/HUMINT/SIGINT, 10obs)  (A 정답타입 CCF_Infantry, 확신도 0.201)
- [sec_dev_0006] A=중공군 잔존대(CCF_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) ↔ B=중공군 재공격대(CCF_Infantry, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)  (A 정답타입 CCF_Infantry, 확신도 0.406)

## ✂️ 단편화 (fragments / 미회수) — 예시 20건

정답상 같은 실체인데 모델이 합치지 못한 경우(보수적 abstain). recall 손실 원인.

- [sec_dev_0000] 정답타입 CCF_Infantry: A=중공군 침투조(CCF_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) / B=분류불가 기동표적(UNKNOWN, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)
- [sec_dev_0000] 정답타입 US_Tank: A=73d Tank Bn 전차(US_Tank, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) / B=미상 차량(UNKNOWN, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)
- [sec_dev_0000] 정답타입 ROK_Infantry: A=분류불가 기동표적(UNKNOWN, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) / B=아군 보병(ROK_Infantry, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)
- [sec_dev_0000] 정답타입 CCF_Infantry: A=중공군 돌격조(CCF_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) / B=중공군 재공격대(CCF_Infantry, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)
- [sec_dev_0000] 정답타입 US_Artillery: A=미군 155mm 곡사포(US_Artillery, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) / B=미군 8인치 곡사포(US_Artillery, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)
- [sec_dev_0000] 정답타입 US_Artillery: A=아군 포병(ROK_Artillery, src=RADAR/VISUAL_OBS, 10obs) / B=분류불가 기동표적(UNKNOWN, src=ACOUSTIC/AERIAL/HUMINT, 10obs)
- [sec_dev_0000] 정답타입 ROK_Infantry: A=국군 소총소대(ROK_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) / B=국군 보병 소대(ROK_Infantry, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)
- [sec_dev_0000] 정답타입 ROK_Infantry: A=아군 보병(ROK_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) / B=아군 보병(ROK_Infantry, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)
- [sec_dev_0000] 정답타입 CCF_Infantry: A=중공군 보병 대대(CCF_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) / B=미상 표적(UNKNOWN, src=ACOUSTIC/AERIAL/HUMINT, 10obs)
- [sec_dev_0001] 정답타입 US_Artillery: A=미군 8인치 곡사포(US_Artillery, src=ARTILLERY_OBS/RADAR, 10obs) / B=미군 155mm 곡사포(US_Artillery, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)
- [sec_dev_0001] 정답타입 ROK_Infantry: A=국군 소총소대(ROK_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) / B=정체불명 부대(UNKNOWN, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)
- [sec_dev_0001] 정답타입 CCF_Infantry: A=중공군 잔존대(CCF_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) / B=중공군 잔존대(CCF_Infantry, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs)

## 🚫 매칭불가 정탐 (dangling 정확) — 예시 15건

한쪽 KG에만 보인 부대를 올바르게 '매칭불가'로 판정.

- [sec_dev_0000] 중공군 박격포대(Mortar, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs) (KG-B)
- [sec_dev_0000] 중공군 차량대(VehicleColumn, src=ARTILLERY_OBS/RADAR, 10obs) (KG-A)
- [sec_dev_0000] 적 공병조(Engineer, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs) (KG-B)
- [sec_dev_0000] 120mm 박격포(Mortar, src=AERIAL/HUMINT/SIGINT, 10obs) (KG-B)
- [sec_dev_0000] 공병조(Engineer, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) (KG-A)
- [sec_dev_0000] 적 보병(CCF_Infantry, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs) (KG-B)
- [sec_dev_0000] 중공군 보병 대대(CCF_Infantry, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs) (KG-B)
- [sec_dev_0000] 탐조등반(Searchlight, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) (KG-A)
- [sec_dev_0000] 미상 표적(UNKNOWN, src=ACOUSTIC/AERIAL/HUMINT, 10obs) (KG-B)
- [sec_dev_0001] UN 차량대(VehicleColumn, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs) (KG-B)

## ⚠️ 매칭불가 미탐 (dangling 놓침) — 예시 15건

- [sec_dev_0000] 국군 보병 소대(ROK_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) (KG-A)
- [sec_dev_0000] 아군 포병(ROK_Artillery, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) (KG-A)
- [sec_dev_0000] 보급 차량대(VehicleColumn, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs) (KG-B)
- [sec_dev_0000] 중공군 잔존대(CCF_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) (KG-A)
- [sec_dev_0000] 중공군 잔존대(CCF_Infantry, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs) (KG-B)
- [sec_dev_0000] 국군 전초분대(ROK_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) (KG-A)
- [sec_dev_0000] 국군 방어중대(ROK_Infantry, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs) (KG-B)
- [sec_dev_0001] 미상 차량(UNKNOWN, src=ACOUSTIC/AERIAL/HUMINT/SIGINT, 10obs) (KG-B)
- [sec_dev_0001] 국군 보병 중대(ROK_Infantry, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) (KG-A)
- [sec_dev_0001] 국군 53전차중대 전차(ROK_Tank, src=ARTILLERY_OBS/RADAR/VISUAL_OBS, 10obs) (KG-A)

## 🛰️ 물리적 불가능 전이 (impossible transitions) — 예시 20건

같은 개체로 묶였으나 운동학/상태상 불가능한 전이(예: Destroyed→이동, 과속).

- [sec_dev_0000] Firing(US_Tank) → Firing(US_Tank), 필요속도 1.51 m/s
- [sec_dev_0000] Approaching(CCF_Infantry) → Approaching(CCF_Infantry), 필요속도 95.54 m/s
- [sec_dev_0000] Engaging(CCF_Infantry) → Engaging(CCF_Infantry), 필요속도 29.71 m/s
- [sec_dev_0000] Engaging(CCF_Infantry) → Engaging(CCF_Infantry), 필요속도 58.37 m/s
- [sec_dev_0000] Holding(ROK_Infantry) → Holding(ROK_Infantry), 필요속도 1.97 m/s
- [sec_dev_0000] Holding(ROK_Infantry) → Holding(ROK_Infantry), 필요속도 2.12 m/s
- [sec_dev_0000] Emplaced(US_Artillery) → Firing(US_Artillery), 필요속도 0.74 m/s
- [sec_dev_0000] Firing(ROK_Artillery) → Firing(ROK_Artillery), 필요속도 1.47 m/s
- [sec_dev_0000] Firing(ROK_Artillery) → Firing(ROK_Artillery), 필요속도 1.37 m/s
- [sec_dev_0000] Firing(ROK_Artillery) → Firing(ROK_Artillery), 필요속도 1.68 m/s
- [sec_dev_0000] Firing(ROK_Artillery) → Firing(ROK_Artillery), 필요속도 0.98 m/s
- [sec_dev_0000] Firing(US_Tank) → Firing(US_Tank), 필요속도 19.44 m/s

## 🔀 타입 혼동 (wrong-merge 타입쌍 상위)

- ROK_Infantry + ROK_Infantry: 14회
- CCF_Infantry + CCF_Infantry: 12회
- ROK_Infantry + UNKNOWN: 8회
- UNKNOWN + US_Artillery: 2회
- ROK_Artillery + ROK_Artillery: 2회
- VehicleColumn + VehicleColumn: 2회
- UNKNOWN + US_Tank: 2회
- ROK_Tank + ROK_Tank: 2회
- US_Artillery + US_Artillery: 1회
- CCF_Artillery + UNKNOWN: 1회