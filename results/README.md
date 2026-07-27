# CITA-Net 실험 결과 인덱스 (백마고지 Hill 395)

모델 실험 데이터셋: `data/battlefield_hill395_large`, `data/hard_ambiguity`, `data/robustness`.
추가로 만든 GT 생성 데이터셋(`hill395_world_gt_*`)은 `data2/`로 분리됨(실험에 미사용).

> 공통 통계 주의: 시드 수가 적어(n=3 또는 5) paired t-test는 저검정력. p 단독이 아니라
> **부호 일관성(SC) + dev/test 일치 + 평균차**로 해석함.

---

## 실험 한눈에

| # | 실험 | 폴더 | 핵심 질문 | 결론 |
|---|------|------|-----------|------|
| 1 | **Motion 난이도함수 (메인)** | [hard_ambiguity_main/](hard_ambiguity_main/) | 모호도↑에서 motion이 도움 되나? | **test 5/5 SC, p=0.013 — YES(역전 확인)** |
| 2 | 구성요소 Ablation | [ablation/](ablation/) | 6개 항 중 무엇이 기여? | m1/m2≫m3 (게이트 유효), 항별 기여는 저검정력 |
| 3 | 난이도 Sweep (강건성) | [robustness/](robustness/) | 노이즈·dangling 증가 시 거동 | 난이도↑ 전반 F1↓, motion 이점 유지 경향 |
| 4 | 학습된 항 게이팅 | [term_gating/](term_gating/) | 학습 게이트가 고정 motion 이기나? | m3_full 대비 +0.025, no_motion 대비 −0.027 (못 넘음) |
| 5 | ∅-되살리기 디코더 | [revive_decode/](revive_decode/) | 낮은 recall이 모델 한계? | **아니오 — 디코드 과보수. 사후 revive로 회복(재학습X)** |
| 6 | ∅-임계 재디코드 스윕 | [threshold_sweep/](threshold_sweep/) | 임계만 바꿔 recall 회복되나? | 임계↑ 단독으론 불충분(revive가 필요) |
| 7 | Ablation × 되살리기 | [ablation_revive/](ablation_revive/) | revive 후에도 항 순위 유지? | margin=0.7 재채점서 순위 대체로 유지 |

---

## 1. Motion 난이도함수 — 메인 결과 ⭐

**가설**: motion 항은 "난이도 함수" — 쉬운 데이터엔 무해/해롭고, 모호한 데이터(amb160, ~109 id/sector)에선 도움.

핵심 표 ([n5_report.md](hard_ambiguity_main/n5_report.md), seeds 06–10):

| split | ΔF1(m3_full−no_motion) 평균 | SC | t(df=4) | p |
|-------|---:|:--:|---:|---:|
| test | **+0.0203** | **YES (5/5)** | 4.24 | **0.013** |
| dev | +0.0062 | no (3/5) | 0.55 | 0.613 |

- test F1: m3_full **0.5661±0.0132** > no_motion 0.5457±0.0113.
- 학습된 게이트: motion 가중이 쉬운쌍 0.649 → 모호쌍 0.968 (5/5 상승) — 모형이 스스로 난이도에 반응.
- 결론: **test에서 motion 이점 견고**(부호 5/5·p<0.05), dev는 부호 혼재(경계). p-해킹 방지 위해 n=5에서 종료.
- 파일: `n5_report.md`(최종), `n3_report.md`, `seed19521006_report.md`, `*_summary.csv`, `*_metrics_raw.csv`.

## 2. 구성요소 Ablation

8개 변형 × 3시드 ([summary.md](ablation/summary.md), [ablation_analysis.md](ablation/ablation_analysis.md)):
- **m1/m2 ≫ m3_full** (dev ΔF1 +0.29, p<0.006, SC 예): 게이트/블로킹 단계가 압도적으로 기여.
- 6개 CTA 항(no_time/no_motion/no_state/no_rel/no_src) 개별 제거는 df=2 저검정력 — 부호 혼재, 단일 유의성 주장 불가.
- 파일: `summary_dev.csv`, `summary_test.csv`.

## 3. 난이도 Sweep (robustness)

3변형 × 노이즈축(1×→2×→4×) × dangling축(0.2→0.35→0.5) × 3시드 ([robustness_analysis.md](robustness/robustness_analysis.md)):
- 난이도↑ → 전 변형 F1 단조 하락(예: m3_full dev F1 0.69→0.66→0.57).
- fragmentation·wrong_merge 상승. motion의 상대 이점은 대체로 유지.
- 파일: `summary_{dev,test}_{noise,dangling}.csv`.

## 4. 학습된 항 게이팅 (term_gating)

score=Σ gₖ(쌍맥락)·termₖ 로 변경, 게이팅 모델만 재학습 ([term_gating_analysis.md](term_gating/term_gating_analysis.md)):
- test F1 **0.677±0.030**: m3_full(0.652) **넘음**, no_motion(0.703) **못 넘음**.
- 충실성(gate==1 == 기존 CTA) Δ=0 통과.
- 결론: 학습 게이팅이 고정 motion을 이기지 못함 → 고정 가중이 이미 충분.
- 파일: `compare_{dev,test}.csv`, `gate_summary.csv`.

## 5. ∅-되살리기 디코더 (revive) — 중요 발견 ⭐

재학습 없이 m3_full 체크포인트 로드, Sinkhorn 배정 불변, argmax가 ∅인 경계 엔티티를 사후 되살림 ([SUMMARY.md](revive_decode/SUMMARY.md)):

| 조건 | m3@0 F1 | m3@opt F1(test) | no_motion 교차 | recall 0→opt |
|------|--------:|----------------:|:--:|:--:|
| D0 | 0.652 | 0.713±0.017 | ✅ | 0.524→0.614 |
| noise2x | 0.654 | 0.692±0.011 | ✅ | 0.536→0.591 |
| dang035 | 0.360 | 0.648±0.035 | ✅ | 0.241→0.575 |

- **결론**: m3_full의 낮은 recall은 모델 한계가 아니라 **디코드 과보수**. 사후 라우팅만으로 no_motion을 교차.
- **비용**: recall↑ 대가로 precision↓·wrong_merge↑·**impossible_transition↑**(공짜 아님).
- 파일: `revive_analysis_{D0,noise2x,dang035}.md`, `summary_{dev,test}_*.csv`.

## 6. ∅-임계 재디코드 스윕 (threshold_sweep)

디코드 임계만 0.25~0.90 변경 ([threshold_analysis_D0.md](threshold_sweep/threshold_analysis_D0.md)):
- 임계 0.5 이상은 recall 포화(변화 없음) — 임계 완화 단독으론 recall 회복 불가.
- → revive(margin 라우팅)가 필요한 이유를 뒷받침.
- 파일: `summary_{dev,test}_D0.csv`.

## 7. Ablation × 되살리기 (ablation_revive)

각 변형을 revive margin=0.7로 재채점 ([ablation_revive_analysis.md](ablation_revive/ablation_revive_analysis.md)):
- 충실성(margin=0 == 기록값) Δ=0 통과.
- revive 후에도 항 순위 대체로 유지(no_src·no_time 소폭 우위, motion 제거는 열위).
- 파일: `ablation_revive_{dev,test}.csv`.

---

## 파일 규칙

- `*_analysis.md` / `*_report.md` / `SUMMARY.md` — 사람이 읽는 해석.
- `summary_*.csv` / `*_summary.csv` — 집계 지표(mean±std).
- `*_metrics_raw.csv` — 시드별 원자료.
- `figures/` — 각 실험 폴더의 그래프.

지표 약어: F1/P/R, wrong_merge(오병합률), frag(단편화율), impossible(물리적 불가능 전이율),
traj_cons(궤적 일관성), dang_P/dang_R(dangling 정밀/재현), SC(부호 일관성).
