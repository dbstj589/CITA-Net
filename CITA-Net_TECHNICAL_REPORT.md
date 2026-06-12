# CITA-Net 기술 전체 분석 문서 (코드 기반)

> 목적: 본 문서는 CITA-Net 코드베이스에 **실제로 구현된 모든 것**(구현 방식, 데이터셋 구축, 방법론·수식, 학습/평가, 백마고지 실험 결과)을 단일 파일로 정리한 것이다. 다른 AI/사람이 이 한 파일만으로 전체를 분석·재현할 수 있도록 파일 경로·함수명·텐서 필드·수식·하이퍼파라미터·정량 수치를 코드 수준으로 명시한다.
>
> 기준 리포지토리: `c:\Users\foxes\LYS\cybermarine\STKG\CITA-Net`
> 의존성: Python ≥ 3.10 (64-bit), `torch`(CPU, 2.12), `numpy`, `PyYAML`. (그래프/발표자료 산출에 `matplotlib`, `python-pptx` 추가 설치)

---

## 0. 한눈에 보기

- **과제**: 두 정보망(KG-A / KG-B)이 비동기·부분적으로 관측한 전장 객체에 대한 **cross-KG 개체정합(entity alignment / co-reference)** + **매칭불가(dangling) 탐지**. 입력은 **관측(observation) 중심 시공간 지식그래프(STKG)**.
- **핵심 아이디어**: 의미(텍스트/타입) + 시공간 + **운동학** + **상태 전이 호환** + **관계 이웃** + **출처 신뢰도** 제약을 하나의 **Constraint-aware Transition Attention(CTA)**로 통합하고, **Sinkhorn 슬롯 디코더**로 군집과 dangling을 동시에 산출.
- **데이터**: 절차적으로 생성한 3개 suite. 본 문서의 실험 대상은 **백마고지(Hill 395)** suite — `data/battlefield_hill395_large/` (1,005,101 트리플 / 75 섹터 / 51,310 관측).
- **결과(백마고지 test)**: F1 **0.686**, Precision **0.895**, Recall **0.561** (40 epoch, CPU).

---

## 1. 리포지토리 구조 (파일 인벤토리)

```
CITA-Net/
├── src/citanet/
│   ├── config.py              # 모든 하이퍼파라미터 dataclass + YAML 로더
│   ├── engine.py              # 소형 suite 학습/평가 + build_model
│   ├── engine_large.py        # 대형 suite 스트리밍 학습/평가 + bundle save/load
│   ├── decode.py              # 디코드: lite/greedy(decode_entities) + M3 full(decode_full)
│   ├── eval.py                # 평가 지표(evaluate_decode, evaluate_full, aggregate)
│   ├── losses.py              # 5개 손실항 + total_loss
│   ├── serialize.py           # Part-B 출력 문서 build_part_b
│   ├── output_schema.py       # Part-B 스키마 검증 validate_output
│   ├── model/
│   │   ├── citanet.py         # 최상위 모델 CITANet + ModelOutput
│   │   ├── encoder.py         # 5채널 출처-게이트 인코더
│   │   ├── graph_encoder.py   # RGAT(관계-맥락 GNN)
│   │   ├── cta.py             # Constraint-aware Transition Attention
│   │   ├── decoder.py         # Sinkhorn 슬롯 디코더
│   │   ├── heads.py           # PairHead, DanglingHead
│   │   ├── featurize.py       # FeatureSpace, ScenarioFeatures, assemble_features, REL_NAMES
│   │   └── text.py            # Vocab, LearnedTextEncoder
│   └── data/
│       ├── schema.py          # Observation/Entity/Landmark/Event/Relation/Scenario
│       ├── ontology.py        # Ontology 접근(API) + load_ontology
│       ├── loader.py          # 소형 suite JSON-LD 로더 + consistency 검증
│       ├── stream.py          # 대형 suite 스트리밍 로더 + featurize_sector
│       ├── blocking.py        # O(M^2) 후보 생성(brute force) + CandidatePair
│       ├── blocking_grid.py   # 그리드+시간 후보 생성(스케일업) + blocking_stats
│       ├── kinematics.py      # 운동학 feasibility
│       └── text_utils.py      # text_cosine
├── scripts/
│   ├── gen_dataset.py             # 소형 suite 생성기(JSON-LD scenarios)
│   ├── gen_dataset_large.py       # 대형 suite 생성기(N-Triples sectors)  [frozen]
│   ├── gen_dataset_hill395.py     # ★ 백마고지 suite 생성기(본 작업)
│   ├── train.py / evaluate.py / predict.py     # 소형 suite 파이프라인
│   ├── train_large.py             # 대형 suite 학습/평가
│   ├── extract_hill395_results.py # ★ 정량/정성 결과 추출(본 작업)
│   ├── plot_hill395_results.py    # ★ 결과 그래프(PNG) 생성(본 작업)
│   └── make_hill395_deck.py       # ★ 발표자료(PPTX) 생성(본 작업)
├── configs/
│   ├── cita_full.yaml / cita_full_large.yaml
│   └── cita_full_hill395.yaml     # ★ 백마고지 학습 설정(본 작업)
├── data/
│   ├── battlefield_stkg_dataset/  # 소형 suite (scenarios, JSON-LD) [frozen]
│   ├── battlefield_stkg_large/    # 대형 suite (sectors, N-Triples) [frozen]
│   └── battlefield_hill395_large/ # ★ 백마고지 suite(본 작업)
├── tests/                         # test_data_m0.py, test_model_m2.py, test_large_pipeline.py, test_hill395_pipeline.py ★
└── hill395_experiment_results/    # ★ 실험 산출물(지표/그림/정성/PPTX)
```

`[frozen]` = 변경 금지(기존 동결 데이터). 본 작업은 **새 형제 디렉토리/스크립트로만** 추가했다.

---

## 2. 문제의 형식적 정의

- **STKG** = (Entities, Observations, Events, Landmarks, Relations).
- **관측(Observation)** `o` = ⟨obs_id, kg_id∈{A,B}, source, label_text, type, type_confidence, state, state_confidence, time(ISO), (easting, northing, elevation, crs, mgrs), cep_m, (speed_mps, heading_deg), relations[], events[], provenance⟩. (`data/schema.py:Observation`)
- **로컬 개체(Entity / track)** = 한 KG 내에서 동일 `local_entity_id`를 공유하는 관측 묶음.
- **입력**: 관측 집합 `O = O_A ∪ O_B`.
- **출력**:
  - 정체성 군집 `{I_k}`, 각 `I_k ⊆ O` (관측→정체성 many-to-one),
  - cross-KG 1:1 정합쌍 `(e^A, e^B)`,
  - 매칭불가 집합 `D`(한쪽 KG에만 존재 → ∅ 슬롯으로 흡수).
- **불변(invariants)**: 관측→정체성 many-to-one; 정체성↔정체성 cross-KG one-to-one; dangling은 강제 매칭 금지(abstain).

---

## 3. 데이터 모델 & 온톨로지

### 3.1 온톨로지 파일 (dataset_root/ontology/)
`data/ontology.py:Ontology`가 4개 YAML을 읽어 모델·블로킹·CTA·인코더에 제공한다.

- **classes.yaml**: `type → {category, mobility, default_state, default_v_max_mps, v_max_mps{state:값}}`.
  - `Ontology.v_max(type, state)` → 블로킹 reach, CTA b_motion, L_trajectory에 사용.
  - `Ontology.category(type)`, `types_compatible(t1,t2,by_category)` → 블로킹 타입 호환(동일/UNKNOWN/동일카테고리).
- **states.yaml**: `states[]`, `compatibility[s1][s2]∈(0,1]`(비대칭), `overrides[type][s1][s2]`, `eps`.
  - `Ontology.state_compat(s1,s2,type)` → CTA b_state. `state_eps` → CTA ε.
- **sources.yaml**: `source → {reliability, type_reliability, pos_reliability, cep_m}`.
  - `source_reliability` → 인코더 게이트 prior. `source_cep` → 블로킹 reach 상한.
- **relations.yaml**: `predicate → {symmetric, event?, description}`.
  - `relation_names`, `is_symmetric`.
- **context.jsonld**: JSON-LD @context(소형 suite의 .jsonld용; 대형 suite는 N-Triples).

### 3.2 표현 포맷
- **소형 suite**: 시나리오별 `kg_A.jsonld + kg_B.jsonld + commons.jsonld`(JSON-LD), `data/loader.py`가 파싱.
- **대형/백마고지 suite**: 섹터별 `stkg.nt`(N-Triples, 1줄=1트리플; 정확·스트리밍 카운팅) + `observations.jsonl`(평탄화 캐시, 학습 fast-path) + `manifest.json` + `labels/{gold_identities.json, dangling.json}`. `data/stream.py:load_sector_observations`가 `observations.jsonl`을 읽는다.
- **gold_identities.json**: `{identities:[{gold_identity_id, true_type, member_observations[], member_local_entities{A,B}}], dangling_observations[], dangling_local_entities[], assignment:[{obs_id, gold_identity_id}]}`. (대형은 O(M²) 페어 목록 없이 assignment만 저장)

---

## 4. 데이터셋 구축 — 백마고지(Hill 395) suite

생성기: `scripts/gen_dataset_hill395.py`. 기존 대형 suite(`gen_dataset_large.py`)의 검증된 **실현·emit 기계부를 재사용**하고 **도메인(온톨로지·로스터·라벨)만 백마고지로 교체**. 온톨로지 YAML도 생성기가 `write_ontology()`로 emit → 단일 스크립트로 완전 재현.

### 4.1 도메인(시드 → 절차적 확장)
디스크의 실제 백마고지 사료(`STKG/백마고지시나리오/`: entity dictionary 386개, taxonomy, relation dictionary, `.trig` 110 이벤트/330 reified statement, Oct 1952 타임라인)를 **시드**로 사용하고, 100만 규모로 절차적 확장.

**객체 타입(faction-functional; 진영을 타입에 내장 → 진영 간 오병합 구조적 차단)** — `gen_dataset_hill395.py:TYPES`
| type | category | mobility | base v(m/s) | default_state |
|---|---|---|---|---|
| UNKNOWN | unknown | unknown | 25 | Unknown |
| Unit | formation | na | 0 | Unknown |
| ROK_Infantry / CCF_Infantry | infantry | foot | 2 | Holding / Approaching |
| US_Tank / ROK_Tank | tank | tracked | 12 / 11 | Moving |
| ROK_Artillery / US_Artillery / CCF_Artillery | artillery | towed | 0 | Emplaced |
| Mortar | mortar | towed | 0 | Emplaced |
| CCF_AA | aa | towed | 0 | Emplaced |
| QuadFifty | aa | wheeled | 6 | Halted |
| Engineer | engineer | foot | 2 | Moving |
| VehicleColumn | vehicle | wheeled | 14 | Moving |
| Searchlight | support | towed | 0 | Emplaced |

`v_max_mps[state]`는 `_v_max_for(speed, mobility)`로 자동 생성: towed/na는 모든 상태 0; 그 외 fast 상태(Moving/Approaching/Withdrawing/Unknown/Destroyed)=speed, Engaging=min(speed,2), 정적(Halted/Occupying/Holding/Emplaced/Firing)=0.5.

**상태(11종)**: Moving, Halted, Approaching, Engaging, Occupying, Holding, Emplaced, Firing, Withdrawing, Unknown, Destroyed. 호환행렬 `_compat_matrix()`(규칙기반·비대칭): 대각 1.0; Destroyed→활성 0.01, Destroyed→Unknown 0.3; 활성→Destroyed 0.6; Withdrawing→Occupying 0.2 등; towed override(Emplaced/Firing→Moving 0.02).

**출처(7종)** — KG 분할로 cross-KG 비대칭 관측 구현:
- KG-A(아군 직접): `VISUAL_OBS`(cep15,typeRel.9), `RADAR`(cep30,.4), `ARTILLERY_OBS`(cep35,.5)
- KG-B(원거리/우회): `AERIAL`(cep25,.7), `SIGINT`(cep50,.6), `ACOUSTIC`(cep60,.3), `HUMINT`(cep90,.7)

**관계(12종)**: 기존 9(partOf, follows, near, firesAt, engagedWith, emplacedAt, movesToward, supports, screens) + participatesIn(event) + **전투 특화 3종(occupies, withdrawsFrom, reinforces)**. 78개 한국어 동사 술어를 이 집합으로 매핑.

**라벨 풀(한국어)** `LABELS` — type별 패러프레이즈(예 CCF_Infantry: "중공군 보병 대대/돌격조/침투조/증원대/재공격대/잔존대"), `UNKNOWN_LABELS`("미상 표적/미상 보병/정체불명 부대/…"), `MISID`(동일카테고리 오식별: US_Tank↔ROK_Tank, ROK_Artillery↔US_Artillery, CCF_AA↔QuadFifty).

### 4.2 섹터 로스터 — `build_sector(rng, knobs, sx, sy)`
하나의 섹터 = 한 하위교전(국면×하위지역). 지형 landmark(crest, north_tip, north_slope, MLR, Objective A/B, valley, YOKKOK River)를 타일 원점 상대 배치(UTM52S, BASE≈(300000, 4238000)). 이벤트 5종(Attack/Counterattack/FireSupport/Occupy/Illuminate). 국면(phase) = (sx/pitch + sy/pitch) mod 3로 CCF 공격 상태 시퀀스 변화.

객체 구성(identities/sector≈40 비례): CCF 보병 30%(돌격/침투/증원/잔존 소부대 분해, follows 체인, 사면 교차궤적 lane-swap → hard negative), ROK 보병 28%(방어 Holding/Engaging, 일부 counterattack Occupying), 전차 10%(Moving→Halted→Firing, supports/firesAt/screens), 포병 8%(Emplaced→Firing), 박격포 5%, 대공 5%, 공병 5%, 차량 4%, 조명 1, Destroyed 잔해 5%(bait).

### 4.3 실현(노이즈) — `realise_sector(...)`
각 객체의 두 KG 트랙을 관측으로 실현:
- 각 트랙 관측 수 = `obs_per_track`(=10). 출처는 KG별 풀에서 샘플.
- **위치 노이즈**: `sig = cep/1.1774`, easting/northing에 가우시안. **시각**: `t + clock_offset(KG) + N(0,1)` (KG-B +2s).
- **타입/라벨 불확실성**: 12% UNKNOWN(conf 0.6), 다음 10% 동일카테고리 오식별(conf 0.7), 그 외 정상 라벨 패러프레이즈.
- **신뢰도**: type_conf = clip(source_typeRel·scale + 0.05·rand), state_conf = clip(source_rel + 0.05·rand).
- 정적 상태(Emplaced/Halted/Destroyed/Holding/Occupying/Firing)는 speed=0.
- **dangling**: `assign_robots`가 `dangling_ratio`(=0.2) 확률로 한 KG만 배정.
- **gold/dangling 산출**: 비-dangling 객체는 identity(member_observations + member_local_entities{A,B}), dangling은 dangling_observations/entities. assignment(obs→gold) 동시 저장.

### 4.4 emit + 섹터 드라이버
- `emit_ntriples(...)`: Unit/Landmark/Event/Entity/Observation 트리플을 N-Triples로 기록(IRI 경로 세그먼트 `…/hill395/…`), 트리플 수 반환.
- `write_sector(split, sid, seed, idx, knobs)`: 섹터 dir에 `stkg.nt + observations.jsonl + manifest.json + labels/` 기록.
- `build_split / build_suite`: dev 12 / test 13 고정, train은 누적 트리플이 target(1e6)에 도달하거나 ≥50섹터까지 증식. **분리 시드베이스**(train 1e6 / dev 2e6 / test 3e6) → 누수 안전. `manifest_global.json` 기록.

### 4.5 생성 결과(검증됨)
- **총 1,005,101 트리플 / 75 섹터(train 50 / dev 12 / test 13) / 51,310 관측.**
- 무결성 전수검사(75섹터): stkg.nt 라인수 == manifest n_triples; observations 수 일치; vocab 위반 0(type/state/source/relation); gold∪dangling = 전체 관측(정확 분할, 중복 0); 모든 매칭 identity가 KG-A·B 양쪽에 존재; dangling.json == gold dangling.
- 분포: type — CCF_Infantry 14,830 / ROK_Infantry 13,060 / UNKNOWN 6,020 / VehicleColumn 2,840 / US_Tank 2,660 / ROK_Tank 2,550 / Mortar 1,350 / Engineer 1,330 / US_Artillery 1,270 / CCF_AA 1,200 / QuadFifty 1,190 / CCF_Artillery 1,190 / ROK_Artillery 1,150 / Searchlight 670. state — Engaging 12,609 / Firing 8,335 / Holding 7,657 / Moving 6,365 / Approaching 4,718 / Occupying 3,763 / Emplaced 3,038 / Withdrawing 2,333 / Destroyed 1,500 / Halted 992. source — 7종 균형(6.3k~8.6k).

---

## 5. 방법론 & 수식 (CITA-Net 파이프라인)

전체 forward: `model/citanet.py:CITANet.forward`. 단계: 인코더 → (선택)GNN → CTA → pair/dangling 헤드 → (선택)디코더.

### 5.1 후보 생성(Blocking) — `data/blocking_grid.py`
(셀x, 셀y, 시간버킷) 공간해시로 시공간 이웃만 검사(O(M²) 회피, 결과는 brute force와 동일 기준).

```
관측별 reach 반경:
  reach_i = reach_vmax_mult · v_max(type_i, state_i) · dt_max + cep_i + max_cep + r_floor + reach_extra
후보쌍 (i,j) 채택 ⇔
  0 ≤ t_j − t_i ≤ dt_max                                   (전방 시간창)
  ∧ types_compatible(type_i, type_j, by_category)          (타입 호환)
  ∧ dist(i,j) ≤ reach_vmax_mult·max(v_i,v_j)·dt + (cep_i+cep_j+r_floor) + reach_extra   (운동학 도달)
  ∧ (use_text_gate? text_cos ≥ θ_text 또는 UNKNOWN)        (텍스트 게이트, 대형은 off)
```
설정(백마고지): dt_max=180s, cell=400m, r_floor=80m, type_by_category=true, use_text_gate=false. 결과: recall ≥ 0.987, ~10× 축소, grid ⊆ brute force.

### 5.2 특징(Featurization) — `model/featurize.py:assemble_features`
`ScenarioFeatures`(텐서 번들) 필드:
- 관측별: `token_ids/token_mask`(라벨), `xy`((E,N) 중심정규화/1000), `t`((t−min)/300), `state_id/type_id/source_id`, `type_conf`.
- 쌍별(P개): `pair_i, pair_j, pair_cross`; `p_dt, p_dist, p_req_speed, p_feas_speed`(운동학), `p_state_compat`(C[s_i,s_j,type]), `p_src_i, p_src_j`, `p_rel_jaccard`(관계+이벤트 집합 자카드), `p_text_cos`; `pair_label`(cross-KG 동일 gold), `dangling_label`.
- 그래프 `edges`(src,dst,rel_id): 동일트랙(sametrack 양방향) + 관계엣지(predicate→rel_id, 미지술어는 near로 폴백) + 공통(kg,event) 엣지(그룹크기 > max_event_group=16이면 스킵하여 O(M²) 방지).
- `entity_of`: obs_id → "kg:local".
- `REL_NAMES`(14): follows, near, partOf, firesAt, engagedWith, emplacedAt, movesToward, supports, screens, **occupies, withdrawsFrom, reinforces**, event, sametrack.

### 5.3 관측 인코더 — `model/encoder.py:ObservationEncoder` (5채널 + 출처 게이트)
```
h(o) = LayerNorm( Σ_c  g_c(src) · f_c(o) ),   c ∈ {text, st, state, type, source}
  f_text   = LearnedTextEncoder(token_ids, mask)
  f_st     = MLP( Fourier(x, y, t) ),  Fourier(u)=[sin(2^b·u), cos(2^b·u)]_{b=0..7}
  f_state  = Embedding(state_id)
  f_type   = Linear([ Embedding(type_id) ; type_confidence ])
  f_source = Embedding(source_id)
게이트:  g(src) = σ( W · gate_embed(src) + logit(reliability(src)) ) ∈ (0,1)^5
```
게이트 prior는 sources.yaml 신뢰도로 초기화(gate_lin 가중치/bias=0). `use_source_gate=false`면 단순 합. d_model=128.

### 5.4 관계-맥락 GNN — `model/graph_encoder.py:RelationContextGNN` (RGAT, 2 layer, 4 head)
self-loop 관계 추가 후 레이어마다:
```
msg(e)  = W·h_{src(e)} + rel_embed( rel(e) )
e_score = LeakyReLU( (msg_h · a_src) + (W·h_{dst} · a_dst) )        # 헤드별
α_e     = softmax_{dst(e)}( e_score )                               # 목적노드별 정규화
h'_v    = LayerNorm( h_v + ELU( Σ_{e: dst=v} α_e · msg_h(e) ) )     # 잔차 + 정규화
```
`n_relations = len(REL_NAMES) = 14` (citanet.py에서 결합; 미지 술어는 near 버킷). 위치 모호 시 관계가 판별신호(예: 'lead를 follows하는 UNKNOWN').

### 5.5 운동학 feasibility — `data/kinematics.py:feasibility`
```
required_speed = dist(i,j) / max(Δt, 1e-3)
feasible_speed = max( v_max(type_i,state_i), v_max(type_j,state_j) ) + (cep_i + cep_j) / Δt
feasible       ⇔ required_speed ≤ feasible_speed
violation      = max(0, required_speed − feasible_speed)
```

### 5.6 CTA — `model/cta.py:CTA` (핵심)
시간정렬 후보쌍 (o_i → o_j)에 대해:
```
score(i→j) = w0·sim_sem + b_time + b_motion + b_state + b_rel + b_src
p_transition = σ(score)

sim_sem  = cos(h_i, h_j)
b_time   = −BIG  if (t_j < t_i − ε)  else 0                  # 하드, 비학습 (BIG=50, ε=5s)
b_motion = −α · softplus( (required_speed − feasible_speed) / β )   # β=exp(logβ), init 2.0
b_state  = γ · log( C[s_i, s_j] + ε_state )                  # ε_state from states.yaml(0.01)
b_rel    = δ · jaccard( {rel,event}_i , {rel,event}_j )
b_src    = SrcBias[ source_i, source_j ]                     # (n_src × n_src) 학습표
```
학습 파라미터: `w0, α, log_beta, γ, δ, src_bias`. `enabled_terms`(config)로 각 항 on/off → **ablation이 설정 변경만으로 가능**(꺼진 항은 0으로 zero-out되어 pair head/rationale에도 누출 없음). 백마고지: `[sem,time,motion,state,rel,src]` 전부.

### 5.7 예측 헤드 — `model/heads.py`
- **PairHead**: 입력 `[h_i, h_j, |h_i−h_j|, h_i·h_j, CTA(6), (Δt/180, dist/1000, text_cos)]` → MLP(in=4d+6+3) → cross-KG same 로짓.
- **DanglingHead**: 입력 `[h_o, (best_logit, mean_logit, n_cand/5, max_sim)]`(관측의 cross-KG 후보 증거 요약) → MLP → dangling 로짓. (증거는 `citanet.forward`에서 scatter_reduce로 집계)

### 5.8 정체성-궤적 디코더 — `model/decoder.py:IdentityDecoder` (Sinkhorn)
```
h_ctx = h + σ(cta_gate) · ( Σ_j p_transition(i,j)·h_j  / Σ_j p_transition(i,j) )   # CTA 컨텍스트 집계
slots = [ learnable_slots(K) ; null(1) ]                                          # (K+1, d)
Z     = ( q(h_ctx) · k(slots)^T ) / √d                                            # (M, K+1) 로짓
A     = sinkhorn_assign(Z/τ, iters):   반복[ 행 정규화(행합=1) → 실슬롯 상대 열균형 ]; ∅열 제외
```
`sinkhorn_assign`은 행-확률(각 관측 분포합=1) + 실슬롯 간 평균중심 열균형(col_strength)로 한 슬롯 독점/∅ 독점 방지(강한 열 marginal은 불균등 크기 정체성을 단편화하므로 약하게). ∅(마지막) 열은 균형서 제외 → dangling 자유 흡수. 디코드 = argmax_k A.
설정: num_slots=48, iters=20, τ=0.5, col_strength=0.1.

**디코드 출력** — `decode.py:decode_full → FullDecodeResult`:
- 개체→슬롯: 관측 mean assignment의 argmax(또는 ∅질량 > 0.5면 dangling). 슬롯 내 KG별 최고질량 개체로 1:1(나머지는 conflicts). 슬롯 내 관측 시간정렬 → trajectory, 연속전이 transitions(feasibility + p_transition). `rationale`=정합쌍 cross 후보의 CTA 항 평균. fused_representative(평균위치+최근상태). dangling은 abstain(이유·최근접 후보 포함).

### 5.9 손실 — `losses.py`
```
L = λ_pair·L_pair + λ_trans·L_trans + λ_traj·L_traj + λ_dang·L_dang + λ_assign·L_assign

L_pair   = BCE_with_logits( pair_logits[cross], pair_label[cross], pos_weight )       # 교차-KG 후보쌍만
L_dang   = BCE_with_logits( dangling_logits, dangling_label )
L_trans  = BCE_with_logits( CTA.score, same_identity? )   # transition_level.json 있으면 그걸로,
                                                          # 없으면 gold assignment에서 유도(동일 gold면 1)
L_traj   = mean( σ(pair_logit) · max(0, req_speed − feas_speed) ) / 10               # 소프트 운동학 일관성
L_assign = CE( Z[valid], target_slot )    # Sinkhorn 할당(detach)에서 greedy 최대중첩으로 slot↔gold 정렬,
                                          # 관측별 타깃슬롯(∅=dangling)에 교차엔트로피
```
백마고지 가중치(`cita_full_hill395.yaml`): λ_pair 2.0, λ_trans 0.3, λ_traj 0.5, λ_dang 1.0, λ_assign 2.0, pair_pos_weight 5.0.

### 5.10 단계(stage) 구성
`config.stage`와 enable 플래그로 ablation 단계 표현(README의 M0–M3 대응): M1=pair+dangling; M2=+GNN+transition(graph.enabled); M3=+decoder+trajectory+assign(decoder.enabled). 백마고지는 **Full(M3)**: graph on, cta 6항, decoder on, source-gate on.

---

## 6. 설정(Config) 전체 — `config.py`

| 그룹 | 키 | 기본 | 백마고지 |
|---|---|---|---|
| top | stage / seed / device / data_root / ontology_dir | — | cita_full_hill395 / 19521006 / auto / battlefield_hill395_large |
| blocking | dt_max_s, theta_text, r_err_floor_m, reach_vmax_mult, reach_extra_m, grid_cell_m, use_text_gate, type_by_category, allow_unknown_type | 180,0.3,5,1,0,400,T,F,T | r_floor 80, text_gate F, by_category T |
| encoder | d_model, text_encoder, text_max_tokens, use_source_gate, spatiotemporal_freqs | 128,learned,16,T,8 | 동일(gate on) |
| graph | enabled, gnn_layers, n_heads, hops | F,2,4,2 | enabled T |
| cta | time_eps_s, big_penalty, motion_beta, enabled_terms | 5,50,2,[sem,time,motion,state,rel,src] | 동일(전항) |
| decoder | enabled, num_slots, sinkhorn_iters, sinkhorn_temp, sinkhorn_col_strength | F,16,30,0.5,0.1 | enabled T, slots 48, iters 20 |
| loss | lambda_pair/transition/trajectory/dangling/assign, pair_pos_weight | 1,1,0.5,1,1,3 | 2,0.3,0.5,1,2,5 |
| train | lr, epochs, weight_decay, grad_clip, log_every | 1e-3,200,0,5,20 | epochs 60(실행 40) |

---

## 7. 학습/평가 엔진 — `engine_large.py`

- `build_feature_space_large(cfg)`: 일부 train 섹터 라벨로 Vocab 빌드 + 온톨로지 로드 → `(FeatureSpace, Ontology)`.
- `train_large(cfg, sectors_per_epoch, full_sectors, dev_eval_sectors=6)`: 섹터 단위 스트리밍(한 번에 한 섹터 featurize→forward→loss→step). Adam, grad-clip. 매 평가시 dev F1로 best state 추적·복원.
- `evaluate_large(cfg, model, fs, ont, split, max_sectors)`: 스트리밍 디코드 → `evaluate_full` → `aggregate`(섹터 매크로 평균).
- `save_large_bundle / load_large_bundle`: `model.pt + vocab.json + feature_space.json`. (재학습 없이 재현)
- 스트리밍 peak memory ~16MB(1M 트리플 suite).

---

## 8. 평가 지표 — `eval.py:evaluate_full`

- gold_pairs = {(e^A,e^B)} (양쪽 존재 identity). pred_pairs = 디코드 identity의 {(A,B)}.
- **Precision** = |pred∩gold| / |pred|, **Recall** = |pred∩gold| / |gold|, **F1** = 조화평균.
- **Wrong-Merge Rate** = (한 예측 정체성이 2개 이상 서로 다른 gold를 포함) / |예측 정체성|.
- **Fragmentation Rate** = (한 gold가 여러 슬롯으로 쪼개짐 + 미회수) / |gold_pairs|.
- **Impossible-Transition Rate** = (전이 중 feasible_motion=False) / 전체 전이; **Trajectory-Consistency** = 1 − 그 값.
- **Dangling Precision/Recall** = (kg,local) 튜플 기준.
- (소형 suite `evaluate_decode`는 추가로 Hits@1/MRR.)

---

## 9. 실험: 백마고지 — 설정·결과

### 9.1 설정
- 데이터: `battlefield_hill395_large`(1,005,101 트리플 / 75 섹터). 모델: Full CITA-Net(`cita_full_hill395.yaml`).
- 명령: `python scripts/train_large.py --config configs/cita_full_hill395.yaml --full-sectors --epochs 40` (전체 50 train 섹터/epoch, CPU ~45s/epoch).

### 9.2 학습 수렴
총손실 17.63(epoch1) → **1.63(epoch40)**; best dev-F1 **0.693**; 항별(ep40): pair 0.16 / dangling 0.015 / transition 0.18 / trajectory 0.024 / assign 1.24.

### 9.3 정량 결과 (섹터 매크로 평균)
| 지표 | dev | test | 기존 large suite(README 참고) |
|---|---|---|---|
| Precision | 0.871 | **0.895** | ~0.80 |
| Recall | 0.548 | 0.561 | ~0.40 |
| F1 | 0.669 | **0.686** | ~0.52 |
| Wrong-merge rate | 0.132 | 0.121 | — |
| Fragmentation rate | 0.730 | 0.685 | — |
| Trajectory consistency | 0.757 | 0.754 | — |
| Impossible-transition | 0.243 | 0.246 | — |
| Dangling precision | 0.682 | 0.621 | — |
| Dangling recall | 0.582 | 0.571 | — |
| Streaming peak (MB) | 16.2 | 16.5 | — |

### 9.4 정성 결과(요지) — `extract_hill395_results.py` 산출
- **올바른 병합**: KG-A(직접관측)와 KG-B(항공·감청)의 동일 실체를 정확 정합(예: '중공군 침투조'↔'중공군 잔존대', 확신도 0.9~1.0).
- **잘못된 병합**: 동일 진영·동일 type 보병 소부대(hard negative)에 집중, 확신도 낮음(0.04~0.4) → 모델이 불확실해함.
- **단편화/dangling**: 불확실하면 abstain(정밀도 우선); 한쪽 KG만 본 부대를 dangling으로 다수 정탐.
- **해석**: 고정밀(P≈0.9)·중회수(R≈0.56). 의도한 hard negative·노이즈·dangling이 실제로 어려운 케이스로 작동, 모델은 보수적.

### 9.5 산출물 폴더 `hill395_experiment_results/`
- `metrics_summary.{md,json}`, `per_sector_metrics.csv`, `training_curve.csv`, `dataset_stats.json`, `qualitative_cases.{md,json}`, `type_confusion.csv`, `samples/output_*.json`(Part-B 전문).
- `figures/`: training_curve / dev_test_metrics / baseline_compare / per_sector_metrics / dataset_distributions / type_confusion (PNG).
- `CITA-Net_백마고지_발표자료.pptx`(21슬라이드, 수식·구조·결과 포함).

---

## 10. 재현 명령 (전체)

```bash
# 0) 64-bit venv 사용
PY=.venv/Scripts/python.exe

# 1) 데이터셋 생성 (~1M 트리플)
$PY scripts/gen_dataset_hill395.py --build-suite
#   온톨로지만 재생성: --emit-ontology ;  일부만: --split dev --n-sectors 2

# 2) 학습 + dev/test 평가
$PY scripts/train_large.py --config configs/cita_full_hill395.yaml --full-sectors --epochs 40

# 3) 결과 추출 / 그래프 / 발표자료 (학습된 모델 재로드)
$PY scripts/extract_hill395_results.py
$PY scripts/plot_hill395_results.py
$PY scripts/make_hill395_deck.py

# 4) 테스트 (구조 무결성 + 회귀)
$PY -m pytest tests/test_hill395_pipeline.py -q
$PY -m pytest -q            # 전체(소형/대형 suite 회귀 포함)
```

---

## 11. 구현상 결합/주의점 (다른 AI 분석 시 필수)

- **REL_NAMES ↔ n_relations 결합**: 관계 술어를 추가하면 `model/featurize.py:REL_NAMES`(edge-type 버킷)와 `model/citanet.py`의 GNN `n_relations`가 함께 맞아야 한다. 과거 `n_relations=11` 하드코딩 → 술어 추가 시 forward에서 `IndexError: index out of range`. 현재 `n_relations=len(REL_NAMES)`로 결합(가산적·하위호환; 미지 술어는 near 버킷 폴백).
- **상태 vocab 일치**: 관측이 쓰는 모든 state는 states.yaml `states[]`에 있어야 한다(과거 `Halted` 누락 버그를 수정·재생성). 생성기가 온톨로지를 emit하므로 STATES 리스트만 고치면 전파.
- **공정 비교 원칙**: blocking을 고정하고 스코어러/디코더만 교체해야 모델 기여가 분리됨(`featurize_sector`는 cfg.blocking으로 후보 생성).
- **동결 데이터**: `battlefield_stkg_large`, `battlefield_stkg_dataset`는 변경 금지(메모리 제약). 본 작업은 새 형제 디렉토리만 추가.
- **train_large.py RUN_DIR**: 현재 `runs/cita_full_large`로 하드코딩되어 stage명과 무관하게 거기에 저장됨(백마고지 결과가 그 경로에 덮임). 분리 저장하려면 소량 수정 필요.
- **결정론성**: 모든 생성/학습은 시드 기반. 단 train_large는 epoch별 섹터 셔플에 `random`을 사용.

---

## 12. 한계 & 확장 지점 (분석/논문용)

- **관측중심 vs 이벤트중심**: 본 suite는 CITA-Net 관측형(드롭인). 원본 백마고지 `.trig`의 reified 이벤트-KG(subject-predicate-object-time-location-event) 충실 재현은 별도 로더/평가가 필요(미구현).
- **표준 시맨틱 표현 미사용**: GeoSPARQL(geo:hasGeometry/WKT), OWL-Time(time:Interval), SKOS(altLabel)는 현재 미사용(좌표는 custom geo:easting/northing). prov-O는 일부 사용(wasAttributedTo/generatedAtTime).
- **좌표 변환**: MGRS는 stub(`_mgrs`), 실제 pyproj 변환 아님.
- **비교 baseline 부재**: 동일 입력 형식 기성 모델이 없음 → (A) 인접분야(KG-EA/ER/데이터연관) 어댑터 이식, (B) 내부 ablation(enabled_terms/graph/gate/decoder), (C) 단순·oracle baseline, 난이도 스윕(dangling/CEP/type밀도/skew), 다중시드 통계검정으로 우수성 입증 권장.
- **재현율 향상 여지**: 보수적 abstain로 R≈0.56 → λ/blocking/텍스트 인코더 튜닝, 더 긴 학습.

---

## 13. 부록 — 본 작업으로 추가/수정된 파일

추가: `scripts/gen_dataset_hill395.py`, `scripts/extract_hill395_results.py`, `scripts/plot_hill395_results.py`, `scripts/make_hill395_deck.py`, `configs/cita_full_hill395.yaml`, `tests/test_hill395_pipeline.py`, `data/battlefield_hill395_large/**`, `hill395_experiment_results/**`, 본 문서.
수정(가산적·하위호환): `src/citanet/model/featurize.py`(REL_NAMES +3), `src/citanet/model/citanet.py`(n_relations=len(REL_NAMES)).
```
```
(문서 끝)
