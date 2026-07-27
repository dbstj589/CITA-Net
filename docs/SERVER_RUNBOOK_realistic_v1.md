# realistic-v1 본 학습 — RTX Pro 6000 서버 런북

realistic-v1 벤치마크에서 **제약 항 조건부 유효성** 본 학습(6변형 × 5시드 = 30런)을
RTX Pro 6000(Blackwell, 96GB) 서버에서 이어서 돌리기 위한 절차. 데이터·코드·설정·대조
결과는 이미 리포지토리에 있음. m1 난이도 게이트는 통과됨(dev 0.7135 / test 0.7377 ≤ 0.91).

가설(사전 등록): 각 제약 항은 자신이 겨냥한 불확실성이 실재할 때 유효하다(시간→시각
비동기, 상태→전이 빈번, 관계→개체 지문, 출처→오차 이질). 결과가 어떻게 나오든 그대로 보고.

---

## 0. 전제
- NVIDIA 드라이버가 **CUDA 12.8+** 지원(`nvidia-smi`에서 CUDA Version ≥ 12.8). Blackwell 필수.
- Python **3.11** (개발 환경과 동일), git.
- 디스크 ≥ 40GB 여유(데이터 0.75GB + feat 캐시 ~18GB + 체크포인트/로그).

## 1. Pull
```bash
git clone https://github.com/dbstj589/CITA-Net.git   # 최초
cd CITA-Net
# 또는 기존 클론이 있으면:  git pull origin main
git log --oneline -1        # d7f733a 이상인지 확인
```

## 2. 환경 (fresh venv + Blackwell용 torch)
```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e .            # citanet(editable) + numpy/PyYAML/matplotlib. torch도 딸려오나 아래로 교체
# ★★ Blackwell(sm_120)은 cu128 휠 필수. 기본 pip torch는 cpu/cu126라 커널이 없어 작동 안 함 ★★
pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
```
검증:
```bash
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0))
print("arch_list", torch.cuda.get_arch_list())   # 'sm_120'(또는 카드 sm)이 있어야 함
a=torch.randn(4096,4096,device="cuda"); (a@a).sum().item(); print("matmul OK")
PY
```
`cuda True` + 카드명 + `sm_120` 포함 + `matmul OK`가 나오면 준비 완료.
(참고: 30런 전체는 반드시 **같은 torch 버전**으로. 도중에 버전 바꾸지 말 것.)

## 3. 학습 전 위생 점검 (빠름)
데이터·게이트가 서버에서도 성립하는지 확인(데이터는 재생성 안 함, 리포의 것 사용):
```bash
python scripts/check_realistic_v1.py --splits dev test --dt 450 --r-err-floor 300
#   기대: GATE a PASS(무결성), GATE b survival ≈ 0.9926 (≥0.98)
python scripts/measure_realistic_v1.py | head -30
#   기대: 161 정체성/섹터, 출처별 오차·지연·상태전이 0.325·KG중첩 0.641 등
```
1에폭 스모크(GPU·OOM·수치 정상 확인, 몇 분):
```bash
CITANET_FEAT_CACHE="$PWD/runs/realistic_v1/feat_cache" \
python scripts/train_large.py --config configs/realistic_v1/m3_full.yaml \
  --seed 19521006 --run-dir /tmp/smoke --full-sectors --epochs 1 \
  --data-root data/realistic_v1 --ontology-dir data/realistic_v1/ontology
#   로그 첫 줄에 "== device=cuda train_max_pairs=None ==" 확인(=전체 후보 GPU). rm -rf /tmp/smoke
```
> 96GB이므로 **서브샘플링/CPU평가 우회 불필요** → 환경변수를 아무것도 안 걸면 amb160과
> **동일한 전체-후보 GPU 학습+평가**가 됨. (8GB GPU였다면 `CITANET_TRAIN_MAX_PAIRS`,
> `CITANET_EVAL_ON_CPU`를 켜야 하지만 서버에선 켜지 말 것 — 방법론 일치를 위해.)

## 4. 본 학습
러너는 순차·resumable(끝난 런은 `metrics.json` 보고 skip), `results.jsonl` 증분 저장,
featurize 캐시 자동 사용. **환경변수 불필요.**

### 4-1. 파일럿 (시드 1개 먼저 — 실측 시간·지표 확인)
```bash
bash scripts/run_realistic_v1_ablation.sh 19521006
#   6변형 순차: m3_full no_motion no_time no_state no_rel no_src
```
첫 런(m3_full) 끝나면 확인:
- `runs/realistic_v1/ablation/m3_full_seed19521006/metrics.json` 의 dev/test **8지표** 기록.
- **starvation 점검**: recall이 비정상적으로 눌리지 않았는지(num_slots=200 > 최대 정체성 161).
- 런당 실측 소요시간(로그 타임스탬프)로 남은 24런 총시간 추정.

### 4-2. 나머지 시드 (파일럿 정상 시)
```bash
for s in 19521007 19521008 19521009 19521010; do
  bash scripts/run_realistic_v1_ablation.sh "$s"
done
```
백그라운드로 돌리려면: `nohup bash -c 'for s in 19521006 19521007 19521008 19521009 19521010; do bash scripts/run_realistic_v1_ablation.sh "$s"; done' > runs/realistic_v1/ablation/all.log 2>&1 &`

### 진행 확인 / 재개
```bash
tail -f runs/realistic_v1/ablation/sweep_seed19521006.log
ls runs/realistic_v1/ablation/*/metrics.json | wc -l     # 30이면 완료
```
중단돼도 같은 명령 다시 실행하면 끝난 런은 건너뛰고 이어감(resumable).

## 5. 집계·판정 (30런 완료 후)
6변형(m3_full 기준) × 5시드, 검증 선택·시험 보고. 핵심 질문:
- (a) 각 항 제거 ΔF1(방향·5시드 부호일관성·paired t df=4)
- (b) amb160 대조(운동학만 뚜렷했음: ΔF1 −0.020, 5/5, p=0.013) — realistic-v1에서 시간/
  상태/관계/출처가 새로 유효한가
- (c) 겨냥 지표(시간→오병합·불가능전이, 상태→오병합, 관계→재현율·단편화, 출처→정밀도)
- (d) 운동학 재현 여부
- n=5 저검정력·다중비교(FWER) 미보정 명시, p 단독 금지. 검증/시험 부호 갈리면 "신뢰 불가".

집계 스크립트는 `scripts/aggregate_amb160_allterms.py`(검증됨)와 구조 동일 —
디렉터리를 `runs/realistic_v1/ablation/`로, 대조 기준을 amb160 결과로 바꿔 작성하면 됨
(30런 완료 후 요청 시 제공/작성). 산출: `results/realistic_v1_ablation/` (CSV·md·F1 막대).

## 6. 이 프로젝트 특유의 주의점
- **Blackwell = cu128** (2단계 ★). 기본 torch면 GPU 인식 실패.
- **환경변수 걸지 말 것**(서버): `CITANET_TRAIN_MAX_PAIRS`/`CITANET_EVAL_ON_CPU`는 8GB
  GPU 전용 우회. 서버에서 켜면 방법론이 amb160과 달라짐. `CITANET_FEAT_CACHE`는 러너가
  자동 설정(켜도 결과 불변, 속도만 향상).
- **기존 데이터·config·결과·frozen/amb160 무수정.** 신규 폴더(`runs/realistic_v1/`,
  `results/realistic_v1_ablation/`)만 사용.
- num_slots=200, blocking dt=450/floor=300, 40 epoch — `configs/realistic_v1/*.yaml`에 고정.
  건드리지 말 것(gate-b 커버리지·slot starvation 방지 조건).
- `data2/` 대용량 world-gt 파일 3종은 GitHub 제한으로 미포함(이 학습엔 불필요).
- 첫 런이 `runs/realistic_v1/feat_cache`에 섹터 캐시를 채움(~18GB). 이후 모든 런/에폭이 재사용.
```
