# Code-level method appendix — `mode: battle_1m` (hill395_world_gt_battle10d_1m)

Read-only audit of the actual generator. Every claim cites `file:line`; formulas/
constants are quoted verbatim from code. Where the code does not implement a thing
the prior method write-up implies, it is marked **"코드에 없음"**. Nothing was
modified or re-run to produce this document.

Files audited: `scripts/gen_world_gt.py` (`run_battle_1m`), `world/battle_sim.py`
(`simulate`), `world/scenario_battle10d.py` (schedule/Agent/day-night),
`world/terrain_real.py` (terrain, via `build_terrain_real`), config
`configs/world_gt_battle10d_1m.yaml`.

Config defaults actually in force (`configs/world_gt_battle10d_1m.yaml`):
`dt_seconds=145` (:11), `static_baseline_seconds=8400` (:12), `rok_fwd=30` (:15),
`rok_reserve=30` (:16), `ccf_echelons=7` (:17), `squads_per_echelon=65` (:18),
`fire_mission_minutes=6` (:19), `emit_state_samples=true` (:21), `global_seed=19521006`
(:6), `window_seconds=864000` (:10). Reify predicates (:35):
`[engagedWith, firesAt, occupies, withdrawsFrom, reinforces, movesToward, supports]`.

---

## 1) Combat-power `s` update

**Function** `attrition(att, dfn, day, coeff_att, coeff_dfn, t0, t1)` — `world/battle_sim.py:141‑156`.

Exact formulas (per attacker `a`, one randomly chosen defender `d`):

```
# battle_sim.py:143-148
d      = dfn[int(rng.integers(0, len(dfn)))] if dfn else None      # random defender
loss_a = coeff_att * (0.5 + 0.5 * (d.s if d else 0.5))             # attacker loss
a.s    = max(0.0, a.s - loss_a)
loss_d = coeff_dfn * (0.4 + 0.6 * a.s)                             # defender loss
d.s    = max(0.0, d.s - loss_d)
```

- **Applied ONCE per engagement**, not integrated over time. **"dt와의 곱: 코드에 없음"** — `dt`/`gap` never multiplies the loss. Each `attrition()` call decrements `s` a single time for the whole engagement window `[t0,t1]`.
- **Coefficient values** (attacker `coeff_att`, defender `coeff_dfn`):
  - Attack op: `coeff_att = 0.22 if fire else 0.13` (`:179`), then `coeff_att *= 0.6` if `op.get("shrink")` (`:180‑181`); `coeff_dfn = 0.14` (hard-coded at call `:184`).
  - Counterattack op: `attrition(occupiers, movers, day, 0.5, 0.12, …)` → `coeff_att=0.5`, `coeff_dfn=0.12` (`:264`).
  - `fire = fire_ok(t0 + 1500)` (`:176`), i.e. **not** inside a fog window.
- **참호/개활 계수**: the code has **no variables named trench/open**. The defender-vs-attacker asymmetry is only `coeff_dfn (0.14 / 0.12)` < `coeff_att (0.22 / 0.5)`; the factors `(0.5+0.5·s_def)` and `(0.4+0.6·s_att)` (`:144,147`) are the only "cover" shaping. → **"참호/개활 계수: 코드에 없음(명명 없음); coeff_att/coeff_dfn 비대칭으로만 표현"**.
- **포격(FireSupport)·전차 사격의 s 감소식**: **"코드에 없음"**. `FireSupport` and tank fire never modify `s`. `FireSupport` is emitted as events+`firesAt` relations only (`fire_missions` `:120‑133`); tank fire is a lone `firesAt` relation (`:178`). Artillery/mortar presence affects `s` *only indirectly* through the boolean `fire` selecting `coeff_att=0.22` vs `0.13` (`:179`).
- **회복(재편) 식 + 지연**:
  - Relief refit: `dfa.s = min(1.0, dfa.s + 0.45)` (`:228`); the value is recorded at `t = tt + 4*3600` (skf `:228`, keyframes `:225‑227` place the squad back home over `tt … tt+4h`). So refit **delay = 4 h**, **gain = +0.45** (discrete, not a rate).
  - Merge absorption: `a.s = min(1.0, a.s + b.s)` (`:237`); recorded at `t_eng + 3600` (`:237‑238`).
  - No continuous/decaying recovery function exists. → gradual regen **"코드에 없음"**; only the two discrete jumps above.

---

## 2) Threshold decisions

All comparisons are single-sided; **no hysteresis / dual thresholds anywhere**.

| Branch | Code (`battle_sim.py`) | Actual condition |
|---|---|---|
| Attacker **Destroyed** | `:197` `if s_end <= 0.25:` | `s ≤ 0.25` → `Destroyed`, `alive=False` |
| Attacker **Occupying** (takes crest) | `:201` `elif take and s_end > 0.4 and j < 3:` | `s > 0.4` **AND** op result=="take" **AND** squad index `j<3` |
| Attacker **Withdrawing** | `:204` `else:` | everything else (no explicit `0.25<s≤0.6` gate) |
| Defender attrit→Withdrawing (marker) | `:151` `if before > 0.5 >= d.s:` | crossing `s` from `>0.5` to `≤0.5` emits a `trans(...,"attrit","Holding","Withdrawing")` **record only** (no state/kf change) |
| Defender **Destroyed** | `:153` `if d.s <= 0.25 and d.alive:` | `s ≤ 0.25` → `Destroyed` |
| **Merge** (force only) | `:235` `wd = [a … if a.alive and a.s <= 0.6]`; pairs `:236` | eligible `s ≤ 0.6`; only under `op.get("force_merge")` (`:229`, D5). Emitted `reinforces` + `trans("merge")` (`:239‑240`) |
| **Relief / 교대** | `:218` `for r_i in range(2):` | **unconditional**: exactly 2 reserves relieve on every `take` (`:211`); worn defenders `worn = sorted(alive dfn by s)` (`:217`), the 2 lowest-`s` refit |

Notes:
- The "계속 임무 (s>0.6)" branch is **not coded** as such — a surviving attacker simply becomes `Occupying`(if take & s>0.4) or `Withdrawing`; a defender keeps `Holding` unless it crosses the markers above. → **"s>0.6 지속 분기: 코드에 없음"**.
- Merge threshold is **`≤0.6`** (`:235`), not `≤0.5`; relief is **unconditional** (not gated on `s≤0.5`); occupy needs **`s>0.4`** (not `>0.6`). See §9.

---

## 3) Keyframe scheduler

**Structure**: each agent holds a Python list `a.kf` of `(t, x, y, state)` and `a.s_kf` of `(t, s)`, initialised in `mk()` `battle_sim.py:76` (`a.kf=[(0.0,x,y,"Holding")]`, `a.s_kf=[(0.0,1.0)]`). Appends via `kf()` `:78‑79` (`a.kf.append((round(t,1), x, y, st))`) and `skf()` `:80‑81`. **Accumulation, never overwrite** — a re-committed echelon appends a new round-trip each op.

**Per-attack round-trip** (`:186‑210`), for squad `a`, `t_app=t0+2100`, `t_eng=t_app+2400` (`APPROACH=2100.0`, `ENGAGE=2400.0`, `:139`):
```
kf(a, t0,               home_x, home_y, "Approaching")       # start at staging (boundary)
kf(a, t0+0.4*APPROACH,  lane_x, lane_y, "Approaching")       # reach approach lane
# then by outcome:
Destroyed:  kf(a, t_app, crest, "Halted"); kf(a, t_eng, crest, "Destroyed")
Occupying:  kf(a, t_app, crest, "Occupying")
Withdrawing:kf(a, t_app, crest, "Halted"); kf(a, t_eng, crest, "Withdrawing")
            kf(a, t_eng+3600, home, "Withdrawing"); kf(a, t_eng+3660, home, "Holding")
```
- **Assembly / staging**: `home` = the boundary staging coord set at creation (`:97` `by+150+U(0,300)`). Withdrawn squads return to `home` (`:207`) → last sample off-map.
- **Approach-lane selection**: `axis = axes[op["axis"]]` where `axes = {0:"approach",1:"approach_w",2:"approach_e"}` (`:92`, chosen `:167`); lane target `sx,sy = lane ± U(-200,200)/U(-120,120)` (`:189`); crest target `tx,ty = crest_x±U(-250,250), crest_y+120±U(-80,80)` (`:190`).
- **Speed**: **not set in the scheduler** — keyframes carry only position/time; the *speed and any clamp* are computed later in the sampler (§ below). → "속도 지정: keyframe에는 없음(샘플러에서 산출)".
- **Halted 판정**: purely scripted — the string `"Halted"` is written at `t_app` for Destroyed/Withdrawing outcomes (`:198,206`). No kinematic test.
- **ROK reserve / counterattack round-trips**: relief `:221‑222` (`Moving`→`Occupying`→`Holding`), counterattack `:256‑263` (`Approaching`→`Occupying`, then `northtip` if `final` else back `Holding`), 9-bu fallback `:233‑234` (`Withdrawing`→`ninebu` `Holding`).

**v_max clamp** — **not in the scheduler**; it is in the sampler `run_battle_1m` `gen_world_gt.py:150‑157`:
```
gap  = max(1e-6, t - ts[idx-1])
de,dn = interp_xy(wps,t) - (e_prev,n_prev)
dist = hypot(de,dn); vmax = ontology.v_max(a.typ, st)
if dist > vmax*gap and dist > 0:
    sc_ = vmax*gap/dist; de,dn = de*sc_, dn*sc_; dist = vmax*gap     # clamp
speed = dist/gap
```
Static states are frozen (speed 0) at `:148‑149` (`if idx==0 or st in STATIC`). `MOVING={Approaching,Moving,Withdrawing}`, `STATIC={Emplaced,Holding,Halted,Occupying,Firing,Destroyed}` (`gen_world_gt.py`, top of `run_battle_1m`).

---

## 4) Proximity / engagement decision

- **공간 격자 해시: 코드에 없음.** There is **no spatial grid / hash / neighbour search**. Engagement pairing is a **uniform random draw**: `d = dfn[int(rng.integers(0, len(dfn)))]` (`battle_sim.py:143`), once per attacker in `attrition`. Cell size, neighbour radius → **"코드에 없음"**.
- **engagedWith 개시·종료**: emitted inside `attrition` `:149‑150` for the chosen `(a,d)` pair with interval `[t0, t1]`; at the attack call site `t0 = t_attack+1200`, `t1 = t_eng` (`:184`). Counterattack: interval `[t0+prep, rt]` (`:264`). Two directed edges (a→d and d→a) per pair. No distance test gates start/stop — the interval is fixed by the op timing.

---

## 5) Fire-mission generation

**Function** `fire_missions(shooters, lmk, a, b, prov, friendly)` — `battle_sim.py:120‑133`:
```
segs = clip_fog(a,b) if friendly else [(a,b)]                 # friendly clipped in fog
step = fmin*60.0 if fmin>0 else max(1.0, b-a)                 # fmin = fire_mission_minutes
for sa,sb in segs:
    t = sa
    while t < sb-1.0:
        te = min(sb, t+step)
        nev("FireSupport", t, te, lmk, [shooter ids], prov)   # one event per step
        for s in shooters: rel(s.eid,"firesAt",lmk,"landmark", t, te)
        t = te
```
- **간격**: fixed `step = fire_mission_minutes*60` s = **6*60 = 360 s** per mission (config `:19`). Not a distribution — deterministic back-to-back missions. → "발생 간격 분포: 균일 고정 360 s".
- **표적 선택**: passed by caller, always a landmark id (`lmid[...]`), never dynamic to enemy position:
  - background `:292` → `lmid["approach"]`, shooters alternate `rok_arty if dd%2==0 else ccf_arty` per day (`:290`);
  - attack support `:177` → the attack `lane`; counterattack prep `:244` → `lmid["crest"]`.
- **fire_mission_minutes 적용부**: `fmin = float(ec.get("fire_mission_minutes", 0.0))` (`:37`), used at `:125`. If 0 → one mission spanning the whole `[a,b]`.
- **안개 창 클리핑** — `clip_fog(a,b)` `battle_sim.py:42‑56` (applied only when `friendly=True`, `:124`):
```
segs=[(a,b)]
for (fa,fb) in fog:
    for (sa,sb) in segs:
        if fb<=sa or fa>=sb: keep (sa,sb)                     # no overlap
        else:
            if sa<fa: emit (sa, min(fa,sb))                   # left part
            if sb>fb: emit (max(fb,sa), sb)                   # right part
return [(sa,sb) … if sb-sa > 1.0]                             # drop <=1s slivers
```
CCF (enemy) fire is **not** clipped (`friendly=False` → `segs=[(a,b)]`). `fog` list is built in `scenario_battle10d.build_schedule` `:95` = `[(d(2)+6h, d(2)+12h)]` = `(194400, 216000)`.

---

## 6) RNG structure

- **Single global stream.** `run_battle_1m` creates `rng = np.random.default_rng(seed)` (`gen_world_gt.py`, near "terrain, tmeta = build_terrain_real(cfg)") and passes the same `rng` into `simulate(cfg, ontology, terrain, rng)`. Inside `simulate`, all draws (`rng.uniform`, `rng.integers`) share this one stream — **no sub-seeds, no per-agent/per-op streams**. → "서브시드: 코드에 없음(전역 1개)".
- **Terrain uses a separate sub-stream**: `build_terrain_real` seeds `np.random.default_rng(int(cfg["global_seed"]) + 777)` (`world/terrain_real.py`). So terrain noise is decoupled from the battle stream, but the battle simulation itself is one stream.
- **seed → reproduce path**: `global_seed` (config `:6`) → `main()` passes `seed` → `run_battle_1m` → `default_rng(seed)` → `simulate`. Deterministic given identical config (ops schedule is fixed data, not RNG). Terrain reproducible via `seed+777`.

---

## 7) Serialisation

**Per-sample trajectory triples — exactly 8** (`gen_world_gt.py:245‑252`), node `iri("sample", eid, "t{int(round(t))}")`:
1 `rdf:type stkg:StateSample` · 2 `stkg:sampleOf → entity` · 3 `stkg:atTime` · 4 `geo:easting` · 5 `geo:northing` · 6 `stkg:hasState` · 7 `stkg:speedMps` · 8 `stkg:combatPower`.
**elevation and heading are NOT serialised as triples** (parquet only) — that is how "row당 8트리플" is reached.

**Reified `Statement` triple sets**:
- Transition (7): `rdf:type Statement, aboutEntity, transitionKind, fromState, toState, combatPower, atTime` (`:212‑218`).
- Reified relation (6): `rdf:type Statement, rdf:subject, rdf:predicate, rdf:object, validFrom, validTo` (`:227‑232`) — only for predicates in the reify set (`:219‑220`).
- Reified event (5): `rdf:type Statement, aboutEvent, provenance, validFrom, validTo` (`:235‑239`).

**Other categories**: entity 4 triples (`type/objectType/affiliation/size`, `:173‑176`) + `partOf`,`participatesIn` counted under *relation* (`:178,180`); unit 4–5 (`:183‑186`); landmark 5 (`:189‑192`); event 4–5 (`:196‑200`); plain relation 1 each (`:205`).

**Day attribution** (`dayof = lambda t: min(n_days-1, max(0, int(t//DAY)))`, defined just after `n_days`):
- trajectory triples → sample day `dayof(row["t"])` (`:243`);
- relation/event → `dayof(interval[0])` (start day, `:195,204`); reified rel/event → start day (`:226,234`); transition → `dayof(tr["t"])` (`:211`);
- entity/unit/landmark → **`D0=0`** (`:170`, all in `d01`).
Intervals themselves are never split — only the *file placement* is by start day.

**De-dup + write** (`:254‑268`): a single global `seen` set; `tagged` is stable-sorted by category order `CATS=[entity,unit,landmark,event,relation,reification,trajectory]` (`:255‑257`) so the *first* occurrence (earliest category) is kept; duplicates counted in `dupes` (`:261`). Each day's kept lines written to `unified_stkg_d{d+1:02d}.nt` (`:265‑267`). `total = Σ day_counts` (`:268`).

---

## 8) Validator (each assert, how it counts) — `gen_world_gt.py:285‑382`

| # | Check (report key) | Implementation |
|---|---|---|
| 1 | `time_monotonic` | per entity, `sum(b<=a for consecutive t)` (`:293`) == 0 |
| 2 | `feasible_motion` | consecutive `hypot(Δe,Δn)/gap > v_max(typ,state)+1e-2` count (`:295‑298`) == 0 |
| 3 | `static_speed_zero` | rows with `state∈STATIC and |speed|>1e-2` (`:299‑301`) == 0 |
| 4 | `vocabulary` | traj states ∉ `state_names` + agent types ∉ `type_names` + rel preds ∉ `relation_names` (`:302‑303`) == 0 |
| 5 | `cross_reference` | rel subject ∉ entity ids, or object ∉ {entity/unit/landmark} pool (`:312‑318`) == 0 |
| 6 | `triple_count_match` | `nt_total (re-read files) == total (Σ day) == len(seen)` (`:377`) |
| 7 | `no_duplicate_triples` | independent re-read of all `d##.nt`, count repeats `d2` (`:305‑311`) == 0 |
| 8 | `op_trajectory_consistency` | each op `squad_id` has ≥2 rows in `[w0-1,w1+1]` incl ≥1 `MOVING` (`:334‑342`) |
| 9 | `daily_approaching_on_attack_days` | attack days with 0 `Approaching` samples (`:343‑348`) == [] |
| 10 | `fog_friendly_fire_overlap_sec` | friendly `firesAt` rels whose interval `overlap_len` with fog >0 (`:349‑353`) == 0 — **interval-length overlap**, `overlap_len = Σ max(0, min(b,fb)-max(a,fa))` (`:351‑352`) |
| 11 | `withdrawn_end_off_map` | assault agents whose last-t state=="Withdrawing" and `northing < by-100` (`:355‑357`) == [] |
| 12 | `boundary_entry` | all assault agents' first-t `northing ≥ by-400` (`:364`) |
| 13 | `coords_in_extent` | traj bbox within `[origin_e, origin_e+res*(nx-1)] × [origin_n, …ny-1]` (`:358‑361`) |
| 14 | `landmark_peak_match` | `|lm_crest.easting − (origin_e+ej*res)| < 1e-6` where `ej=argmax` (`:362‑363`) |
| 15 | `seam_position_continuity` | across each midnight, `hypot(Δ)/gap > v_max+1e-2` (`:329‑330`) == 0 |
| 16 | `seam_s_continuity` | across each midnight, `|Δs| > 0.15 + 0.001*gap` (`:331‑332`) == 0 |
| 17 | `total_1M_hard` | `0.9e6 ≤ total ≤ 1.1e6` (`:369`) |
| 18 | `no_duplicate_triples` is #7; **daily band** `band_ok[d] = 0.85·1e5 ≤ c ≤ 1.15·1e5` (`:366‑368`) is reported but **not** in the pass-gate |

**Seam logic** (`:320‑332`): for each `mt` in `midnights=[d*DAY for d in 1..n_days-1]` (`:133`, and a sample is *forced* at every midnight via `set(midnights)` union in the sampler `:136`), take last sample `≤mt` and first `≥mt`; check position feasibility (#15) and `|Δs|` (#16). `p4_ok = all(report[k]["pass"])` (`:… after :382`); daily band is **not** part of `p4_ok`.

---

## 9) Discrepancies vs the prior method write-up / docs

| Topic | Prior write-up / doc says | Code actually does | Cite |
|---|---|---|---|
| `dt_move` | 120 s (default 120, "필요시 60") | `dt_seconds: 145` in the active config | cfg `:11` |
| Engagement pairing | "근접쌍 계산은 공간 격자 해시로 O(N)" | **uniform random defender**, no spatial hash at all | `battle_sim.py:143` |
| Attrition & time | s decreases "dt와의 곱" | **single decrement per engagement**, no `dt` factor | `battle_sim.py:145,148` |
| Fire/tank on s | "교전·포격·전차 사격 각각" s식 | fire/tank **do not change s**; only the `fire` flag sets `coeff_att` | `battle_sim.py:176‑179` |
| Trench/open coefficients | named 참호/개활 계수 | only `coeff_att` / `coeff_dfn` (no trench/open naming) | `battle_sim.py:179,184,264` |
| Occupy threshold | "계속 임무 s>0.6" | occupy needs **`s>0.4`** (+ `j<3`, take); no `>0.6` branch | `battle_sim.py:201` |
| Merge threshold | "합류 둘 다 s≤0.5" | eligible **`s≤0.6`**, and only via `force_merge` (D5) | `battle_sim.py:235,229` |
| Relief threshold | "방어 s≤0.5 → 교대" | **unconditional** 2 reserves per crest loss | `battle_sim.py:218` |
| Emergent merge | "규칙 창발 합류 허용, emergent 태깅" | **only forced** merges exist; no emergent-merge rule / no `emergent` provenance | `battle_sim.py:229‑240` |
| Hysteresis | (implied stability) | none — all single-sided comparisons | `battle_sim.py:151,153,197,201,235` |
| Recovery | "s 서서히 회복 + 재편 지연" | discrete `+0.45` at +4 h (relief) / `+b.s` (merge); no gradual regen | `battle_sim.py:228,237` |
| Withdraw band | "철수 0.25~0.6" | withdraw is the **else** branch (any non-destroyed non-occupier), not a `0.25<s≤0.6` gate | `battle_sim.py:204` |
| Category balance | (1M via balanced knobs) | trajectory ≈ **91 %** of triples (knob ② dominant) | measured; emission `:240‑252` |
| RNG streams | (reproducibility) | single global `default_rng(seed)` for the sim (terrain uses `seed+777`) | `run_battle_1m`; `terrain_real.py` |

**Items marked "코드에 없음"**: spatial grid hash / cell size / neighbour radius (§4); `dt`-multiplied attrition (§1); explicit fire/tank `s` formulas (§1); trench/open coefficient names (§1); `s>0.6` "continue" branch (§2); emergent-merge rule and `emergent` provenance tag (§2); gradual `s` recovery function (§1); RNG sub-seeds (§6); keyframe-level speed field (§3).

---

*Generated by reading the code only; no generator was modified and no run was
launched. Combat power `s` is an ontology extension (not in `classes/states/
relations/sources.yaml`) — see `gt_manifest.json.combat_power_s`.*
