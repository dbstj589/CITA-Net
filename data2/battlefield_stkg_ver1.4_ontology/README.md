# battlefield_stkg_ver1.4 기반 CTA 온톨로지

`data/battlefield_stkg_ver1.4.ttl` 의 **객체 상하위 클래스 계층(bf:)** 위에,
CITA-Net STKG 관측 계층(**cta:**)을 얹은 온톨로지 모듈 묶음입니다.
상태 11종·관계 13종은 `data/battlefield_hill395_large/ontology/` 에서 쓰인 것을
**그대로** 사용하며, **상태 전이 규칙**과 **객체별 허용 상태**만 ver1.4 클래스
계층에 맞춰 변형했습니다.

## 파일 구성

| 파일 | 내용 |
|------|------|
| `battlefield_stkg_cta.ttl` | 최상위 온톨로지. 아래 3개 모듈 + ver1.4 를 `owl:imports` |
| `states.ttl` | 관측 상태 11종, 정적여부(`cta:isStatic`), 상태 전이 규칙(`cta:StateTransition`) |
| `classes.ttl` | bf: 클래스별 운동 프로파일 7종(허용 상태·기본 상태·상태별 v_max·전이 예외) |
| `relations.ttl` | 전술 관계 13종(대칭/이벤트 플래그, domain/range, ver1.4 속성 정렬) |

> ver1.4 원본(`bf:`)은 이 폴더로 복사하지 않고 `owl:imports` 로 참조합니다.
> 원본 파일 `data/battlefield_stkg_ver1.4.ttl` 는 수정하지 않았습니다.

## 네임스페이스

- `bf:`  `https://example.org/onto/battlefield-stkg#`  — ver1.4 클래스/속성(재사용)
- `cta:` `https://example.org/stkg/cta#`               — 본 관측 계층(신규)

## 상태 11종 (hill395 그대로)

동적(`isStatic=false`): **Moving, Approaching, Withdrawing, Engaging, Unknown**
정적(`isStatic=true`): **Halted, Holding, Occupying, Emplaced, Firing, Destroyed**

## 관계 13종 (hill395 그대로)

partOf, follows, near(대칭), firesAt, engagedWith(대칭), emplacedAt,
movesToward, supports, screens, occupies, withdrawsFrom, reinforces,
participatesIn(이벤트). — 대칭은 `near`, `engagedWith` 둘뿐.

## 운동 프로파일 7종 (ver1.4 클래스에 맞춰 변형)

| 프로파일 | base v_max | 기본 상태 | 허용 상태 특징 | 부착 bf: 클래스 |
|----------|-----------:|-----------|----------------|-----------------|
| Foot        | 2.0  | Holding   | 전(全) 상태 허용 | Personnel, Soldier, Squad, Unit |
| Tracked     | 12.0 | Moving    | Emplaced 없음 | Tank, ArmoredVehicle |
| Wheeled     | 14.0 | Moving    | 전투 상태 최소 | Truck, GroundVehicle |
| Towed       | 0.0  | Emplaced  | Emplaced/Firing/Halted만 (자력이동 불가) | Mortar, AntiAircraftGun, AntiTankWeapon, Artillery |
| SelfPropelled | 8.0 | Emplaced | 이동 후 사격 | SelfPropelledArtillery, MultipleRocketLauncher |
| Airborne    | 50.0 | Moving    | 정적 상태 없음 | AirVehicle, UAV |
| Static      | 0.0  | Occupying | 위치 불변 | Facility, MilitaryPosition, CommandPost, LogisticsFacility, GeographicFeature, Sensor |

**상속 규칙(소비자):** 어떤 bf: 클래스에 `cta:hasKinematicProfile` 이 직접
없으면 `rdfs:subClassOf` 를 타고 올라가 가장 가까운 조상의 프로파일을 상속.
예) `bf:T55`(→Tank) 는 TrackedProfile, `bf:SelfPropelledArtillery`(Artillery
하위지만) 는 자체 지정으로 SelfPropelledProfile 을 사용.

### v_max 규칙 (hill395 `_v_max_for` 계승)

- towed/정적(base=0): 모든 상태 0
- 그 외: 빠른 상태(Moving/Approaching/Withdrawing/Unknown/Destroyed)=base,
  Engaging=min(base, 2.0), 나머지 정적 상태=0.5

## 상태 전이 규칙 (hill395 골격 + 클래스별 예외)

- 미지정 전이 = **0.85**, 자기전이 = **1.0**, `eps=0.01`
- 전술 비합리 전이(low_after): 철수→점령 0.20, 진지→이동 0.30, 사격→이동 0.40 등
- 전역: 파괴→활성 0.01, 파괴→미상 0.30, 미상↔any 0.80, 활성→파괴 0.60
- **클래스별 예외**(`cta:profileTransitionOverride`): 견인화기는
  진지→이동/접근 = **0.02** (거의 불가) — TowedProfile 에 부착

점수식: `b_state = gamma · log(C[from][to] + eps)`

## 검증

```bash
.venv/Scripts/python.exe -c "import rdflib; [rdflib.Graph().parse(f, format='turtle') for f in \
 ['states.ttl','classes.ttl','relations.ttl','battlefield_stkg_cta.ttl']]"
```
