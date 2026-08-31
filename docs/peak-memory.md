# QNN export의 피크 RAM 줄이기 — Qwen3-8B를 251 GB 머신에 태운 과정

2026-08-26/27 · ExecuTorch QNN backend · Qwen3-4B / 8B hybrid export

Qwen3-8B hybrid export가 이 머신에서 **세 번 죽었다.** 네 번째에 성공했다. 이 문서는 무엇이
메모리를 잡고 있었는지, 어떤 가설이 틀렸는지, 무엇이 실제로 통했는지를 남긴다.

핵심만 먼저: **피크를 만든 건 fp32 가중치 3벌을 동시에 들고 있던 것이고, 그걸 1벌 공유로 바꾼
것이 유일하게 효과가 있었다.** 그 전에 시도한 세 가지는 전부 헛짚었다.

---

## 결과

| 시도 | 적용한 것 | 피크 RSS | 결과 |
|---|---|---|---|
| 1 | 없음 | (가용 3 GB에서 사망) | ✗ |
| 2 | 없음 | ~153 GB 소비 후 미완 | ✗ |
| 3 | orphan fp32 purge + `malloc_trim` + qdq 게이트 | **154.1 GiB** | ✗ |
| 4 | **+ fp32 가중치 3벌 → 1벌 공유** | **126.6 GiB** | **✓ 49분** |

4B 대조군: 지렛대 없음 **102.0 GiB**, purge만 적용 **101.8 GiB** (변화 없음).

---

## 왜 이렇게 많이 먹었나

`HybridTextDecoder.__init__`(`llm_wrappers.py:941-960`)은 `TextDecoder`를 **세 개** 만든다:

| 인스턴스 | 용도 |
|---|---|
| `Mode.DECODE` | 토큰 1개씩 처리하는 kv_forward 그래프 |
| `Mode.PREFILL` | ar_len개를 한 번에 처리하는 prefill_forward 그래프 |
| `Mode.CALIBRATE` | 양자화 스케일 수집 전용 (`_encoding_override`로 앞의 둘에 전파) |

세 인스턴스는 **`ar_len` / `max_batch_size` / `use_kv_cache` / 마스크만 다르다**
(`base_component.py:93-117`이 모드별로 바꾸는 것의 전부다). 그런데 각자
`_prepare_model()` 끝의 `.to(fp32)`에서 **자기 몫의 fp32 가중치를 따로 만들었다.**

```
Qwen3-4B: 16.4338 GiB × 3 = 49.3 GiB
Qwen3-8B: 30.52   GiB × 3 = 91.6 GiB
```

8B RSS 곡선이 이걸 그대로 보여준다:

```
 0 → 4.4분     0 → 94 GiB     인스턴스 3개 생성 ← 여기가 기저
 4.4 → 17.5분  97.6 GiB 평탄   캘리브레이션 (메모리 안 늘어남)
18.6분         131.5 GiB 급등  convert_pt2e 순간 피크
               154.1 GiB 사망  (lowering은 시작도 못 함: Visiting 0줄)
```

---

## 통한 것 — fp32 가중치 1벌 공유

`_prepare_model()`의 `.to(dtype_override)` 직후, 첫 인스턴스의 state_dict를 클래스 변수에
캐시하고 이후 인스턴스는 `load_state_dict(..., assign=True)`로 **같은 텐서 객체를 가리키게** 한다.

```python
can_share = (
    not self.config.r1
    and not self.config.r2
    and not getattr(self.config, "seq_mse_candidates", 0)
    and not getattr(self.control_args, "qat", False)
)
if can_share:
    shared = type(self)._shared_fp32_weights
    if shared is None:
        type(self)._shared_fp32_weights = dict(decoder.state_dict())
    elif set(shared) == set(decoder.state_dict()):
        decoder.load_state_dict(shared, strict=True, assign=True)
        gc.collect()
```

**왜 안전한가**: decode/prefill의 가중치 *값*은 애초에 산출물에 도달하지 못한다 —
`_encoding_override`가 그 둘의 `get_attr` 타깃을 calibration_prefill의 텐서로 다시 가리킨다.
즉 이 지렛대는 **버려질 사본을 안 만드는 것**이지 계산을 바꾸지 않는다.

**가드가 반드시 필요하다.** 아래 셋은 가중치를 인스턴스별로 *제자리에서* 고친다:

| 조건 | 이유 |
|---|---|
| `r1` / `r2` | SpinQuant 회전이 가중치를 in-place로 덮어쓴다 → 공유하면 **3번 적용**된다 |
| `seq_mse_candidates > 0` | 인스턴스별 스케일을 탐색해 되쓴다 |
| `qat` | 학습한다 |

실제 등록 모델 기준:

```
qwen3-4b           r1=F r2=F seq_mse=0   → 공유 O
qwen3-4b-sqnr      r1=F r2=F seq_mse=0   → 공유 O
qwen3-8b           r1=F r2=F seq_mse=0   → 공유 O
qwen3-4b-r2        r1=F r2=T seq_mse=0   → 공유 X
qwen3-4b-kv16-mse  r1=F r2=F seq_mse=50  → 공유 X
```

로그로 확인할 것:
```
[Mode.DECODE]    cached fp32 weights for later instances
[Mode.PREFILL]   reusing the cached fp32 weights
[Mode.CALIBRATE] reusing the cached fp32 weights
```

---

## 통하지 않은 것 (기록으로 남긴다)

### ✗ 캘리브레이션 샘플 줄이기

가장 먼저 의심했고 **완전히 틀렸다.** 39샘플 8분 동안 RSS가 1 GiB도 안 늘었다(위 곡선의 평탄
구간). 캘리브레이션은 이미 만들어진 그래프에 데이터를 흘릴 뿐이다.

### ✗ lowering 최적화

"`qnn_preprocess`가 범인"이라고 두 번째로 의심했는데, 8B는 **lowering에 도달하지도 못하고**
죽었다(`Visiting:` 0줄). 4B에서 53 → 102 GiB로 자라는 구간이 lowering이라 오해했다.

### △ orphan fp32 파라미터 purge — 진짜 버그지만 피크는 안 줄인다

`convert_pt2e`는 가중치를 int8 `_frozen_paramN`으로 접은 뒤 원본 fp32를 지우려 하는데,
가드가 `if hasattr(gm, node.target)`이고(`torchao/quantization/pt2e/constant_fold.py:354`)
`ExportedProgram.module()` 이후 타깃이 `"layers.0.attention.wq_conv.weight"` 같은 **점 표기
FQN**이라 `hasattr`이 항상 False다. **`delattr`이 한 번도 실행되지 않는다.**

이건 실제 버그이고 고칠 값어치가 있다(4B 49.3 GiB / 8B 91.5 GiB가 죽은 채로 lowering까지
따라간다). 하지만 **피크는 안 줄었다** — purge가 `convert_pt2e` *다음*에 오는데 피크는
`convert_pt2e` *중*에 발생하기 때문이다. 순서가 맞지 않는다.

```
4B: 지렛대 없음 102.0 GiB → purge 적용 101.8 GiB   (3 × 16.44 GiB 해제했는데 변화 없음)
```

### △ `malloc_trim(0)`

Python이 객체를 해제해도 glibc가 arena를 OS에 반환하지 않아 RSS가 안 내려간다. PyTorch CPU
텐서는 `posix_memalign`을 거치므로 `malloc_trim`이 적용되고, 이건 **맞는 처방이다** —
하지만 위와 같은 이유로 피크에는 영향이 없었다(8B 시도 3: 154.1 GiB, 여전히 사망).

`gc.collect()`만으로는 부족하다는 사실 자체는 기억할 값어치가 있다.

### ✓ `decode_qdq.pt2` 게이트

SQNR eval에서만 읽는 파일인데 **항상** 쓰고 있었다. 4B에서 **22.76 GiB**, 8B에서 그 이상.
게다가 로컬 `qdq_ep`가 calibration_prefill 그래프 전체를 붙들어 아래의
`del self.calibration_prefill`을 **무력화**한다. `--eval_methods`에 `sqnr_eval`이 있을 때만
쓰도록 게이트했다. 피크에 미치는 영향은 작지만 디스크 22 GB와 `del`의 정상화는 확실한 이득이다.

---

## 배운 것

1. **피크는 "무엇을 해제하느냐"가 아니라 "언제 해제하느냐"로 정해진다.** 죽은 fp32를 49 GiB
   해제해도 그 해제가 피크 다음이면 아무 소용이 없다.
2. **RSS 곡선을 먼저 그린다.** 5초 간격 샘플링 한 번이 세 번의 잘못된 가설을 즉시 잘라냈을 것이다:
   ```bash
   while kill -0 $PID 2>/dev/null; do
     echo "$(date +%H:%M:%S) rss_kb=$(ps -o rss= -p $PID | tr -d ' ')" >> rss.log
     sleep 5
   done
   ```
   평탄 구간은 범인이 아니고, 계단이 지는 곳이 범인이다.
3. **공유 머신에서는 워치독을 붙인다.** 가용 메모리가 하한 밑으로 가면 **내 프로세스를** 죽여야
   한다. OOM killer는 가장 큰 프로세스를 고르지만, 그게 남의 작업일 수도 있다. 폴링은 5초로 —
   20초 간격에서는 12 GB → 3 GB 급락을 놓쳤다.
4. **`del`이 실제로 해제하는지 확인한다.** `del self.calibration_prefill` 뒤로 RSS가 평탄하면
   누군가 참조를 붙들고 있다는 뜻이다. 여기서는 `qdq_ep`였다.

---

## 남은 카드 (아직 필요 없었다)

* **meta device 생성** — `LlamaModel(args)`가 kaiming 초기화로 fp32 한 벌을 만들고
  `load_state_dict(assign=True)`가 곧바로 버린다. 순수한 낭비지만, `static_llama.py:680-698`이
  `__init__` 안에서 `freqs_cos`/`freqs_sin`을 **계산**하므로 통짜 `with torch.device("meta")`는
  모델을 깨뜨린다. 선별적으로 해야 한다.
* **샤드별 순차 lowering** — `to_edge_transform_and_lower_to_qnn`이 파티션 6개를 다 들고
  낮춘다. 프로세스를 나눠 하나씩 낮추면 lowering 항이 줄지만, `.pte` 패키징을 바꿔야 한다.
* **임베딩 re-tie** — 4B는 `.to(fp32)`가 tied 임베딩을 끊어 1.449 GiB를 낭비한다.
  **8B는 애초에 untied라 값이 0이다.**

---

## 수정한 파일

`examples/qualcomm/oss_scripts/llama/wrappers/llm_wrappers.py` 한 곳뿐이다.
`torch`나 `torchao` 쪽은 건드리지 않았다 — 환경 업그레이드 때 사라지고,
`constant_fold.py` 사본이 `torch/_inductor`와 `torchao/quantization/pt2e` 양쪽에 byte-identical로
존재해서 엉뚱한 파일을 고치기 쉽다. 실제로 실행되는 건 **torchao 쪽**이다.

| 위치 | 내용 |
|---|---|
| `_prepare_model()`, `.to(dtype_override)` 직후 | fp32 가중치 공유 + 가드 |
| `TextDecoder` 클래스 변수 | `_shared_fp32_weights = None` |
| `quantize()`, `del graph_module` 직후 | orphan fp32 purge + `malloc_trim(0)` |
| `HybridTextDecoder.quantize()`, qdq 저장부 | `sqnr_eval`일 때만 실행 + `del qdq_ep` |

관련: [`export-guide.md`](export-guide.md),
[MobiSpec.cpp `qnn_input_pos_bug_and_al_recovery.md`](https://github.com/mkjsym/MobiSpec.cpp/blob/main/docs/impl-specs/qnn_input_pos_bug_and_al_recovery.md)
