# QNN 모델 export 가이드 — 체크포인트에서 기기 실행까지

2026-08-27 기준 · ExecuTorch QNN backend · 검증 환경: Qwen3-4B / 8B, Galaxy S26 (Snapdragon 8 Elite Gen 5)

이 문서만 따라 하면 HF 체크포인트에서 기기에서 도는 QNN 컨텍스트 바이너리까지 갈 수 있다.
각 단계가 **무엇을 하는지**도 함께 설명한다 — 실패했을 때 어디를 봐야 할지 알아야 하기 때문이다.

4장(export 실행)과 6장(컨텍스트 추출)은 `scripts/export_target.sh`·`export_draft.sh`·
`extract_context.sh`가 그대로 감싸고 있다. 처음이라면 이 문서를 읽고, 두 번째부터는 스크립트를
쓰면 된다.

---

## 0. 전체 그림

```
HF 체크포인트
   │ ① 등록          레시피 + 모델 config를 ExecuTorch에 알린다
   ▼
llama.py --compile_only
   │ ② 모델 구성      LlamaModel 3개 (DECODE / PREFILL / CALIBRATE)
   │ ③ 가중치 적재    load_state_dict(assign=True), Linear → Conv2d 치환
   │ ④ prepare_pt2e   그래프에 관측자(observer) 삽입
   │ ⑤ 캘리브레이션   실제 프롬프트를 흘려 활성 범위 수집
   │ ⑥ convert_pt2e   관측 범위로 가중치/활성을 양자화
   │ ⑦ encoding 전파  CALIBRATE가 모은 스케일을 DECODE/PREFILL에 이식
   │ ⑧ lowering       to_edge_transform_and_lower_to_qnn → QNN 그래프
   ▼
hybrid_llama_qnn.pte          ← ExecuTorch 패키지 (샤드별 QNN 컨텍스트를 품고 있음)
   │ ⑨ 추출           dump_context_from_pte
   ▼
kv_forward_N.bin / prefill_forward_N.bin   (N = 샤드 번호)
   │ ⑩ 메타 추출      qnn-context-binary-utility → *_json.json
   │ ⑪ params.json    ★ 수작업 (export가 만들어 주지 않는다)
   ▼
기기로 push → llama-simple-qnn / llama-speculative-dflash-qnn
```

**중요**: `.pte` 하나에 샤드별 컨텍스트가 들어 있고, 각 컨텍스트에는 **그래프가 두 개**
(`kv_forward`, `prefill_forward`) 들어 있다. 러너는 `batch.n_tokens > 1`이면 prefill,
아니면 kv를 쓴다(`llm_decode_runner_multi_context.cpp:1646`).

---

## 1. 환경

```bash
cat > env.sh <<'EOF'   # 리포 루트의 env.sh.example 과 같은 내용
source ~/anaconda3/etc/profile.d/conda.sh && conda activate etqnn
export QNN_SDK_ROOT=/data/QNN_SDK/qairt/2.37.1.250807
export ANDROID_NDK_ROOT=$HOME/android-ndk-r26d
export EXECUTORCH_ROOT=$HOME/dev/llm/executorch
export PYTHONPATH=$EXECUTORCH_ROOT/..:$PYTHONPATH
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
EOF
source env.sh
```

### ⚠️ 함정 1 — 설치된 executorch가 리포보다 낡을 수 있다

`site-packages`의 wheel이 별도로 존재한다. `PYTHONPATH=$EXECUTORCH_ROOT/..`가 **리포를
이기게** 만드는 부분이다. 이게 없으면 리포에 등록한 새 모델이 `KeyError`로 죽거나, 더 나쁘게는
낡은 코드가 조용히 실행된다. 확인:

```bash
python -c "import executorch.examples.qualcomm.oss_scripts.llama as m; print(m.__file__)"
# → $EXECUTORCH_ROOT/... 이어야 한다 (site-packages가 아니라)
```

### ⚠️ 함정 2 — SoC에 맞는 런타임 skel

S26(SM8850)은 Hexagon **v81**이다. SDK 2.37.1에는 v81 skel이 없어서
Maven Central의 `qnn-runtime-2.44.0.aar`에서 가져왔고, `~/qairt/2.37.1-v81`에 심볼릭 미러를
만들어 쓴다. **기기의 5개 .so 버전이 전부 같아야 한다** — 섞으면
`Stub lib id mismatch`로 죽는다.

```bash
export QNN_SDK_ROOT=$HOME/qairt/2.37.1-v81   # export 시에만 이걸로 덮어쓴다
```

기기에 있어야 하는 것:
```
libQnnHtp.so  libQnnHtpPrepare.so  libQnnHtpV81Skel.so  libQnnHtpV81Stub.so  libQnnSystem.so
```

---

## 2. ① 모델 등록 — 파일 세 곳

> `qwen3-4b` / `qwen3-8b`를 그대로 쓸 거라면 이 절과 3절은 **손으로 할 필요가 없다**.
> `tools/qnn-export/apply.sh $EXECUTORCH_ROOT` 한 줄이면 등록과 불균등 샤딩, export 피크 RAM
> 수정까지 전부 들어간다. 아래는 그 패치가 무엇을 하는지, 그리고 **새 모델을 추가할 때**
> 어디를 고쳐야 하는지에 대한 설명이다.

새 모델이나 새 양자화 레시피를 쓰려면 **세 파일**을 모두 고쳐야 한다. 하나라도 빠지면 죽는다.

### (a) `examples/qualcomm/oss_scripts/llama/static_llm_quant_recipe.py` — 양자화 레시피

```python
class Qwen3_4BQuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()
        self.recipe = (
            QuantRecipe(self.default_quant_dtype, False,
                        act_observer=MinMaxObserver,
                        granularity=QuantGranularity.PER_TENSOR, verbose=verbose)
            # 모든 Linear가 1x1 Conv2d로 치환되므로 conv2d 하나로 전부 잡힌다
            .add_node_target({torch.ops.aten.conv2d.default},
                             QuantDtype.use_16a4w_block, False,
                             act_observer=MinMaxObserver,
                             granularity=QuantGranularity.PER_BLOCK,
                             extra_kwargs={"block_size": (1, 16, 1, 1)})
            # 특정 층만 올리고 싶으면 add_regex. 나중에 추가한 것이 이긴다.
            .add_regex({r"output\.conv"}, QuantDtype.use_16a8w, False,
                       act_observer=MinMaxObserver,
                       granularity=QuantGranularity.PER_CHANNEL)
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)  # KV를 8bit로
```

`add_regex`가 매칭하는 문자열은 노드의 `nn_module_stack` 경로이고, 실제 형태는
`layers.14.feed_forward.forward.__self__.w2_conv` 이다. **`forward.__self__.`가 들어간다** —
문서 예제(`layers.[7-13].feed_forward.w2_conv`)를 그대로 쓰면 매칭되지 않는다.

**우선순위**: 전략은 `(priority, 삽입순)` 역정렬이라 **나중에 추가한 것이 이긴다.**
따라서 `add_node_target`로 전체를 깔고 `add_regex`로 예외를 덮는 순서가 맞다.

### (b) `examples/qualcomm/oss_scripts/llama/__init__.py` — 모델 등록

```python
@register_llm_model("qwen3-8b")
@dataclass(init=False, frozen=True)
class Qwen3_8B(LLMModelConfig):
    repo_id: str = "Qwen/Qwen3-8B"
    params_path: str = os.path.join(BASE_DIR, "../../../models/qwen3/config/8b_config.json")
    convert_weights = convert_qwen3_weights
    transform_weight = False
    instruct_model = True
    num_sharding = 6                       # >1 이어야 샤딩이 켜진다
    shard_layers = [2, 10, 18, 26, 34]     # 경계 층 (없으면 균등 분할)
    masked_softmax = True
    seq_mse_candidates = 0
    r1 = False; r2 = False; r3 = True      # SpinQuant 회전
    quant_recipe = Qwen3_8BQuantRecipe
```

* **상속은 안 된다.** `@dataclass(init=False, frozen=True)`라 기존 클래스를 subclass하면
  동작하지 않는다. 통째로 복사할 것.
* `shard_layers`는 경계를 특정 층에 고정한다. 투기 디코딩용이면 **드래프트가 읽는 hidden이
  나오는 층**에 맞춰야 한다. Qwen3-4B/8B(36층, K=5 포착)는 `[2,10,18,26,34]`이고, 이는
  디코더 층 1/9/17/25/33의 출력에 해당한다.
* import 블록에도 레시피 클래스를 추가해야 한다(파일 상단 `from ...static_llm_quant_recipe import (...)`).

### (c) `examples/qualcomm/oss_scripts/llama/decoder_constants.py` — 채팅 템플릿 매핑

```python
DECODER_MODEL_VERSION = {
    ...
    "qwen3-8b": "qwen3",
}
```

**빠뜨리면 `register_llm_model`이 `KeyError`로 즉사한다.** 가장 흔한 실수다.

### 등록 확인

```bash
python -c "
from examples.qualcomm.oss_scripts.llama import SUPPORTED_LLM_MODELS as M
c = M['qwen3-8b']
print(c.quant_recipe.__name__, c.shard_layers, c.num_sharding)"
```

---

## 3. 캘리브레이션 데이터

`--calib_samples`가 받는 JSON 형식:

```json
[
  {"messages": [{"role": "user", "content": "Compose a travel blog post about Hawaii."}]},
  {"messages": [{"role": "user", "content": "def fibonacci(n):"}]}
]
```

* **런타임 분포와 맞출 것.** 상류 가이드도 명시한다. 우리는 Spec-Bench 13카테고리를 쓰므로
  거기서 39샘플을 뽑아 `calib_specbench.json`을 만들었다.
* `--calib_tasks wikitext`는 **Qwen3에서 쓰지 말 것** (lm_eval 경로가 깨진다). `--calib_samples`로 우회.
* 샘플 수는 export 메모리에 영향이 **없다**(측정: 39샘플 8분 동안 RSS 변화 1 GiB 미만).
  시간만 늘어난다 — 4B 기준 샘플당 약 12초.

---

## 4. Export 실행

```bash
source env.sh
export QNN_SDK_ROOT=$HOME/qairt/2.37.1-v81     # ← v81 skel

python examples/qualcomm/oss_scripts/llama/llama.py \
  --decoder_model qwen3-8b \
  --model_mode hybrid \
  --prefill_ar_len 32 \
  --max_context_len 1024 \
  --soc_model SM8850 \
  --build_folder build-android \
  --artifact ./out_8b \
  --compile_only \
  --calib_samples ./calib_specbench.json \
  --prompt "Once upon a time"
```

### 인자 설명

| 인자 | 의미 |
|---|---|
| `--decoder_model` | ②(b)에서 등록한 이름 |
| `--model_mode` | `hybrid`(kv+prefill 둘 다) / `kv` / `lookahead`. **투기 디코딩은 hybrid 필수** |
| `--prefill_ar_len` | prefill 그래프의 배치 폭. **컴파일 타임 상수다** — 아래 참고 |
| `--max_context_len` | 총 컨텍스트. KV 창은 `max_context_len - ar_len` |
| `--soc_model` | S26 = `SM8850` |
| `--compile_only` | 기기 없이 컴파일만 (기기 연결 시 생략 가능) |
| `--prompt` | **필수**. `--compile_only`여도 파서가 요구한다 |
| `--calib_samples` | ③의 JSON |
| `--eval_methods sqnr_eval` | 층별 양자화 오차 CSV를 원할 때만 |

### `--prefill_ar_len`을 고르는 법 — 투기 디코딩이면 중요하다

**prefill 그래프의 폭보다 넓은 배치는 그래프를 여러 번 호출한다.** 실측(Qwen3-4B, S26):

| 검증 배치 | ar_len=8 빌드 | ar_len=32 빌드 |
|---:|---:|---:|
| 8 | 59.8 ms | 61.0 ms |
| 16 | 119.8 ms (**2.0×**) | 63.3 ms (+3.7%) |
| 32 | 235.7 ms (**3.9×**) | 66.5 ms (+9%) |

* 체인 K=7이면 검증 배치가 8이므로 `--prefill_ar_len 8`이 딱 맞다.
* 트리나 더 긴 블록을 쓸 거면 **가장 넓은 배치에 맞춰** 굽는다. 좁은 배치를 넓은 그래프에
  넣는 비용은 거의 없다(59.8 → 61.0, +2%).

---

## 5. Export가 실제로 하는 일 (②~⑧)

`llama.py:190` `multi_modal_mgr.quantize(...)` → `llama.py:198` `.compile(...)` 두 줄이 전부지만
그 안에서:

### ② 모델 구성 — 인스턴스 3개

`HybridTextDecoder.__init__`이 `TextDecoder`를 셋 만든다: `DECODE`(ar_len 1),
`PREFILL`(ar_len N), `CALIBRATE`(전체 시퀀스, 스케일 수집 전용). 셋은 ar_len·배치·마스크만
다르고 가중치는 같다.

### ③ 가중치 적재 + Conv2d 치환

`torch.load(mmap=True)` → `load_state_dict(assign=True)` → 각 층의
`prepare_attention_conv()` / `prepare_feedforward_conv()`가 **`nn.Linear`를 1×1 `nn.Conv2d`로
바꾸고 원본을 `del` 한다.** HTP가 conv 경로에서 더 빠르기 때문. 이후 그래프에 Linear는 없다 —
레시피에서 `aten.conv2d.default` 하나로 전부 잡는 이유가 이것이다.

### ④⑤⑥ PTQ

`prepare_pt2e`(관측자 삽입) → 캘리브 데이터 forward(범위 수집) → `convert_pt2e`(양자화).
`convert_pt2e`는 int8 `_frozen_paramN` 버퍼를 새로 만든다.

### ⑦ encoding 전파

`_encoding_override`가 CALIBRATE가 모은 스케일/제로점을 DECODE·PREFILL 그래프에 이식한다.
**세 그래프가 같은 인코딩을 쓰게 하는 장치**다. 그래서 decode/prefill의 가중치 *값*은 산출물에
독립적으로 도달하지 못한다.

### ⑧ lowering

`to_edge_transform_and_lower_to_qnn`이 QNN 그래프로 낮추고 샤드별로 컴파일한다.
로그의 `Visiting: <node>` 줄이 이 단계다(4B 기준 약 76,000줄).

성공 신호:
```
Finish compile_only and save to ./out_8b
```

---

## 6. ⑨⑩ 컨텍스트 바이너리 추출

`.pte`는 ExecuTorch 패키지라 우리 C++ 러너가 직접 못 읽는다. 안의 QNN 컨텍스트를 꺼낸다.

```bash
python - <<'PY'
from executorch.backends.qualcomm.utils.utils import dump_context_from_pte
files = dump_context_from_pte("./out_8b/hybrid_llama_qnn.pte")
print(len(files), "binaries")   # 샤드 6개 × 그래프 2개 = 12
PY
```

`.pte`와 **같은 디렉터리**에 `kv_forward_0.bin` … `prefill_forward_5.bin`이 생긴다.

러너는 샤드당 파일 하나를 `forward_N.bin`이라는 이름으로 기대한다. 컨텍스트 하나에 두 그래프가
**모두** 들어 있으므로 `kv_forward_N.bin`만 쓰면 된다(`prefill_forward_N.bin`은 같은 내용):

```bash
Q=$QNN_SDK_ROOT/bin/x86_64-linux-clang
OUT=./ctx_8b_out; mkdir -p $OUT
for i in 0 1 2 3 4 5; do
  cp ./out_8b/kv_forward_$i.bin $OUT/forward_$i.bin
  LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang \
    $Q/qnn-context-binary-utility \
      --context_binary $OUT/forward_$i.bin \
      --json_file $OUT/forward_${i}_json.json
done
```

`*_json.json`은 러너가 텐서 이름·shape·양자화 파라미터를 읽는 곳이다. **없으면 안 된다.**

확인:
```bash
python -c "
import json
j=json.load(open('./ctx_8b_out/forward_0_json.json'))
print([g['info']['graphName'] for g in j['info']['graphs']])"
# → ['kv_forward', 'prefill_forward']
```

---

## 7. ⑪ `params.json` — ★ 수작업

**export는 이 파일을 만들어 주지 않는다.** 없으면 러너가 아무 출력도 없이 조용히 실패한다.
(실제로 여기서 한 번 막혔다: 파일 12개만 push되고 13번째가 없었다.)

```bash
cat > ./ctx_8b_out/params.json <<'EOF'
{
 "dim": 4096,
 "n_layers": 36,
 "n_heads": 32,
 "n_kv_heads": 8,
 "head_dim": 128,
 "vocab_size": 151936,
 "ffn_dim_multiplier": 1,
 "multiple_of": 256,
 "norm_eps": 1e-06,
 "rope_theta": 1000000.0,
 "use_scaled_rope": false
}
EOF
```

값은 `examples/models/qwen3/config/8b_config.json`에서 가져온다. `head_dim`은 **반드시 명시**할
것 — 생략하면 러너가 `dim / n_heads`로 유도하는데 Qwen3는 그게 틀리다(4B: 2560/32 = 80이지만
실제는 128).

---

## 8. 기기 배포 및 실행

```bash
D=R5KL20FMLCN; D2=/data/local/tmp/qnn6
adb -s $D shell "mkdir -p $D2/ctx_8b"
for f in ./ctx_8b_out/*; do adb -s $D push "$f" $D2/ctx_8b/; done
# 13개여야 한다: bin 6 + json 6 + params.json 1
adb -s $D shell "ls $D2/ctx_8b | wc -l"
```

AR 생성:
```bash
adb -s $D shell "cd $D2 && LD_LIBRARY_PATH=$D2 ADSP_LIBRARY_PATH=$D2 ./llama-simple-qnn \
  --qnn --multi-context --ctx-dir $D2/ctx_8b \
  --tokenizer /data/local/tmp/Qwen3-8B-q4_0-outq4.gguf \
  --params $D2/ctx_8b/params.json --log-level 0 \
  -n 96 -p 'The capital of France is'"
```

`--tokenizer`는 **어휘만** 쓰는 GGUF다(`vocab_only`). 가중치는 QNN 쪽에서 온다.

투기 디코딩:
```bash
adb -s $D shell "cd $D2 && LD_LIBRARY_PATH=$D2 ADSP_LIBRARY_PATH=$D2 ./llama-speculative-dflash-qnn \
  --qnn --multi-context --ctx-dir $D2/ctx_8b \
  --tokenizer /data/local/tmp/Qwen3-8B-q4_0-outq4.gguf \
  --params $D2/ctx_8b/params.json --log-level 0 \
  -md /data/local/tmp/Qwen3-8B-DFlash-b7-q4_0-emb.gguf -ngld 99 \
  -n 96 -c 1024 --temp 0.0 -f prompt.txt"
```

---

## 9. ★ 반드시 통과시켜야 하는 검사

**`--temp 0.0`에서 투기 디코딩의 토큰 열은 AR의 토큰 열과 완전히 같아야 한다.**
같지 않으면 그 빌드의 acceptance length는 아무 의미가 없다.

```bash
ar=$(adb -s $D shell "... QNN_TOKDBG=1 ./llama-simple-qnn ... -n 14 -p \"$P\"" \
     | grep -ao "TOKDBG\] [0-9]* [0-9]*" | awk '{printf "%s ", $3}')
sd=$(adb -s $D shell "... DQ_TOKDBG=1 ./llama-speculative-dflash-qnn ... -n 15 -p \"$P\"" \
     | grep -ao "TOKDBG\] [0-9]* [0-9]*" | awk 'NR>1{printf "%s ", $3}')
case "$sd" in "$ar"*) echo PASS;; *) echo FAIL;; esac
```

세 가지 주의:

1. **텍스트가 아니라 토큰 ID로 비교한다.** QNN 로거가 같은 stdout에 줄 단위로 끼어들어 토큰
   조각을 찢고, 로그 줄을 제거하면 **가짜 공백**이 남는다.
2. **길이가 다르다.** AR은 프리필에서 나온 첫 토큰을 루프 밖에서 찍어 인덱스가 1부터다.
   완전일치가 아니라 **접두사 일치**로 판정한다.
3. **양쪽이 비었을 때 PASS가 나오지 않게** 한다. 빈 문자열은 어떤 접두사에도 매치된다.

이 검사가 없어서 **위치 정보가 파괴된 타겟 위에서 하루치 양자화 실험을 돌린 적이 있다**
(`qnn_input_pos_bug_and_al_recovery.md`).

---

## 10. 메모리 — 큰 모델을 구울 때

Qwen3-8B hybrid는 251 GB 머신에서 **세 번 죽었다.** 원인과 해법은
`qnn_export_peak_memory.md`에 따로 정리했다. 요약:

| 모델 | 피크 RSS |
|---|---|
| Qwen3-4B (`prefill_ar_len=32`) | 102 GiB |
| Qwen3-8B (지렛대 적용 후) | 127 GiB |

* 피크는 **인스턴스 3개 생성 + `convert_pt2e`** 구간에서 난다. lowering이 아니다.
* 캘리브 샘플 수를 줄여도 **소용없다**.
* 공유 머신이라면 워치독을 붙일 것 — 가용 메모리가 하한 밑으로 가면 **내 프로세스를** 죽인다.
  폴링 5초(20초면 급락을 놓친다).

```bash
python llama.py ... & EXPID=$!
while kill -0 $EXPID 2>/dev/null; do
  avail=$(free -g | awk '/^Mem:/{print $7}')
  [ "$avail" -lt 12 ] && { kill -9 $EXPID; echo "watchdog killed"; break; }
  echo "$(date +%H:%M:%S) rss_kb=$(ps -o rss= -p $EXPID|tr -d ' ')" >> rss.log
  sleep 5
done
```

---

## 11. 자주 막히는 곳

| 증상 | 원인 |
|---|---|
| `KeyError: 'my-model'` | `decoder_constants.py`의 `DECODER_MODEL_VERSION`에 안 넣음 |
| 기기에서 아무 출력 없음 | `params.json` 없음 |
| `Stub lib id mismatch` | 기기의 QNN .so 5개 버전이 섞임 |
| 리포 수정이 반영 안 됨 | `PYTHONPATH`가 site-packages wheel에 짐 |
| `add_regex`가 매칭 안 됨 | `forward.__self__.`가 빠졌거나 `_0` vs `@0` 이름 규칙 차이 |
| head_dim이 이상함 | `params.json`에 `head_dim` 명시 안 함 |
| export가 OOM | 인스턴스 3개 fp32 — `qnn_export_peak_memory.md` 참고 |
| SD 출력이 AR과 다름 | §9 게이트 실패. AL 숫자를 믿지 말 것 |
| 트리/넓은 배치가 느림 | `--prefill_ar_len`이 배치보다 좁아 그래프를 여러 번 호출 |

---

관련 문서:
[`peak-memory.md`](peak-memory.md) — 피크 RAM 줄이기
[MobiSpec.cpp `docs/impl-specs/qnn_input_pos_bug_and_al_recovery.md`](https://github.com/mkjsym/MobiSpec.cpp/blob/main/docs/impl-specs/qnn_input_pos_bug_and_al_recovery.md) — 정합 게이트가 왜 필요한지
[MobiSpec.cpp `docs/impl-specs/qnn_dflash_driver_port.md`](https://github.com/mkjsym/MobiSpec.cpp/blob/main/docs/impl-specs/qnn_dflash_driver_port.md) — 러너 쪽 구조
[`quantization.md`](quantization.md) — 레시피를 그렇게 고른 이유
