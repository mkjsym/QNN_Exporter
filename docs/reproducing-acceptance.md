# AL(수용 길이) 재현 가이드 — 모델 굽기부터 추론 CLI까지

이 문서 하나만 따라 하면 아래 표의 AL이 **소수 둘째 자리까지** 나와야 한다. 안 나오면 설정이
다른 것이고, 어디가 다른지는 §7의 진단표에서 증상으로 찾는다.

관련 문서: [`export-guide.md`](export-guide.md) (export 절차 상세) ·
[`quantization.md`](quantization.md) (레시피를 그렇게 고른 이유) ·
MobiSpec.cpp `examples/speculative-dflash-qnn-npu/README.md` (드라이버 옵션)

---

## 0. 먼저 알아야 할 것 세 가지

**① AL은 결정적이다.** `--temp 0.0`(greedy)에서 투기 디코딩은 타깃의 AR 출력과 토큰 단위로
같은 결과를 낸다. 같은 컨텍스트·같은 프롬프트·같은 기법 설정이면 **AL은 몇 번을 돌려도, 기기가
뜨겁든 차갑든, 소수 넷째 자리까지 같다.** 그러니 AL이 다르면 노이즈가 아니라 설정 차이다.
(반대로 `t/s`는 같은 설정에서도 세션마다 ±7%, 스윕 내 위치에 따라 5~6% 움직인다. §8 참고.)

**② 그런데 export는 결정적이지 않다.** 같은 레시피·같은 플래그로 두 번 구우면 `.pte` 크기는
같은데 sha256이 다르다(미해결, MobiSpec.cpp #81). **직접 구운 컨텍스트로는 아래 AL이 정확히
안 나올 수 있다.** 재현을 확인하려는 목적이면 먼저 §7의 게이트(AR 토큰열 일치)를 통과시키고,
그다음 AL이 ±0.05 안에 들면 정상으로 본다. 정확히 같은 값을 원하면 같은 컨텍스트 바이너리를
받아 써야 한다.

**③ AL의 정의가 두 개였다.** 예전 드라이버는 `mean_accept_len`에 **커밋된** 토큰을 셌고
(마지막 스텝이 `-n` 상한을 넘겨 만든 것 포함), 지금은 **전달된** 토큰을 센다. 차이는 1.6~3.1%다.
낡은 바이너리를 쓰면 아래 표보다 그만큼 높게 나온다. 지금 드라이버는 둘 다 찍는다:
```
mean_accept_len=4.6154 committed_len=4.7308
```
표와 비교할 것은 **앞의 값**이다.

---

## 1. 재현 대상 — 기준 AL

Qwen3-4B 타깃(`qwen3-4b-sqnr`, `prefill_ar_len=32`) · Spec-Bench 13카테고리 평균 ·
`-n 128` · greedy · 챗 템플릿 적용.

| 기법 | 예산/설정 | AL (NPU 드래프트) | AL (Adreno 드래프트) |
|---|---|---:|---:|
| DARTree | `DARTREE=31 DARW=8` (NPU) / `DARTREE=15 DARW=8` (GPU) | **4.67** | **4.39** |
| PCTree | `PCTREE=23` (NPU) / `PCTREE=31` (GPU), `PCK=4 PCTOPN=256 PCEXACT=0` | **4.64** | **4.62** |
| TileMenu | `TILEMENU=15.8,56.7,57.9` (NPU) / `17.3,59.3,60.3` (GPU) | **4.28** | **4.14** |
| DDTree | `DDTREE=15 DDK=2` | **4.02** | **3.95** |
| DSpark | 체인, markov GGUF 필요 | **3.76** | **3.70** |
| DFlash | 체인 (block 7) | **3.46** | **3.33** |

단일 카테고리로 빠르게 확인하려면 `sbprompts3/4_0.txt`(coding):
NPU 드래프트 PCTree `B=23`에서 **AL 5.1200, n_steps 25** 가 나온다.

---

## 2. 굽기 ① — 타깃

```bash
source env.sh                                   # EXECUTORCH_ROOT, QNN_SDK_ROOT, PYTHONPATH
./patch/apply.sh                                # ExecuTorch 88a5f60 기준
./scripts/export_target.sh qwen3-4b-sqnr /work/art_q4b 32 1024
./scripts/extract_context.sh /work/art_q4b /work/ctx_ar32 6
./scripts/make_params.py --hf Qwen/Qwen3-4B --out /work/ctx_ar32/params.json
```

AL에 직접 영향을 주는 선택지는 셋뿐이다.

| | 값 | 틀리면 |
|---|---|---|
| 레시피 | `qwen3-4b-sqnr` | 다른 레시피는 타깃이 **다른 토큰을 낸다**. 드래프트가 맞춰야 할 정답이 바뀌므로 AL이 통째로 달라진다 |
| `prefill_ar_len` | **32** | 31노드 트리+루트가 한 번의 그래프 호출. 64로 구우면 AL은 같고 속도만 4~18% 손해 |
| 캘리브 데이터 | `data/calib_specbench.json` | **챗 템플릿이 적용된** 39개 프롬프트. 런타임은 항상 템플릿을 쓰므로, 원문 프롬프트로 캘리브하면 관찰자가 실제와 다른 분포에서 범위를 잡는다 |

## 3. 굽기 ② — 드래프트 (여기가 AL을 망치는 곳)

```bash
# (a) 타깃의 hidden state를 먼저 덤프한다 -- 드래프트의 입력은 토큰이 아니라 이것이다
python $EXECUTORCH_ROOT/examples/qualcomm/oss_scripts/llama/dump_target_hidden.py \
  --target Qwen/Qwen3-4B --calib data/calib_specbench.json --out /work/hidden_4b.pt
# (b) 굽기
./scripts/export_draft.sh qwen3-4b-dflash /work/art_dflash /work/hidden_4b.pt
./scripts/extract_context.sh /work/art_dflash /work/ctx_dflash 1
./scripts/make_params.py --hf Qwen/Qwen3-4B --out /work/ctx_dflash/params.json --draft --n-layers 5
# (c) 임베딩 테이블은 그래프 밖, 호스트 fp16
./scripts/dump_draft_embd.py --ckpt dflash4b.pth --out /work/draft4b_embd_f16.bin
```

**세 가지가 전부 필요하고, 하나라도 빠지면 AL이 무너진다.**

- **hidden 덤프 없이 굽지 말 것.** 드래프트의 `fc`는 타깃의 hidden을 받는다. 토큰 임베딩으로
  캘리브하면 절대 보지 못할 범위를 학습한다 — 타깃 33층 hidden은 최대 16384까지 가는데 평균
  |x|는 4.5다.
- **`MarginMinMaxObserver`(min/max ×2)가 켜져 있어야 한다.** `qwen3-4b-dflash` 레시피에 이미
  들어 있다. 이게 없으면 활성 클리핑으로 **기기 AL이 1.03**이 된다. 잘리는 원소는 전체의
  0.002%지만 그것들이 어텐션을 결정하는 극단값이다.
- **임베딩은 fp16 별도 파일**이어야 한다. 타깃 GGUF에 있는 Q4_0 테이블을 그냥 쓰면 고정 입력
  8토큰에서 top-1이 3/8까지 떨어진다.

드래프트 종류: `qwen3-4b-dflash`(DFlash 체인/DDTree/TileMenu용),
`qwen3-4b-dspark`(DSpark/PCTree/DARTree용 — markov 헤드가 필요한 기법들).

## 4. `params.json` — export 산출물이 아니다

**손으로 만들어야 하고, 없으면 러너가 조용히 아무것도 출력하지 않는다.** 에러도 안 난다.
`make_params.py`가 만들어 주지만 한 필드는 반드시 확인할 것:

```json
{ "dim": 2560, "n_layers": 36, "n_heads": 32, "n_kv_heads": 8,
  "head_dim": 128,          ← dim/n_heads = 80 이 아니다. Qwen3는 독립적으로 정한다
  "vocab_size": 151936, "norm_eps": 1e-06, "rope_theta": 1000000.0 }
```
드래프트용에는 `n_inject: 16, n_target_layers: 5, block_size: 7, mask_token_id: 151669`가 더 붙는다.
`block_size`가 8이면 체인 AL이 어긋난다 — 블록은 8칸이지만 앵커 1 + 초안 7이다.

## 5. 기기 배치

```
/data/local/tmp/qnn6/
├── ctx_ar32/       forward_{0..5}.bin  forward_{0..5}_json.json  params.json   ← 13개
├── ctx_dflash4b/   forward_0.bin  forward_0_json.json  params.json
├── ctx_dspark4b/   forward_0.bin  forward_0_json.json  params.json
├── draft4b_embd_f16.bin
└── libQnnHtp.so libQnnHtpPrepare.so libQnnHtpV81Skel.so libQnnHtpV81Stub.so libQnnSystem.so
/data/local/tmp/
├── Qwen3-4B-q4_0-outq4.gguf            ← 토크나이저 (vocab만 쓴다)
├── Qwen3-4B-DSpark-b7-q4_0-emb.gguf    ← markov 헤드 (DSpark/PCTree/DARTree)
└── sbprompts3/{0..12}_0.txt            ← Spec-Bench 13카테고리
```

⚠️ **기기의 QNN `.so` 5개는 버전이 전부 같아야 한다.** 섞이면 `Stub lib id mismatch`로 죽는다.
S26(SM8850)은 Hexagon v81이고 SDK 2.37.1에는 v81 skel이 없어 Maven AAR에서 가져와야 한다
([`export-guide.md`](export-guide.md) §1 함정 2).

⚠️ **바이너리는 매번 새 이름으로 push할 것.** 같은 이름을 재사용하다 낡은 바이너리로 측정한
사고가 여러 번 있었다. 그리고 **정적 빌드**(`build-android`)를 쓸 것 — 공유 라이브러리
빌드(`build-android-hex`, 1.7 MB)를 올리면 `.so` 없이는 `CANNOT LINK EXECUTABLE`로 죽는다.

## 6. 추론 CLI

두 드라이버가 있다. 타깃은 둘 다 NPU고 드래프트가 어디서 도는지가 다르다.

### 6.1 드래프트도 NPU (`llama-speculative-dflash-qnn-npu`)

```bash
D2=/data/local/tmp/qnn6
TOK=/data/local/tmp/Qwen3-4B-q4_0-outq4.gguf
MK=/data/local/tmp/Qwen3-4B-DSpark-b7-q4_0-emb.gguf

adb shell "cd $D2 && LD_LIBRARY_PATH=$D2 ADSP_LIBRARY_PATH=$D2 \
  DFLASH_CHAT=1 \
  DFLASH_DRAFT_EMBD=$D2/draft4b_embd_f16.bin \
  DFLASH_DRAFT_CTX=$D2/ctx_dspark4b DFLASH_DRAFT_PARAMS=$D2/ctx_dspark4b/params.json \
  DFLASH_MARKOV_GGUF=$MK \
  DFLASH_PCTREE=23 DFLASH_PCK=4 DFLASH_PCTOPN=256 DFLASH_PCEXACT=0 \
  ./llama-speculative-dflash-qnn-npu --qnn --multi-context --ctx-dir $D2/ctx_ar32 \
  --tokenizer $TOK --params $D2/ctx_ar32/params.json \
  --log-level 1 -n 128 -c 1024 --temp 0.0 -f /data/local/tmp/sbprompts3/4_0.txt"
```

### 6.2 드래프트는 Adreno (`llama-speculative-dflash-qnn`)

`DFLASH_DRAFT_CTX`/`_PARAMS`/`_EMBD` 대신 `-md <draft.gguf> -ngld 99`를 준다.

### 6.3 기법별 설정 — 기본값에 기대지 말 것

| 기법 | 드래프트 컨텍스트 | 환경변수 |
|---|---|---|
| DFlash | dflash | (없음) |
| DSpark | dspark | `DFLASH_MARKOV_GGUF=…` |
| DDTree | dflash | `DFLASH_DDTREE=15 DFLASH_DDK=2` |
| TileMenu | dflash | `DFLASH_TILEMENU=D,cv8,cv16` |
| PCTree | dspark | `DFLASH_PCTREE=23 DFLASH_PCK=4 DFLASH_PCTOPN=256 DFLASH_PCEXACT=0` + markov |
| DARTree | dspark | `DFLASH_DARTREE=31 DFLASH_DARW=8` + markov |

- **`DFLASH_CHAT=1`은 필수다.** 캘리브를 템플릿 적용 프롬프트로 했으므로 런타임도 같아야 한다.
- **`DFLASH_PCEXACT=0`을 빼지 말 것.** exact 확장으로 돌면 AL은 같은데 호스트가 스텝당
  757 ms를 쓴다(정상은 6~24 ms). 결과가 "느린 백엔드"처럼 보이지 설정 실수처럼 안 보인다.
- **`DFLASH_TILEMENU`의 세 수는 실측 스텝 비용**(draft, verify@8, verify@16)이다. 백엔드마다
  다르고, 남의 값을 그대로 쓰면 예산 선택이 틀어진다. `per step:` 출력으로 재적합할 것.
- `DFLASH_DARW`의 기본값은 논문값 12지만 이 타깃에서는 **8**이 재현값이다.

## 7. 맞는지 확인하는 법 — 정합 게이트부터

AL을 보기 전에 **같은 프롬프트의 AR 출력과 토큰열이 같은지** 먼저 확인한다. greedy 투기 디코딩은
정확히 AR과 같은 토큰을 내야 하므로, 다르면 그 시점에서 이미 틀린 것이다. (이 게이트가 없어서
`input_pos` 미기록 버그가 오래 숨어 있었다 — prefill 그래프의 모든 행이 position 0으로 RoPE되고
있었고 AR 기준선까지 같이 망가져서 비교로는 안 보였다.)

```bash
# AR
adb shell "cd $D2 && LD_LIBRARY_PATH=$D2 ADSP_LIBRARY_PATH=$D2 QNN_CHAT=1 ./llama-simple-qnn \
  --qnn --multi-context --ctx-dir $D2/ctx_ar32 --tokenizer $TOK --params $D2/ctx_ar32/params.json \
  -n 128 -c 1024 -f /data/local/tmp/sbprompts3/4_0.txt" > ar.txt
# SD (위 6.1) > sd.txt  → 생성 토큰열이 같아야 한다
```

### AL이 안 맞을 때 — 증상별 진단

| 증상 | 원인 | 확인/조치 |
|---|---|---|
| AL ≈ **1.00**, 출력은 정상 | 드래프트 프리필 누락 | 드래프트가 프롬프트를 안 읽었다. 첫 스텝부터 컨텍스트 없이 제안한다 |
| AL ≈ **1.03** | 활성 **클리핑** (마진 관찰자 없음) | 레시피가 `MarginMinMaxObserver`인지 확인. `HistogramObserver`는 더 나쁘다(범위를 *좁힌다*) |
| AL ≈ **1.04**, 속도 0.66×AR | 드래프트를 **w4a8**로 구움 | 활성 8비트는 이 드래프트에 너무 거칠다. 16a8w 또는 w4a16을 쓸 것 |
| AL 1.97 (DARTree, 기대 4.67) | `DARK`/`DARBETA` 기본값 이탈 | `DARK=64 DARBETA=-0.2`가 맞다. `DARW=8` |
| AL 4.32 (PCTree, 기대 4.64) | `PCTOPN`이 16 | 256으로 |
| AL이 표보다 **1.6~3.1% 높음** | 낡은 바이너리(커밋 기준 AL) | `committed_len`이 같이 찍히는지 확인. 없으면 재빌드 |
| 체인 AL이 미묘하게 다름 | `block_size` 7 vs 8 | `params.json`의 `block_size`는 **7**(앵커 1 + 초안 7 = 블록 8칸) |
| **출력이 아예 없음** | `params.json` 부재 | 에러 없이 조용히 끝난다. 가장 먼저 볼 것 |
| 출력이 그럴듯한데 AR과 다름 | `head_dim` 유도값 사용 | Qwen3-4B는 128이지 80이 아니다 |
| 짧은 카테고리에서 AL이 2.50 / 3.33 같은 분수 | EOS로 조기 종료 | 정상이다. 11토큰 생성이면 3~5스텝뿐이라 평균이 거칠다. 카테고리별 비교 시 주의 |
| top-1은 맞는데 AL이 낮음 | 임베딩을 GGUF Q4_0으로 | fp16 별도 파일을 쓸 것 |
| 전부 맞는데 ±0.05 차이 | **직접 구운 컨텍스트** | export가 결정적이지 않다(§0-②). 정합 게이트를 통과하면 정상 범위 |

## 8. AL은 맞는데 속도가 안 맞을 때

**정상이다.** AL과 달리 `t/s`는 기기 상태를 탄다. 같은 부팅 세션에서 **동일 설정**을 스윕 앞뒤에
놓고 재면:

| 설정 | AL 앞 | AL 뒤 | step 앞 | step 뒤 |
|---|---:|---:|---:|---:|
| PCTree B=31 | 4.62 | 4.62 | 82.4 ms | 87.4 ms (+6.0%) |
| DARTree B=15 | 4.39 | 4.39 | 79.3 ms | 82.1 ms (+3.5%) |

AL은 소수 둘째 자리까지 같은데 스텝은 6%까지 벌어진다. 세션이 다르면 AR 기준선도 5%쯤 움직여
`×AR`은 ±0.2~0.4를 달고 다닌다. 그래서:

- 비교는 **같은 세션 안에서** 할 것.
- **AR 기준선을 스윕 앞에서만 재지 말 것.** 차가울 때의 AR로 뜨거울 때의 SD를 나누면 ×AR이
  계통적으로 낮게 나온다. 앞뒤로 두 번 재서 드리프트를 괄호로 묶는 게 낫다.
- 런 사이 온도 게이트는 **런 도중을 못 본다**. 5초 간격으로 재면 실제 피크는 100~105 °C인데
  런 사이 대표값은 37 °C로 나온다.
- 이 기기는 **한 부팅 세션 안에서 드래프트 백엔드를 번갈아 쓰면 재부팅한다.** 백엔드별로 세션을
  나눌 것.

## 9. 체크리스트

굽기
- [ ] 캘리브 프롬프트가 **챗 템플릿 적용본**인가
- [ ] 드래프트를 **타깃 hidden 덤프**로 캘리브했는가
- [ ] 드래프트 레시피에 **마진 관찰자(×2)** 가 있는가
- [ ] `prefill_ar_len=32`인가
- [ ] `params.json`을 만들었고 **`head_dim`이 명시**돼 있는가
- [ ] 드래프트 `params.json`의 `block_size`가 **7**인가
- [ ] 임베딩을 **fp16 별도 파일**로 덤프했는가

돌리기
- [ ] `DFLASH_CHAT=1`
- [ ] 기법별 환경변수를 **전부** 줬는가 (특히 `PCEXACT=0`, `PCTOPN=256`)
- [ ] `--temp 0.0`
- [ ] markov 기법이면 `DFLASH_MARKOV_GGUF`
- [ ] **정적 빌드** 바이너리를, **새 이름으로** push했는가
- [ ] `mean_accept_len` 옆에 `committed_len`이 찍히는가 (낡은 바이너리 판별)
- [ ] AR 토큰열과 일치하는가
