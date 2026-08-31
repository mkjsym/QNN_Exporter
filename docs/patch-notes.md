# 패치가 ExecuTorch에서 바꾸는 것

`patch/executorch-qnn-export.patch`는 상류 `88a5f60`(2026-08-24) 기준 21개 파일을 건드린다.
export는 이 리포가 아니라 ExecuTorch 소스 트리에서 돌기 때문에 이 방식이다 — 포크하지 않는
이유는 상류가 올라갔을 때 리베이스가 "충돌 해결"이지 "고고학"이 아니길 바라서다.

적용은 `./patch/apply.sh $EXECUTORCH_ROOT`, 되돌리기는 `-R`. 정확 적용 → 3-way 순서로 시도하고,
끝나면 등록이 먹었는지 확인하는 한 줄짜리 import 체크를 출력한다.

절차 전체(캘리브레이션 데이터, CLI, 컨텍스트 추출, 손으로 써야 하는 `params.json`)는
[`export-guide.md`](export-guide.md)에 있다. 이 문서는 **패치가 무엇을 바꾸는지**만 다룬다.

## 패치가 하는 일

### 1. 불균등 샤딩 (`shard_layers`)

`get_split_graph_pass()`는 층을 균등 분할만 할 수 있었다(`range(0, num_layers, n/shares)`).
DFlash 드래프트는 타깃의 `hidden_states[2,10,18,26,34]`를 읽는데, 이 텐서들을 **샤드 경계
텐서로 만들면** 그래프 출력을 추가하지 않고도 런타임이 그대로 읽을 수 있다. 36층을
2/8/8/8/8/2로 자르는 건 `range()`로 표현이 안 되므로 명시 리스트를 받도록 했다.

`model_sharding.py`(파라미터 추가) + `__init__.py`(`LLMModelConfig.shard_layers` 필드) +
`llm_wrappers.py`(TextDecoder / Modality 두 군데 배선).

### 2. `qwen3-4b` / `qwen3-8b` 등록

세 파일에 각각 등록해야 한다 — 레시피(`static_llm_quant_recipe.py`), 모델 설정
(`__init__.py`), 채팅 템플릿 매핑(`decoder_constants.py`). 여기에 `8b_config.json`이 새로
들어간다(상류에 4B까지만 있다).

레시피는 `Qwen3_1_7BQuantRecipe`를 그대로 물려받는다. 같은 아키텍처 계열(QK-norm, HF rope,
qkv bias 없음)이고 폭과 깊이만 다르다. 16a4w + block-16 LPBQ, LM head만 16a8w per-channel,
KV 캐시 8비트.

### 3. export 피크 RAM (`llm_wrappers.py`)

Qwen3-8B가 251 GB 머신에서 세 번 죽었던 것을 126.6 GiB로 내린 변경. 셋 다 산출물에는
영향이 없다 — 나가는 `.pte`는 바이트 단위로 같다.

| 변경 | 효과 |
|---|---|
| DECODE/PREFILL/CALIBRATE 3벌의 fp32 가중치를 1벌 공유 | **154.1 → 126.6 GiB** (8B가 사는 유일한 이유) |
| `convert_pt2e`가 지우지 못한 orphan fp32 상수 purge + `malloc_trim` | 실제 버그(8B 91.5 GiB가 죽은 채로 lowering까지 따라감), 단 **피크는 안 줄임** |
| `decode_qdq.pt2` 저장을 SQNR eval일 때만 | 디스크 22.8 GiB, `del calibration_prefill` 정상화 |

가운데 항목을 표에 남겨둔 건, 그게 "진짜 버그를 고쳤는데 목표 지표는 안 움직인" 사례이기
때문이다. 피크는 **무엇을 해제하느냐가 아니라 언제 해제하느냐**로 정해지고, purge는 피크를
만드는 `convert_pt2e` 뒤에 온다. 자세한 건 [`peak-memory.md`](peak-memory.md).

### 4. DFlash 드래프트 헤드 (`qwen3-4b-dflash` / `qwen3-8b-dflash`)

투기 디코딩 드래프트를 **스텝당 그래프 호출 1회**로 굽는다. 드래프트는 커밋된 토큰을 반영하는
inject 패스와 다음 블록을 제안하는 패스를 도는데, 둘 사이에 순차 의존이 없다 — 커밋 행은 매 층
같은 `fc` 출력에서 K/V만 뽑고 residual stream에 합류하지 않는다. 원 구현도 한 forward로 돈다.
NPU에서는 호출 수가 곧 비용이므로(고정비 = 가중치 스트리밍) 이 융합이 inject 단계 전체를 아낀다.

`model/dflash_draft.py`가 모델, `models/qwen3_dflash/`가 가중치 변환(`fc`를 층별 블록으로 분할 —
층마다 활성 스케일을 따로 갖게 하려고), `dump_target_hidden.py`가 캘리브레이션용 타깃 hidden 덤프.
**드래프트 export에는 `--dflash_hidden`이 필수다**: 활성값이 토큰 임베딩이 아니라 타깃의 hidden
이라, 토큰 프롬프트로 캘리브레이션하면 관찰자가 전혀 다른 분포를 학습한다.

이 과정에서 파이프라인의 "첫 입력은 토큰이다" 가정 일곱 군데를 역할 기반으로 바꿨다 —
토크나이저 출처(`tokenizer_repo_id`), 마스크 템플릿 인덱스, `max_seq_len`/`max_context_len` 혼용,
export 입력 평탄화, 콜레이터 패딩 폭, 마스크의 질의/키 축 분리(`ar_len`·`open_prefix`),
CALIBRATE 창 분할. KV 캐시만 `*args`에 남겨야 한다는 제약도 있다 — 탐지가 `args_<i>` 이름으로
층 인덱스를 읽는다.

### 5. `LLMModelConfig.__str__`의 리스트 포맷

`format_value()`가 int를 변환하지 않고 돌려주는데 그 결과를 `join()`에 넣고 있었다.
`shard_layers` 같은 숫자 리스트가 처음이라 이제야 터진다.

## 일부러 뺀 것

측정하고 버린 양자화 변형은 패치에 넣지 않았다 — 등록 7개(`qwen3-4b-w8`, `-kv16`,
`-kv16-mse`, `-kv16-hist`, `-sqnr`, `-r2`, `-lc`)와 레시피 4개(8비트 가중치, KV16, KV16+
histogram observer, SQNR 혼합정밀), 그리고 SpinQuant R2 부활 코드. 어느 것도 체크포인트에서
`ctx_4b`/`ctx_8b`로 가는 경로에 없고, 보고된 수치는 전부 그 두 빌드에서 나왔다.

결론만 남긴다:

* **KV 16비트는 졌다.** 13카테고리로 재보니 AL 1.925 vs 8비트 1.955, tps는 12% 낮다.
  5-프롬프트 표본에서 나왔던 "+15%"는 노이즈였다.
* **8비트 가중치·SQNR 혼합정밀·SpinQuant R2**는 AL을 유의하게 올리지 못했다. 당시 AL이 낮았던
  진짜 원인은 양자화가 아니라 `input_pos` 버그였다 (`qnn_input_pos_bug_and_al_recovery.md`).
* **SQNR 자체를 근거로 쓰지 말 것.** analyzer의 SQNR은 완전 양자화 그래프에서 end-to-end로
  재기 때문에 오차가 앞으로 전파되고, 순위가 깊이와 교란된다. 깊은 층의 낮은 점수는 그 층이
  약해서가 아니라 약한 층 뒤에 있어서일 수 있다.

`r1`/`r2` 설정 필드는 상류 것이라 그대로 있고, 상류처럼 `RuntimeError`로 막혀 있다.
