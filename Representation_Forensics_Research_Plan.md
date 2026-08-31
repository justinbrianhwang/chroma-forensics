# Model Autopsy: Base-Free Forensic Attribution of Compositional Post-Training Histories

> **연구계획서 초안**  
> 연구 분야: AI Security · Model Provenance · Representation Analysis · Topological Data Analysis  
> 문서 상태: 연구 착수 및 파일럿 실험 설계용

---

## 0. 연구 요약

본 연구는 **최종 모델 checkpoint만 관찰했을 때, 해당 모델이 어떤 post-training intervention을 어떤 순서로 거쳤는지 어디까지 식별할 수 있는가**를 조사한다.

기존 연구는 주로 다음 중 하나에 집중한다.

- 알려진 base model과 fine-tuned model의 activation 또는 logit 차이 분석
- 특정 intervention 하나의 흔적 탐지
- 부모 모델 또는 model family 식별
- 동일 학습 과정에서 representation dynamics 추적
- 특정 공격·unlearning·alignment가 남긴 topology 분석

그러나 실제 모델 공급망에서는 base checkpoint, training log, sibling model, 정확한 데이터셋 및 hyperparameter가 공개되지 않는 경우가 많다. 또한 하나의 모델은 `SFT → preference optimization → unlearning → quantization`처럼 여러 intervention을 연속해서 거칠 수 있다.

따라서 본 연구의 핵심 문제를 다음과 같이 정의한다.

> **Given an unknown final model and a fixed public probe set, can we infer an ordered set of post-training interventions without access to its base checkpoint, metadata, training logs, or sibling models?**

연구의 중심은 단순한 closed-set classification이 아니다. 다음 다섯 가지를 함께 다룬다.

1. **Base-free attribution**: 원본 checkpoint 없이 분석한다.
2. **Compositional history**: 단일 intervention이 아니라 intervention sequence를 복원한다.
3. **Cross-root generalization**: 학습에서 보지 않은 root checkpoint와 model family에 일반화한다.
4. **Open-set attribution**: 알려지지 않은 intervention 또는 조합에 대해 기권할 수 있어야 한다.
5. **Laundering robustness**: 흔적을 지우기 위한 추가 fine-tuning, pruning, quantization, merging 등에 얼마나 견디는지 평가한다.

TDA는 연구의 목적 자체가 아니라, **training-history signature가 root, seed, probe domain 및 laundering을 넘어 유지되는지 측정하기 위한 structural descriptor 중 하나**로 사용한다.

---

# 1. 연구 배경과 문제의식

## 1.1 왜 training history가 중요한가

공개 또는 제3자 제공 모델을 사용할 때 사용자는 일반적으로 다음 사실을 완전히 검증할 수 없다.

- 어떤 base model에서 파생됐는가
- 어떤 데이터와 objective로 fine-tuning됐는가
- preference optimization이나 safety tuning이 적용됐는가
- backdoor 또는 malicious fine-tuning을 거쳤는가
- unlearning 또는 safety removal이 실제로 수행됐는가
- distillation, pruning, quantization, model merging이 적용됐는가
- 공개된 model card의 training history가 사실인가

이는 model supply-chain security, provenance, compliance, unlearning verification 및 third-party model auditing 문제와 직접 연결된다.

## 1.2 기존 접근의 한계

기존 activation-difference 또는 logit-difference 기반 방법은 강력하지만, 상당수가 다음 조건을 요구한다.

- 정확한 base checkpoint가 알려져 있음
- base와 target의 architecture 및 hidden dimension이 호환됨
- 비교 가능한 sibling model이 존재함
- 하나의 intervention만 가정함
- 알려진 operation만 존재하는 closed-set 환경임
- detector를 회피하려는 adaptive adversary를 고려하지 않음

실제 forensic setting에서는 이 조건들이 충족되지 않을 수 있다.

## 1.3 핵심 연구 전환

본 연구는 다음 질문에서 출발하지 않는다.

> “TDA로 clean, fine-tuned, distilled, poisoned, unlearned 모델을 분류할 수 있는가?”

이 문제는 class 정의가 상호배타적이지 않고, root·seed·architecture leakage에 취약하며, 관련 선행연구와 중복될 가능성이 높다.

대신 다음 질문을 다룬다.

> **“모델의 과거에 대해 관찰 가능한 증거는 무엇이며, 어떤 history는 식별 가능하고 어떤 history는 원천적으로 구별할 수 없는가?”**

즉, detector 성능뿐 아니라 **forensic identifiability의 한계**를 함께 분석한다.

---

# 2. 선행연구 포화도와 연구 공백

## 2.1 이미 점유된 주장

다음 수준의 주장은 기존 연구와 직접적으로 겹치므로 본 연구의 novelty claim으로 사용하지 않는다.

| 주장 | 포화도 판단 |
|---|---|
| Fine-tuning objective가 activation에 흔적을 남긴다 | 점유됨 |
| Base–target activation/logit difference로 fine-tuning 내용을 추론한다 | 점유됨 |
| Training order 또는 recency가 activation에 인코딩된다 | 부분적으로 점유됨 |
| Unlearning이 representation 또는 output에 흔적을 남긴다 | 점유됨 |
| Poisoning/backdoor가 latent topology를 변화시킨다 | 점유됨 |
| Persistent homology로 서로 다른 representation 상태를 비교한다 | 점유됨 |
| Representation 또는 weight로 부모 모델·lineage를 식별한다 | 점유됨 |
| Weight-space meta-classification으로 training setup을 예측한다 | 점유됨 |

## 2.2 상대적으로 열린 연구 공백

본 연구는 다음 공백의 결합을 목표로 한다.

| 연구 공백 | 본 연구의 대응 |
|---|---|
| Base checkpoint가 없는 intervention attribution | Target model 단독 fingerprint 구축 |
| 복수 intervention이 적용된 최종 모델 | Ordered compositional history prediction |
| Root/seed/model family 변화에 대한 취약성 | Grouped cross-root·cross-family split |
| Unknown operation을 기존 class로 강제 분류 | Open-set detection과 calibrated abstention |
| 추가 post-processing으로 forensic trace 제거 | Laundering benchmark 및 adaptive evasion |
| 단순 탐지 성능 중심 평가 | Observational equivalence와 identifiability limit 분석 |
| TDA의 실질적 필요성 불분명 | Cross-root 및 laundering 조건에서 incremental value 검증 |

## 2.3 연구의 예상 위치

본 연구는 다음 세 분야의 교차점에 위치한다.

- **Model provenance**: 모델의 기원과 파생 관계
- **Model forensics**: 모델이 거친 조작 및 학습 과정의 사후 추론
- **Representation security**: 내부 표현에 남는 구조적 흔적과 회피 가능성

---

# 3. 연구 목표

## 3.1 최종 목표

알 수 없는 최종 모델에 대해, 제한된 observation interface와 공개 probe set만 사용하여 다음을 추정하는 forensic framework와 benchmark를 구축한다.

1. 어떤 intervention이 적용되었는가
2. 여러 intervention이 적용됐다면 어떤 순서였는가
3. 추론 결과의 confidence는 어느 정도인가
4. 현재 evidence만으로 식별 불가능하거나 unknown인 경우 기권할 수 있는가
5. 흔적을 지우려는 후처리에도 결과가 유지되는가
6. 어떤 조건에서 두 training history가 observationally indistinguishable한가

## 3.2 실용적 목표

- Model card 또는 공급자 주장에 대한 보조적 검증 수단 제공
- Model supply-chain incident 조사에 활용 가능한 forensic evidence 생성
- Unlearning·safety modification·backdoor removal 주장 검증
- 향후 training-history transparency 표준을 위한 평가 틀 제안

## 3.3 비목표

본 연구는 다음을 주장하지 않는다.

- 최종 모델만으로 실제 training history를 암호학적으로 증명한다.
- 모든 model family와 모든 intervention을 완벽히 구분한다.
- TDA가 모든 descriptor보다 항상 우월하다.
- 관찰된 fingerprint만으로 법적 provenance를 확정한다.

출력은 어디까지나 **명시된 model population과 observation interface 아래에서의 calibrated forensic evidence**이다.

---

# 4. 연구 질문

## RQ1. Base-free identifiability

Base checkpoint, training log, sibling model 없이도 단일 post-training intervention을 식별할 수 있는가?

## RQ2. Compositionality

서로 다른 intervention이 연속 적용된 모델에서 operation의 존재 여부뿐 아니라 적용 순서를 복원할 수 있는가?

## RQ3. Cross-root generalization

훈련에서 보지 않은 root checkpoint, initialization seed, dataset 및 model family에서도 history signature가 유지되는가?

## RQ4. Observation interface

Black-box output, logit, activation, attention, weight 중 어떤 정보가 training history의 어떤 부분을 식별하는 데 필요한가?

## RQ5. Open-set attribution

학습하지 않은 intervention 또는 조합이 들어왔을 때 기존 class로 오분류하지 않고 unknown으로 기권할 수 있는가?

## RQ6. Laundering resistance

추가 fine-tuning, quantization, pruning, merging, secondary distillation 및 detector-aware optimization 이후에도 forensic trace가 남는가?

## RQ7. TDA의 추가 가치

Persistent-homology 기반 descriptor가 spectrum, CKA, intrinsic dimension, logit statistics보다 cross-root 또는 laundering 조건에서 독립적인 정보를 제공하는가?

## RQ8. Identifiability limit

서로 다른 history가 동일하거나 매우 유사한 observable을 생성할 때, 어떤 history pair가 실질적으로 구별 불가능한가?

---

# 5. 연구 가설

## H1. Intervention signature

Post-training intervention은 predictive performance를 matching한 이후에도 representation과 parameter space에 통계적으로 검출 가능한 structural signature를 남긴다.

## H2. Compositional non-additivity

여러 intervention이 조합되면 fingerprint는 단순 합으로만 설명되지 않으며, 일부 descriptor는 operation order에 민감하다.

## H3. Root dominance

통제하지 않을 경우 root checkpoint와 initialization seed의 fingerprint가 intervention signal보다 강하게 나타난다.

## H4. Interface hierarchy

Black-box evidence만으로 식별 가능한 history와 white-box activation 또는 weight 접근이 필요한 history가 구분된다.

## H5. Topological robustness

TDA descriptor는 in-distribution 정확도에서 반드시 최고가 아니더라도, 일부 operation에 대해 cross-root 또는 laundering 이후 상대적으로 안정적인 separation을 제공한다.

## H6. Fundamental ambiguity

일부 서로 다른 history는 선택한 observation interface와 probe distribution 아래에서 observationally equivalent하며, 정확한 attribution이 불가능하다.

---

# 6. 정식 문제 정의

## 6.1 Training history

Root model을 \(M_0\), post-training operation을 \(g_i\)라 하자.

```text
h = (g_1, g_2, ..., g_T)
M_h = g_T ∘ ... ∘ g_2 ∘ g_1(M_0)
```

여기서 \(h\)는 모델의 post-training history이고, \(M_h\)는 최종 모델이다.

## 6.2 Observation interface

감사자가 사용할 수 있는 정보 수준을 \(I\)로 정의한다.

```text
I_black-box     : generated outputs 또는 predicted labels
I_logit         : probability 및 logits
I_activation    : layer-wise hidden states
I_attention     : attention maps 또는 attention differences
I_weight        : model weights, adapter weights, spectral statistics
I_reference     : 알려진 base model과 target model의 차이
```

본 연구의 핵심 setting은 `I_reference` 없이 수행되는 **reference-free setting**이다.

## 6.3 Probe set

공개 probe set을 \(P = \{x_1, ..., x_n\}\)라 하고, observation interface에서 얻은 정보를 feature extractor \(\Phi_I\)로 변환한다.

```text
z = Φ_I(M_h, P)
```

Forensic estimator \(f\)는 다음을 출력한다.

```text
f(z) → posterior over histories, predicted sequence, uncertainty, abstention decision
```

## 6.4 Identifiability

두 history \(h_a, h_b\)가 선택한 interface와 probe distribution 아래에서 구별 불가능하면 다음과 같은 observational equivalence relation을 정의한다.

```text
h_a ~_(I,P) h_b
```

실험적으로는 다음 조건을 함께 사용한다.

- fingerprint distribution 간 거리
- classifier distinguishability
- permutation test
- confidence interval
- equivalence margin
- cross-root consistency

본 연구는 단순 정확도뿐 아니라 **어떤 history들이 같은 equivalence class에 들어가는가**를 보고한다.

---

# 7. 위협 모델

## 7.1 감사자 능력

감사자는 다음 중 일부 또는 전부를 사용할 수 있다.

- Target model checkpoint
- 공개된 tokenizer 및 architecture
- 고정된 public probe set
- 제한된 model query budget
- activation 또는 weight에 대한 white-box 접근

다음 정보는 기본적으로 제공되지 않는다.

- 정확한 base checkpoint
- training data
- training logs
- random seed
- optimizer state
- intervention hyperparameter
- sibling checkpoint

## 7.2 비적응적 공급자

공급자는 history를 공개하지 않지만 detector를 적극적으로 회피하지는 않는다.

## 7.3 적응적 공급자

공급자는 forensic detector의 존재를 알고 다음과 같은 laundering을 수행할 수 있다.

- benign continued fine-tuning
- unrelated-data fine-tuning
- quantization
- pruning
- model merging
- secondary distillation
- parameter permutation 또는 function-preserving transformation
- representation matching regularization
- detector-aware adversarial fine-tuning

## 7.4 보안적 한계

- 모델이 arbitrary architecture conversion을 거친 경우 직접 feature alignment가 어려울 수 있다.
- 완전히 black-box인 경우 functionally equivalent model 간 provenance는 원천적으로 어려울 수 있다.
- 데이터셋 provenance와 intervention provenance는 동일한 문제가 아니다.

---

# 8. Intervention taxonomy

## 8.1 1차 실험에 포함할 operation

서로 다른 추상 수준의 label을 섞지 않고, 구체적인 post-training operation 단위로 정의한다.

| 계열 | Operation | 구현 예시 |
|---|---|---|
| Control | Null / continued benign training | 동일 데이터 분포에서 짧은 추가 학습 |
| Adaptation | Supervised fine-tuning | Full SFT 또는 LoRA-SFT |
| Alignment | Preference optimization | DPO 중심, 필요 시 PPO 계열 추가 |
| Unlearning | Machine unlearning | NPO, RMU 중 최소 1개 이상 |
| Compression | Post-training compression | Structured/unstructured pruning, quantization |
| Transfer | Knowledge distillation | Architecture-matched self-distillation 우선 |
| Composition | Model merging | Weight averaging 또는 task-vector 기반 merge |
| Security | Controlled backdoor fine-tuning | 안전한 synthetic trigger와 benign target 사용 |

## 8.2 History 길이

### Stage A: 길이 0–1

```text
Root
Root → SFT
Root → DPO
Root → Unlearning
Root → Quantization
```

### Stage B: 길이 2

```text
Root → SFT → DPO
Root → SFT → Unlearning
Root → Backdoor-SFT → Unlearning
Root → SFT → Quantization
Root → DPO → Quantization
Root → SFT → Distillation
```

### Stage C: 길이 3 이상

핵심 결과가 확인된 이후 제한적으로 확장한다.

```text
Root → SFT → DPO → Quantization
Root → Backdoor-SFT → Unlearning → Benign fine-tuning
```

## 8.3 순서 반전 쌍

Order sensitivity를 확인하기 위해 가능한 경우 다음처럼 동일 operation set의 순서를 바꾼 pair를 만든다.

```text
SFT → DPO        vs. DPO → SFT
SFT → Quantize   vs. Quantize → SFT
Backdoor → NPO   vs. NPO → Backdoor
Prune → SFT      vs. SFT → Prune
```

모든 조합이 의미 있거나 구현 가능한 것은 아니므로, operation semantics가 성립하는 pair만 사용한다.

---

# 9. Model zoo 구성

## 9.1 기본 원칙

- 가능한 한 서로 다른 model family를 포함한다.
- 동일 family 내에서도 서로 다른 root checkpoint를 사용한다.
- Train/test에 동일 root의 descendant가 동시에 들어가지 않도록 한다.
- Architecture와 hidden size가 history label의 trivial cue가 되지 않게 한다.
- Distillation 실험에서는 architecture-matched self-distillation을 우선한다.
- Quantization format 등 명백한 metadata cue를 제거한 평가와 포함한 평가를 분리한다.

## 9.2 권장 파일럿 규모

리소스를 통제하기 위해 0.5B–3B급 open-weight language model을 중심으로 시작한다.

```text
Model families              : 2
Root checkpoints per family : 2
Total roots                 : 4
Single-operation variants   : 5 operations × 4 variants/seeds
Compositional variants      : 6 selected ordered pairs × 3 variants/seeds
Approximate total           : 150 models 내외
```

실제 수는 intervention별 비용과 deterministic operation 여부에 따라 조정한다.

## 9.3 확장 규모

파일럿이 Go 기준을 충족하면 다음을 추가한다.

- 세 번째 model family
- unseen architecture 크기
- full fine-tuning과 parameter-efficient fine-tuning 비교
- sequence length 3
- alternative unlearning 및 preference optimization 방법
- vision model을 이용한 modality transfer sanity check

---

# 10. 데이터와 probe 설계

## 10.1 Intervention data와 probe data 분리

Probe가 training data를 직접 재현하면 intervention attribution이 아니라 dataset membership을 학습할 수 있다. 따라서 다음을 분리한다.

```text
D_operation : 실제 post-training에 사용하는 데이터
D_probe     : forensic fingerprint 추출용 공개 데이터
D_eval      : behavior 및 utility 측정용 데이터
```

세 집합은 가능한 한 겹치지 않게 구성한다.

## 10.2 Probe set 구성

Probe set은 단일 domain에 의존하지 않도록 여러 범주를 포함한다.

- 일반 문장 및 encyclopedic text
- instruction-following prompts
- reasoning-free neutral prompts
- synthetically generated minimal pairs
- random 또는 semantically unrelated text
- 안전성·거부 행동 측정용 분리된 audit prompts

## 10.3 Probe robustness

다음 조건을 비교한다.

- Probe domain 변경
- Probe 수 감소
- Token length 변경
- Prompt paraphrase
- Random subset
- Entirely unrelated domain

원하는 결과는 특정 trigger나 training corpus를 알아야만 작동하는 detector가 아니라, **범용 probe에서도 유지되는 signature**이다.

---

# 11. Behavior matching과 confound 통제

## 11.1 왜 matching이 필요한가

Operation 간 task accuracy나 perplexity가 크게 다르면 detector는 training history가 아니라 성능 차이를 사용할 수 있다.

## 11.2 Matching 기준

가능한 history pair를 다음 기준으로 matching하거나 통계적으로 conditioning한다.

- task accuracy
- perplexity
- output agreement
- logit KL divergence
- calibration error
- refusal rate
- output entropy
- generation length

## 11.3 강한 평가 조건

가장 중요한 실험은 다음이다.

> **Behaviorally matched models remain forensically distinguishable.**

즉, probe 또는 downstream evaluation에서 거의 같은 output을 생성하는 모델들이 history fingerprint에서는 구분되는지 확인한다.

## 11.4 Confound-only baselines

다음 정보만으로 history를 예측하는 baseline을 별도로 둔다.

- Parameter count
- Hidden dimension
- Number of layers
- File size
- Quantization metadata
- Training utility
- Output entropy
- Root model ID

Forensic descriptor가 이 baseline을 넘지 못하면 연구 가설을 지지하지 못한다.

---

# 12. Forensic fingerprint 설계

## 12.1 Black-box descriptor

- Output token frequency
- Sequence length
- Refusal style
- Calibration proxy
- Label agreement
- Generation diversity
- Prompt sensitivity

## 12.2 Logit descriptor

- Logit margin
- Entropy
- Top-k probability profile
- Token-rank statistics
- Logit covariance
- Probe-wise differential pattern

Base model을 사용하는 difference feature는 **reference-aware upper bound**로만 사용한다.

## 12.3 Activation geometry

- Layer-wise mean 및 covariance
- Pairwise distance distribution
- Cosine similarity distribution
- Class 또는 prompt-group centroid distance
- kNN graph statistics
- Local neighborhood overlap
- Effective rank
- Participation ratio
- Singular-value decay
- Intrinsic dimension
- Local intrinsic dimensionality
- CKA, SVCCA 또는 유사 representation-similarity measure

## 12.4 Attention descriptor

- Head-wise entropy
- Attention distance
- Attention sparsity
- Layer/head covariance
- Prompt group 간 attention shift

## 12.5 Weight-space descriptor

- Layer-wise norm
- Weight update spectrum
- Adapter singular values
- Stable rank
- Hessian 또는 curvature proxy
- Task-vector statistics
- Layer-wise sparsity 및 quantization residual

Reference-free setting에서는 absolute statistics를 사용하고, reference-aware 실험에서는 delta statistics를 upper bound로 비교한다.

---

# 13. TDA 모듈

## 13.1 역할

TDA는 latent representation의 multi-scale global structure를 요약한다. 본 연구에서의 질문은 다음이다.

> **Topological descriptor가 root, seed, probe 변화 및 laundering에 대해 non-topological descriptor보다 더 안정적인 history evidence를 제공하는가?**

## 13.2 Point cloud 구성

각 layer \(l\)에서 probe token 또는 sequence representation을 추출한다.

```text
H_l ∈ R^(N × d_l)
```

계산 비용과 concentration 문제를 줄이기 위해 다음을 적용한다.

- Sequence-level pooling과 token-level 분석 분리
- PCA 또는 random projection
- Probe group별 balanced sampling
- 동일한 point 수로 subsampling
- 여러 subsample에 대한 반복 측정

## 13.3 계산 대상

우선 다음 homology dimension을 사용한다.

- H0: connected components
- H1: loops
- H2: 계산 가능성과 안정성이 확인된 경우에만 제한적으로 사용

## 13.4 Topological summary

- Persistence diagram
- Betti curve
- Persistence landscape
- Persistence image
- Persistence entropy
- Total persistence
- Lifetime distribution
- Sliced Wasserstein distance
- Bottleneck 또는 Wasserstein distance

## 13.5 계산 최적화

- 전체 layer가 아니라 early/middle/late representative layers 우선
- Vietoris–Rips complex의 point 수 제한
- Witness 또는 sparse filtration 검토
- 동일 model에서 subsampling distribution을 만들어 불확실성 추정
- Ripser, GUDHI 또는 동등한 재현 가능한 구현 사용

## 13.6 TDA의 성공 기준

다음 중 하나 이상을 충족해야 TDA가 실질적 기여를 가진다.

1. Leave-root-out에서 non-topological baseline 대비 유의한 개선
2. Laundering 이후 성능 저하가 더 작음
3. Operation order에 추가 정보 제공
4. Open-set detection에서 unknown separation 개선
5. Descriptor ablation에서 독립적인 정보량 확인

단순 in-distribution accuracy 1–2% 개선만으로는 충분하지 않다.

---

# 14. Attribution model

## 14.1 단순 baseline 우선

Forensic signal 자체를 검증하기 위해 복잡한 neural classifier보다 다음을 먼저 사용한다.

- Logistic regression
- Linear SVM
- Random forest
- Gradient boosting
- Nearest-centroid classifier
- Metric learning baseline

## 14.2 Operation presence prediction

각 operation의 적용 여부를 multi-label classification으로 예측한다.

```text
ŷ_presence ∈ {0,1}^K
```

## 14.3 Order prediction

두 가지 접근을 비교한다.

### Pairwise precedence

각 operation pair에 대해 다음을 예측한다.

```text
g_i precedes g_j
```

예측된 pairwise relation으로 DAG 또는 sequence를 복원한다.

### Sequence decoder

Fingerprint를 입력받아 operation token sequence를 생성한다.

복잡한 decoder가 root leakage를 학습할 가능성이 있으므로 pairwise baseline을 우선한다.

## 14.4 Hierarchical prediction

```text
Step 1: known vs. unknown
Step 2: operation family prediction
Step 3: exact operation prediction
Step 4: order prediction
Step 5: confidence calibration 및 abstention
```

## 14.5 Open-set mechanism

다음을 비교한다.

- Energy score
- Mahalanobis distance
- One-class classifier
- Deep SVDD
- Conformal prediction
- Distance-to-class prototype
- Ensemble disagreement

최종 시스템은 낮은 confidence에서 forced attribution 대신 `unknown / insufficient evidence`를 출력해야 한다.

---

# 15. Baseline 및 직접 비교 대상

## 15.1 Baseline 범주

| 범주 | Baseline |
|---|---|
| Trivial | Architecture, file size, utility, entropy |
| Black-box | Output statistics, label agreement |
| Logit | Entropy, margin, token-rank profile |
| Activation | CKA, covariance spectrum, effective rank, intrinsic dimension |
| Attention | Head entropy, attention difference |
| Weight | Norm, spectrum, sparsity, adapter SVD |
| Topology | Persistence image, landscape, Betti curve |
| Reference-aware | Base–target activation/logit/weight differences |
| Provenance | Parent/model-family fingerprinting methods의 재현 가능한 variant |

## 15.2 Reference-aware upper bound

Base model을 사용할 수 있는 기존 계열 방법은 본 연구의 직접 setting과 다르지만 다음을 확인하기 위한 upper bound로 포함한다.

- Base가 알려졌을 때 성능이 얼마나 상승하는가
- Base-free setting에서 손실되는 정보는 무엇인가
- Reference-free descriptor가 어떤 intervention에서 충분한가

## 15.3 동일 정보량 비교

TDA와 다른 descriptor는 동일한 layer, 동일한 probe, 동일한 sample 수를 사용해 비교한다. 그렇지 않으면 TDA의 효과인지 더 많은 정보 사용의 효과인지 구분할 수 없다.

---

# 16. 데이터 분할과 leakage 방지

## 16.1 금지할 split

Model instance를 무작위로 나누는 random split은 기본 결과로 사용하지 않는다.

같은 root에서 파생된 sibling model이 train과 test에 동시에 들어가면 classifier가 intervention이 아니라 root fingerprint를 기억할 수 있다.

## 16.2 필수 split

### Leave-root-checkpoint-out

Test root의 모든 descendant를 학습에서 제외한다.

### Leave-seed-out

특정 initialization 또는 fine-tuning seed를 학습에서 제외한다.

### Leave-family-out

하나의 model family 전체를 test로 사용한다.

### Leave-dataset-out

특정 intervention dataset을 test에서만 사용한다.

### Leave-intensity-out

학습에서 보지 않은 learning rate, epoch, rank, pruning ratio, unlearning strength를 평가한다.

### Leave-composition-out

개별 operation은 봤지만 특정 조합은 보지 않은 상태에서 sequence를 평가한다.

## 16.3 그룹 단위 평가

통계적 독립 단위는 probe point가 아니라 **독립적으로 생성된 model instance 또는 root group**으로 둔다.

---

# 17. Laundering benchmark

## 17.1 목적

Forensic trace가 존재하더라도 간단한 후처리로 사라진다면 공급망 보안에서의 실용성은 제한적이다.

## 17.2 비적응적 laundering

- 짧은 benign continued fine-tuning
- Unrelated public data fine-tuning
- 8-bit 및 4-bit quantization
- Structured/unstructured pruning
- Model merging
- Secondary distillation
- Adapter merge 및 재분해
- Weight noise 또는 low-rank perturbation

## 17.3 적응적 laundering

공격자가 detector 또는 surrogate detector의 fingerprint loss를 최소화하면서 utility를 유지하도록 최적화한다.

예시 목적함수:

```text
L_total = L_utility + λ_1 L_fingerprint_evasion + λ_2 L_behavior_preservation
```

## 17.4 평가 질문

- Trace를 지우는 데 필요한 utility cost는 얼마인가
- 어떤 descriptor가 가장 먼저 무너지는가
- TDA signature는 다른 geometry보다 오래 유지되는가
- Operation presence는 지워져도 order information은 남는가
- Detector-aware laundering이 transfer되는가

## 17.5 공격 성공 정의

공격 성공은 다음을 동시에 충족해야 한다.

- Attribution confidence 또는 accuracy 유의하게 감소
- Target model utility 유지
- Output behavior 변화가 허용 범위 이하
- Obvious metadata artifact를 남기지 않음

---

# 18. 평가 지표

## 18.1 Operation presence

- Macro/micro F1
- AUROC 및 AUPRC
- Balanced accuracy
- Per-operation sensitivity 및 specificity

## 18.2 Sequence reconstruction

- Exact sequence match
- Normalized edit distance
- Pairwise order accuracy
- Kendall-style rank agreement
- Prefix accuracy
- Set accuracy와 order accuracy 분리

## 18.3 Open-set

- AUROC for unknown detection
- AUPRC
- FPR@95TPR
- OSCR 또는 이에 준하는 open-set metric
- Selective risk
- Coverage–risk curve

## 18.4 Calibration

- Expected calibration error
- Brier score
- Negative log-likelihood
- Abstention coverage와 residual error

## 18.5 Generalization

- In-root 대비 leave-root-out 성능 저하
- In-family 대비 leave-family-out 성능 저하
- Seen 대비 unseen dataset/intensity/composition 성능 저하

## 18.6 Laundering robustness

- Attack 전후 attribution degradation
- Utility–evasion Pareto curve
- Trace half-life: 추가 training step에 따른 신호 감소
- 공격 비용 대비 detection 감소량

## 18.7 Descriptor contribution

- Incremental AUROC/F1
- Conditional mutual-information proxy
- Feature-group ablation
- Permutation importance
- Shapley-style group importance는 계산 비용이 허용될 때만 사용

---

# 19. 통계 분석 계획

## 19.1 독립 단위

독립적인 model instance를 기본 표본 단위로 한다. 동일 모델의 여러 probe와 여러 layer를 독립 표본처럼 취급하지 않는다.

## 19.2 불확실성

- Model-level bootstrap confidence interval
- Root-grouped bootstrap
- Seed-grouped bootstrap
- 필요한 경우 hierarchical bootstrap

## 19.3 가설 검정

- Operation 간 fingerprint distance에 대한 permutation test
- Within-operation과 between-operation distance 비교
- Behavior-matched pair에 대한 paired test
- Descriptor별 cross-root 성능 비교

## 19.4 다중 비교

다수의 operation, descriptor, layer 및 laundering 조건을 비교하므로 다음을 사전 지정한다.

- Primary endpoints: Holm correction
- Exploratory analysis: Benjamini–Hochberg FDR

## 19.5 Effect size

p-value만 제시하지 않고 다음을 함께 보고한다.

- Mean/median difference
- Standardized effect size
- Confidence interval
- Generalization gap
- Laundering degradation ratio

## 19.6 Equivalence analysis

두 history의 차이가 의미 있게 작다는 주장을 위해 equivalence margin을 사전 정의한다.

이를 통해 다음 두 결론을 구분한다.

- 차이를 검출하지 못함
- 실질적으로 동등하다는 증거가 있음

---

# 20. 핵심 실험

## Experiment 1. Single-operation attribution

목적: Base 없이 개별 operation을 식별할 수 있는지 확인한다.

```text
Train roots: A, B, C
Test root : D
History   : length 0–1
```

출력:

- Operation multi-class/multi-label performance
- Descriptor별 cross-root generalization
- Trivial cue baseline 비교

## Experiment 2. Behavior-matched forensics

목적: 성능 차이를 제거한 뒤에도 history signal이 유지되는지 확인한다.

방법:

- Utility와 output agreement가 유사한 model pair 구성
- Matched-pair attribution
- Output-only baseline과 representation/weight descriptor 비교

## Experiment 3. Compositional history reconstruction

목적: Operation set과 order를 함께 복원한다.

평가:

- Seen pair
- Unseen composition
- Reversed order pair
- Sequence exact match 및 edit distance

## Experiment 4. Open-set attribution

목적: Unknown operation 또는 조합을 식별한다.

```text
Train: SFT, DPO, NPO, Quantization
Test unknown: Model merging 또는 새로운 unlearning method
```

출력:

- Unknown detection
- Calibrated abstention
- Forced classification 대비 risk 감소

## Experiment 5. Observation-interface ablation

목적: History별 최소 필요 information을 분석한다.

```text
Black-box → Logit → Activation → Attention → Weight
```

출력:

- Interface별 identifiable operation
- 비용–성능 trade-off
- Reference-aware upper bound와의 차이

## Experiment 6. Laundering robustness

목적: Forensic trace가 추가 후처리에 얼마나 견디는지 평가한다.

출력:

- Utility–evasion curve
- Descriptor별 degradation
- Adaptive attack 결과

## Experiment 7. Topological incremental value

목적: TDA가 실제로 필요한 조건을 규명한다.

비교:

```text
Spectrum only
Geometry only
TDA only
Spectrum + Geometry
Spectrum + Geometry + TDA
```

핵심 조건:

- Leave-root-out
- Leave-family-out
- Laundering
- Open-set

## Experiment 8. Identifiability map

목적: History pair별 구별 가능성을 지도화한다.

```text
Rows    : true history
Columns : alternative history
Cell    : calibrated distinguishability / equivalence evidence
```

최종적으로 어떤 history는 구별 가능하고 어떤 history는 같은 observational equivalence class에 속하는지 보고한다.

---

# 21. Ablation study

## 21.1 Probe ablation

- Probe 수
- Probe domain
- Prompt length
- Random vs. curated probe
- Training-related vs. unrelated probe

## 21.2 Layer ablation

- Early layer
- Middle layer
- Late layer
- Layer aggregation
- Layer-selection leakage 방지

## 21.3 Representation ablation

- Last token
- Mean pooling
- All-token point cloud
- Prompt-token과 generation-token 분리

## 21.4 Descriptor ablation

- Output 제거
- Weight 제거
- TDA 제거
- Spectrum 제거
- Intrinsic dimension 제거

## 21.5 Confound ablation

- Architecture-matched subset
- Hidden-size-matched subset
- Utility-matched subset
- Quantization metadata 제거
- Root identity adversarial removal

## 21.6 Attribution model ablation

- Linear model
- Tree ensemble
- Metric-based method
- Sequence decoder
- Open-set calibration method

---

# 22. 이론 및 분석 구성

## 22.1 최소 이론 목표

완전한 general theorem보다 다음을 명확히 정식화한다.

1. Observation interface에 따른 identifiability 차이
2. Probe distribution에 따른 indistinguishability
3. Functionally equivalent model의 black-box attribution 한계
4. Parameter permutation 등 representation symmetry가 white-box descriptor에 미치는 영향

## 22.2 가능한 정리 방향

두 history가 probe distribution에서 동일한 output distribution을 생성하면, 해당 black-box interface만 사용하는 어떤 detector도 두 history를 안정적으로 구분할 수 없다는 indistinguishability statement를 제시할 수 있다.

White-box setting에서는 다음 symmetry를 고려한다.

- Neuron permutation
- Scaling symmetry
- Low-rank reparameterization
- Function-preserving transformations

Descriptor가 이러한 symmetry에 invariant하지 않으면 history가 아니라 parameterization artifact를 포착할 수 있다.

## 22.3 실증적 identifiability

각 history pair에 대해 다음을 보고한다.

- Best achievable cross-validated distinguishability
- Confidence interval
- Equivalence result
- 필요한 최소 observation interface
- Laundering 이후의 변화

---

# 23. 예상 기여

## Contribution 1. 문제 정의

Base checkpoint와 metadata 없이 compositional post-training history를 추론하는 **base-free model forensics** 문제를 명시적으로 정식화한다.

## Contribution 2. Benchmark

다양한 root, seed, intervention, sequence 및 laundering 조건을 포함하는 reproducible model-history benchmark를 구축한다.

## Contribution 3. Cross-root evaluation protocol

Sibling leakage를 방지하는 leave-root-out, leave-family-out, leave-composition-out 평가 프로토콜을 제안한다.

## Contribution 4. Open-set forensic attribution

Unknown operation에 대한 calibrated abstention을 포함하여 단순 closed-set 분류보다 현실적인 forensic setting을 제시한다.

## Contribution 5. Laundering analysis

추가 post-training으로 forensic trace를 제거할 수 있는지 체계적으로 평가하고 utility–evasion trade-off를 정량화한다.

## Contribution 6. Identifiability limits

모든 history가 구별 가능한 것이 아님을 명시하고, observation interface별로 구별 가능성과 observational equivalence를 분석한다.

## Contribution 7. TDA의 조건부 역할 규명

TDA가 언제 도움이 되고 언제 spectrum·CKA·logit statistics로 충분한지를 비교하여, topology의 실질적 기여 범위를 명확히 한다.

---

# 24. 예상 결과 시나리오

## 시나리오 A. 강한 긍정 결과

- Leave-root-out에서도 operation presence가 안정적으로 식별됨
- 일부 sequence order 복원 가능
- Unknown operation에 대한 기권 가능
- Laundering 이후에도 특정 structural signature 유지
- TDA가 cross-root 또는 laundering에서 독립적 성능 제공

이 경우 model supply-chain forensics framework로 강하게 주장할 수 있다.

## 시나리오 B. 부분적 긍정 결과

- In-family에서는 잘되지만 leave-family-out에서 약함
- Operation presence는 가능하지만 order는 어려움
- TDA는 일부 operation에서만 도움
- Benign fine-tuning으로 흔적이 빠르게 약화됨

이 경우 연구의 중심을 **forensic identifiability map과 failure boundary**로 둔다.

## 시나리오 C. 부정 결과

- Behavior matching과 root holdout 이후 signal이 대부분 사라짐
- History보다 seed/root fingerprint가 지배적
- 간단한 laundering으로 모든 detector가 무너짐

이 결과도 가치가 있다. 다음과 같은 결론으로 전환할 수 있다.

> **Final-checkpoint-only training-history attribution is fundamentally unreliable without trusted provenance metadata.**

즉, detector 제안 논문이 아니라 provenance claim의 한계를 실증하는 negative-results/SoK형 연구가 된다.

---

# 25. Go / No-Go 기준

## 25.1 Pilot Go 기준

다음 중 다수를 충족하면 전체 연구로 확장한다.

- Leave-root-out에서 chance 대비 명확한 operation attribution
- Utility 및 output matching 이후에도 signal 유지
- Root ID adversarial removal 이후에도 성능 유지
- Unknown operation detection이 forced classification보다 안정적
- 길이 2 history에서 set prediction 또는 order prediction이 가능
- 최소 하나의 descriptor가 laundering 이후 유의한 robustness 보유
- TDA가 특정 어려운 조건에서 독립적 정보 제공

## 25.2 Pivot 기준

다음 결과가 나오면 detector 중심에서 identifiability-limit 중심으로 전환한다.

- Single operation은 가능하지만 composition/order가 불가능
- Cross-family에서 성능이 크게 붕괴
- Laundering에 매우 취약
- Descriptor 간 차이가 작고 TDA 이점이 없음

## 25.3 No-Go 기준

다음이 확인되면 현재 formulation은 중단한다.

- Random split에서만 성능이 높고 grouped split에서 chance 수준
- Architecture/file metadata baseline과 차이가 없음
- Base reference 없이는 신호가 거의 없음
- Behavior matching 후 attribution이 사라짐
- 단순 benign continued training만으로 완전히 무력화
- 모델 수를 늘려도 confidence interval이 chance와 구분되지 않음

---

# 26. 실행 단계

## Phase 1. 최소 파일럿

- 2개 model family, 4개 root
- Null, SFT, DPO, unlearning 중심
- Length 0–1 history
- Output/logit/spectrum/CKA/TDA baseline
- Leave-root-out 평가

### 산출물

- Model-generation pipeline
- Fingerprint extraction pipeline
- Leakage-free evaluation script
- 초기 Go/No-Go 보고서

## Phase 2. Compositional history

- 선택된 ordered pair 생성
- Multi-label presence와 pairwise order prediction
- Behavior matching
- Unseen composition test

### 산출물

- History graph dataset
- Sequence attribution baseline
- Order sensitivity 분석

## Phase 3. Open-set 및 cross-family

- Unknown operation 추가
- Leave-family-out
- Confidence calibration
- Conformal 또는 energy-based abstention

### 산출물

- Open-set benchmark
- Coverage–risk 분석

## Phase 4. Laundering

- 비적응적 laundering
- Detector-aware adaptive laundering
- Utility–evasion Pareto analysis

### 산출물

- Laundering attack suite
- Descriptor robustness ranking

## Phase 5. Identifiability map 및 논문화

- History pair별 distinguishability matrix
- Equivalence analysis
- 실패 사례 정리
- Artifact packaging 및 paper 작성

---

# 27. 구현 구조 제안

```text
model-autopsy/
├── configs/
│   ├── roots/
│   ├── operations/
│   ├── histories/
│   ├── probes/
│   └── experiments/
├── src/
│   ├── model_zoo/
│   ├── interventions/
│   │   ├── sft.py
│   │   ├── dpo.py
│   │   ├── unlearning.py
│   │   ├── pruning.py
│   │   ├── quantization.py
│   │   ├── distillation.py
│   │   └── merging.py
│   ├── probes/
│   ├── extraction/
│   │   ├── outputs.py
│   │   ├── logits.py
│   │   ├── activations.py
│   │   ├── attention.py
│   │   └── weights.py
│   ├── descriptors/
│   │   ├── spectral.py
│   │   ├── geometry.py
│   │   ├── intrinsic_dimension.py
│   │   ├── topology.py
│   │   └── metadata_controls.py
│   ├── attribution/
│   │   ├── presence.py
│   │   ├── ordering.py
│   │   ├── open_set.py
│   │   └── calibration.py
│   ├── laundering/
│   ├── evaluation/
│   └── statistics/
├── manifests/
│   ├── models.jsonl
│   ├── histories.jsonl
│   └── probes.jsonl
├── scripts/
├── tests/
├── results/
└── paper/
```

## 27.1 Model manifest 예시

```json
{
  "model_id": "rootA_sft_dpo_seed3",
  "root_group": "rootA",
  "family": "family_1",
  "history": ["sft", "dpo"],
  "operation_configs": ["sft_cfg_02", "dpo_cfg_01"],
  "seed": 3,
  "probe_version": "probe_v1",
  "behavior_metrics": {},
  "artifact_hash": "..."
}
```

모든 모델과 결과에 config, seed, code commit 및 artifact hash를 기록한다.

---

# 28. 재현성 계획

- 모든 history를 declarative config로 정의
- Root, operation, seed 및 dataset split 고정
- Model artifact hash 기록
- Probe version 고정
- Figure와 table을 raw result에서 자동 재생성
- Model-generation log와 failure log 보존
- Primary hypothesis와 endpoint 사전 등록
- Test root와 unknown operation은 분석 전 고정
- Hyperparameter search와 final evaluation 분리

가능한 경우 공개 시 다음을 제공한다.

- Code
- Config
- Probe set
- Fingerprint feature
- Model manifest
- 경량 adapter 또는 delta weight
- Full checkpoint를 공유할 수 없는 경우 재생성 script

---

# 29. 자원 관리

## 29.1 계산비용 절감 원칙

- LoRA 기반 operation으로 파일럿 시작
- Activation은 representative layer만 우선 추출
- TDA point cloud subsampling
- Fingerprint를 cache하여 attribution 실험 반복
- Full fine-tuning은 핵심 가설 확인 후 제한적으로 수행
- Distillation과 adaptive laundering은 후반 단계로 배치

## 29.2 저장공간

각 모델 전체 checkpoint를 중복 저장하지 않고 가능한 경우 다음을 활용한다.

- Base + adapter
- Base + delta
- Quantization config
- Sparse pruning mask
- Deterministic regeneration manifest

---

# 30. 위험요인과 대응

| 위험 | 영향 | 대응 |
|---|---|---|
| Root/seed leakage | 가짜 고성능 | Grouped split, root adversarial removal |
| Class가 상호배타적이지 않음 | 잘못된 문제 정의 | Multi-label presence + ordered sequence |
| Distillation architecture cue | Trivial classification | Architecture-matched self-distillation |
| Probe가 training data를 암시 | Dataset membership으로 변질 | Unrelated probe 및 leave-dataset-out |
| TDA 계산비용 | 실험 규모 축소 | Subsampling, representative layers, sparse filtration |
| Laundering으로 신호 소실 | 보안적 가치 약화 | Failure boundary 및 provenance-limit 논문으로 pivot |
| Cross-family 일반화 실패 | 범용성 부족 | Family-specific vs. invariant signal 분리 분석 |
| Unknown operation 오분류 | 실제 배치 위험 | Abstention 및 conformal calibration |
| 선행연구의 빠른 증가 | Novelty 감소 | Base-free composition·open-set·laundering 결합 유지 |

---

# 31. 윤리 및 보안 고려

- Backdoor 실험은 실제 피해가 없는 synthetic dataset과 benign target으로 제한한다.
- 공격 코드는 detector 평가에 필요한 수준으로 공개하며 악용 가능성을 검토한다.
- Attribution 결과를 provenance의 확정적 증거로 표현하지 않는다.
- False positive가 공급자 또는 연구자에게 잘못된 비난으로 이어질 수 있으므로 calibrated uncertainty를 필수로 보고한다.
- 공개 모델 license와 dataset license를 준수한다.
- 민감한 training data 복원이나 개인정보 추출을 연구 목표로 삼지 않는다.

---

# 32. 논문 구성 초안

## 1. Introduction

- Model supply-chain opacity
- 기존 provenance/trace detection의 reference 의존성
- Compositional post-training history 문제
- 주요 기여

## 2. Related Work

- Activation/logit traces of fine-tuning
- Training order and unlearning traces
- Model provenance and lineage
- Neural-network weight fingerprints
- Topological analysis of representations
- Open-set recognition and forensic attribution

## 3. Problem Formulation

- History sequence
- Observation interface
- Base-free setting
- Open-set 및 identifiability

## 4. Model-History Benchmark

- Root models
- Operations
- Sequence generation
- Probe sets
- Split protocol

## 5. Forensic Descriptors

- Output/logit
- Geometry/spectrum
- Attention/weight
- TDA

## 6. Attribution and Abstention

- Presence prediction
- Order reconstruction
- Open-set calibration

## 7. Experiments

- Single operation
- Behavior matching
- Composition
- Cross-root/family
- Open-set
- Laundering

## 8. Identifiability Limits

- Equivalence analysis
- Failure cases
- Observation-interface map

## 9. Discussion

- What evidence can and cannot establish
- Deployment implications
- Provenance metadata와의 결합

## 10. Limitations and Ethics

## 11. Conclusion

---

# 33. 제목 후보

## 권장 제목

**Model Autopsy: Base-Free Forensic Attribution of Compositional Post-Training Histories**

## 대안

- **What Can We Infer About a Model’s Past? Identifiability Limits of Post-Training Histories**
- **Training Leaves Traces, But Which Ones? Open-Set Forensics of Post-Trained Models**
- **Beyond Model Lineage: Inferring Compositional Training Histories from Final Checkpoints**
- **Forensic Identifiability of Post-Training Interventions**

보안과 실용성을 강조하려면 첫 번째 제목이 좋고, 부정 결과와 한계 분석까지 중심에 두려면 두 번째 제목이 적합하다.

---

# 34. 초기 논문 주장 초안

연구 결과가 가설을 지지할 경우 다음 수준으로 주장한다.

> We study whether the ordered post-training history of an unknown model can be inferred from its final checkpoint without access to its base model, training logs, or sibling checkpoints. We introduce a leakage-resistant benchmark spanning multiple roots, operation compositions, open-set interventions, and laundering transformations. Our results characterize which histories remain identifiable under different observation interfaces and when apparently strong forensic signals collapse under root shifts or adaptive post-processing.

TDA 관련 주장은 결과 확인 후 다음처럼 제한한다.

> Topological descriptors are not universally superior, but provide complementary and, for selected interventions, more laundering-resistant evidence than local geometric and spectral summaries.

이 결과가 실제로 나오지 않으면 TDA를 주요 contribution에서 제외한다.

---

# 35. 파일럿 착수 체크리스트

- [ ] Root model 4개 선정
- [ ] Operation 4개와 null control 확정
- [ ] History manifest schema 작성
- [ ] Intervention data, probe data, behavior-evaluation data 분리
- [ ] Sibling leakage 없는 split generator 구현
- [ ] Output/logit/activation/weight extraction 구현
- [ ] Spectrum, CKA, intrinsic dimension baseline 구현
- [ ] H0/H1 persistence pipeline 구현
- [ ] Confound-only baseline 구현
- [ ] Single-operation leave-root-out 실험 실행
- [ ] Behavior matching protocol 구현
- [ ] Go/No-Go 결과 기록
- [ ] Go 판정 후 length-2 composition 생성
- [ ] Open-set operation 한 개를 test-only로 고정
- [ ] Laundering transformation 최소 3개 구현
- [ ] Model-level bootstrap 및 multiple-testing correction 구현

---

# 36. 초기 참고문헌 목록

아래 문헌은 본 연구의 novelty와 baseline을 정리할 때 우선적으로 검토할 대상이다. 최종 원고 작성 전 bibliographic metadata와 최신 버전을 다시 검증한다.

1. **Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences**. arXiv:2510.13900.
2. **Delta Activations: A Representation for Finetuned Large Language Models**. arXiv:2509.04442.
3. **Diff Mining: Logit Differences Reveal Finetuning Objectives**. arXiv:2608.26462.
4. **Fresh in Memory: Training-order Recency is Linearly Encoded in Language Model Activations**. arXiv:2509.14223.
5. **Unlearning Isn’t Invisible: Detecting Unlearning Traces in LLMs from Model Outputs**. arXiv:2506.14003.
6. **Detecting Safety Training Modification in Language Models via Activation Analysis**. arXiv:2608.05578.
7. **Mapping the Multiverse of Latent Representations (PRESTO)**. ICML 2024.
8. **Tracking Representation Dynamics in Large Language Models with Persistent Homology**. arXiv:2606.19542.
9. **The Shape of Adversarial Influence: Characterizing LLM Latent Spaces with Persistent Homology**. arXiv:2505.20435.
10. **Persistent Topological Features in Large Language Models**. arXiv:2410.11042.
11. **REEF: Representation-based Model Lineage and Provenance Analysis**. arXiv:2410.14273.
12. **TokenPrint**. arXiv:2608.08139.
13. **SeedPrints**. arXiv:2509.26404.
14. **MoTHer: Model Heritage Tree Reconstruction**. arXiv:2405.18432.
15. **Model Provenance Testing for Large Language Models**. arXiv:2502.00706.
16. **Knowledge Distillation Detection for Open-weights Models**. arXiv:2510.02302.
17. **Classifying the Classifier: Dissecting the Weight Space of Neural Networks**. arXiv:2002.05688.
18. **Self-Supervised Representation Learning on Neural Network Weights for Model Characteristic Prediction**. NeurIPS 2021.

---

# 37. 최종 의사결정

본 연구는 **“TDA를 이용한 training-type classifier”**로 진행하지 않는다.

최종 방향은 다음과 같다.

> **Base-free, cross-root, compositional, open-set, and laundering-aware forensic attribution of post-training histories, together with an explicit analysis of identifiability limits.**

TDA는 이 문제를 해결하는 여러 descriptor 중 하나로 엄격하게 비교한다. 성공 시에는 cross-root 또는 laundering robustness에 대한 보완적 기여를 주장하고, 실패 시에는 topology가 제공하지 못하는 정보까지 정직하게 보고한다.

연구의 가장 중요한 결과는 높은 classification accuracy 하나가 아니라 다음 지도이다.

```text
어떤 training history가
어떤 observation interface에서
어떤 root 변화와 laundering 조건까지
어느 정도 신뢰도로 식별 가능한가.
```

이 지도를 제공할 수 있다면 detector가 강하게 성공하든, 예상보다 크게 실패하든 학술적 가치가 남는다.
