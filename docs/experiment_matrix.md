# Controlled ablation experiment matrix

日期2026-09-14；所有planned均未訓練。詳細公式、evaluation及隔離規範見[improvement plan](improvement_plan.md)。

| ID | Spacing | Preprocess | Training sampling | Loss | Input | Dataset | Status |
|---|---|---|---|---|---|---|---|
| B0_baseline | 2mm | original z-score | uniform cases / fg .33 | Dice+CE | DWI | 001 | completed |
| P1_spacing_1p5mm | 1.5mm | baseline | baseline | Dice+CE | DWI | 011 reserved | planned primary |
| P2_spacing_1mm | 1mm | baseline | baseline | Dice+CE | DWI | 012 reserved | planned primary |
| P3a_clip_zscore | 2mm | .5–99.5 percentile + image-mask z-score | baseline | Dice+CE | DWI | 013 reserved | planned primary |
| P3b_clip_zscore | 2mm | 1–99 percentile + image-mask z-score | baseline | Dice+CE | DWI | 014 reserved | planned secondary |
| S1a_fg_oversampling | 2mm | baseline | fg .50 / uniform cases | Dice+CE | DWI | 001 read-only | planned primary |
| S1b_fg_oversampling | 2mm | baseline | fg .75 / uniform cases | Dice+CE | DWI | 001 read-only | planned secondary |
| S2_small_case_sampling | 2mm | baseline | fg .33 / 1–5mL case weight2 | Dice+CE | DWI | 001 read-only | planned secondary |
| L1a_dice_focal | 2mm | baseline | baseline | Dice+Focal γ2 | DWI | 001 read-only | planned primary |
| L1b_dice_focal | 2mm | baseline | baseline | Dice+Focal γ1 | DWI | 001 read-only | planned secondary |
| L2a_tversky | 2mm | baseline | baseline | Tversky FP.3/FN.7 + CE | DWI | 001 read-only | planned primary |
| L2b_focal_tversky | 2mm | baseline | baseline | (1−TI)^(4/3) + CE | DWI | 001 read-only | planned secondary |
| C1–C4 | evidence-selected | stepwise | stepwise | stepwise | DWI | assign after results | deferred |
| M0 | fixed improved model | fixed | fixed | fixed | DWI | final DWI dataset | deferred |
| M1 | matched planner policy | channel-specific | fixed | fixed | DWI+ADC | 021 reserved | deferred |
| M2 | matched planner policy | registered FLAIR | fixed | fixed | DWI+ADC+FLAIR | 022 reserved | deferred |

Reserved IDs必須在三個nnUNet roots及experiment registry檢查未占用才建立。P3 mask不同於B0時先增加P3-control（image-only z-score無clipping），独立dataset/output。

## 固定與衍生設定

除該列因素外維持B0：fold0的200/50 case IDs、SGD/PolyLR、augmentation、1000epochs、250/50 iterations、final checkpoint、binary labels、mixed precision。Sampling只改training loader；validation loader固定。Focal不設class alpha，各loss components權重1:1。2mm DS=4/7,2/7,1/7,0。
P1/P2由planner自動產生patch、batch、stages與DS；確切數字見[planning audit](improvement_planning_audit.json)，不得固定原patch而忽略FOV。這些是衍生混因，metadata全部記錄；VRAM/time待GPU實測。

## Unique trainer 與輸出

S1a/S1b分別nnUNetTrainer_FG050／FG075；S2=nnUNetTrainer_SmallCase2x；L1a/L1b=nnUNetTrainer_DiceFocalG2／G1；L2a/L2b=nnUNetTrainer_TverskyCE／FocalTverskyCE。這些是預定class名稱，目前未實作。P3使用獨立normalizer/preprocessor。
結果使用experiments/<ID>/<unique_run>/；每列有experiment_config.json、commands、metrics、logs、notes。禁止共用result directory或改官方class後仍沿用nnUNetTrainer名稱。

## 執行與比較順序

Baseline保護→evaluation v2→各primary可行性試跑→各primary完整1000epochs→secondary→單因子支持後逐步組合→paired seed確認／其餘folds→M0/M1→FLAIR registration QA後M2。
短程只驗證可執行，不用早期validation排名。所有執行開啟可見Terminal並保存獨立log；本次只完成文件與audit。
每列輸出相同Dice分布、HD95、volume errors、robustness counts、lesion metrics及五組case-volume分層；依catastrophic miss/severe under、small-lesion recall、overall Dice順序比較，FP增加標明tradeoff。Fold0是development validation，不是獨立測試。
