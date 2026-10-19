# ISLES2022 DWI-only 模型改善計畫

日期：2026-09-14。狀態：設計與 planning audit；未執行新 preprocessing 或訓練。
目標順序：降低 catastrophic miss／false negative > 改善 small-lesion recall > 提高 overall Dice。
交叉索引：[experiment matrix](experiment_matrix.md)、[baseline report](training_report.md)、[planner audit](improvement_planning_audit.json)。

## 1. Current baseline 與 failure pattern

B0：ISLES2022、DWI 單通道、background=0／lesion=1、fold 0（200 train／50 validation）。nnUNetv2 2.8.1、nnUNetTrainer、nnUNetPlans、3d_fullres、PlainConvUNet；2×2×2 mm、patch 80×96×80、batch 8、1000 epochs（250 training／50 validation iterations per epoch）。SGD LR=0.01、momentum=0.99、Nesterov、weight decay=3e-5、PolyLR、mixed precision，RTX 5080 16GB。

Loss 為 DeepSupervisionWrapper(DC_and_CE_loss)：MemoryEfficientSoftDiceLoss + RobustCrossEntropyLoss，權重 1:1，batch Dice=False，Dice 排除背景，smooth=1e-5，ignore_label=None。四層 scales 1、1/2、1/4、1/8，weights 4/7、2/7、1/7、0。
Augmentation 固定：rotation ±30°、scale 0.7–1.4、noise、blur、brightness、contrast、low resolution、gamma、mirroring；無 elastic deformation。

完整 3D Mean Dice 0.778、Median 0.829、SD(population) 0.179；31/49 ≥0.8，4/49 <0.5，3/49 <0.3；minimum 0.162。Volume MAE 4.51 mL，mean signed error -2.62 mL，36/49 非空病例低估體積。case_0151 雙空，Dice N/A。
1–5 mL 組 mean Dice 0.655（12例），5–20 mL 0.792，20–50 mL 0.868，≥50 mL 0.852；<1 mL 有效 n=3，mean 0.828，樣本不足。
case_0068／0182／0181／0140／0212 的 Pred/GT 分別為 0.098／0.114／0.191／0.379／0.428。主問題是 lesion coverage 不足；case_0140 為大型病灶（141.37 mL）低估87.74 mL。影像異質性是否為原因尚未確認，不以體積數字推論影像成因。

Final train loss -0.8897、validation loss -0.7770、patch pseudo-Dice 0.857。Patch 指標不得與 full-volume Dice 直接比較。負 composite loss 正常。後期改善有限，但主 ablation 固定1000 epochs及 checkpoint_final.pth；不以不同 loss 的數值大小選模。

## 2. Hypothesis 與 spatial resolution

H1：較密集的輸出格點可能改善小病灶 representation，但不增加原始 MRI 的真正空間解析度；也可能放大 interpolation、label resampling 誤差。1 mL 在2／1.5／1 mm網格約125／296／1000 voxels，僅為體積換算。
P1／P2 使用官方 planner 的 overwrite_target_spacing 概念，先計算3d_fullres configuration；不手填 patch。Audit 僅呼叫 get_plans_for_configuration，不呼叫 save_plans 或 plan_experiment。8GB planner target 是官方預設假設，不等同於GPU容量；先檢查2mm能否重現80×96×80／batch8，不能由重現結果反推歷史 target 已被證明。
Baseline physical FOV=160×192×160 mm。相同 physical FOV 下，1.5mm約需107×128×107、1mm需160×192×160 voxels；這些只是幾何參考，不能直接作為合法網路 patch。
以 audit 的 patch×spacing 報告實際 FOV。全 volume voxel 數相對2mm約2.37倍與8倍；不是VRAM或時間倍率。自動 planning 可能改 stages、patch、batch、DS層數，因此P1/P2是 spacing/planning 系統介入，不是純 spacing 因果比較。
正式實作先以planner batch做短程GPU試跑，記錄peak allocated/reserved VRAM、20個warm-up後100 iterations耗時、patches/sec與估計epoch工時。若OOM，停止該run並建立新revision；不得偷偷縮batch沿用ID。需要降低memory target時新增2mm matched-planner control，再比較相同target的P1/P2。8GB估計並非5080 16GB可執行保證；不得自動把target調至16GB。
固定epochs及iterations仍不等於相同patch exposure（batch可能變）；metadata同時報告optimizer steps、sampled patches、wall time。DS使用同一個幾何衰减／末層0規則，層數依plans。

## 3. DWI intensity preprocessing

### 已完成的 planner configuration 計算

| Spacing | Patch voxels | Batch | Physical FOV mm | Stages |
|---|---|---:|---|---:|
| 2mm | 80×96×80 | 8 | 160×192×160 | 5 |
| 1.5mm | 96×112×96 | 5 | 144×168×144 | 5 |
| 1mm | 112×160×128 | 2 | 112×160×128 | 6 |

官方預設8GB memory target的2mm結果重現B0 patch/batch。1mm FOV明顯縮小，可能犧牲大型病灶context，須同時檢查≥50mL分層與case_0140 coverage。這是CPU planner計算結果，尚無新GPU VRAM或訓練工時實測。Audit確認被保護的baseline檔案前後SHA256一致。

H2：clipping可降低少數高強度值對均值／SD的影響，但也可能壓縮真正高DWI病灶訊號。P3a=P0.5–P99.5、P3b=P1–P99（次要候選）。每case獨立計算percentiles、clipped mean／SD，不跨validation fitting。
實際B0為ZScoreNormalization(use_mask_for_norm=True)，以seg>=0選region，沒有percentile clipping。crop_to_nonzero建立DWI非零mask並fill holes，但只把mask外且GT=0的seg標為-1；GT在非零mask外時可能影響normalization support。這不表示已證明B0發生leakage，需檢查是否真的有此種voxel。
P3定義mask=M=binary_fill_holes(DWI!=0)，由影像單獨建立；M不是經驗證的brain extraction。先在M內clip，再在相同M內z-score，epsilon=1e-8、float32；M外0。empty mask或非有限值停止，constant image避免除0並記錄。與GT無關，train/inference相同；只normalization一次，再接nnUNet resampling。
實作採獨立dataset／custom preprocessor或normalizer，保留B0幾何plans；不得離線z-score後再默默套用另一輪標準化。若B0與image-only mask確實不同，新增image-only z-score但無clipping的P3-control，避免把mask修正效果歸因於clipping。
CLAHE／histogram equalization／N4先不加入正式ablation：CLAHE可能放大噪音、改變病灶對比，2D逐片處理可能產生跨片不連續；若研究，使用original DWI加CLAHE channel的獨立實驗。全域equalization會改變強度分布與組織相對對比，跨scanner效果待驗證。N4估計平滑乘性bias field，不能修正DWI EPI distortion；在低SNR或廣泛高訊號病灶可能產生不當校正，需先證明bias確實存在並做train-only QA。

## 4. Lesion-aware sampling

H3：增加前景exposure可能提升coverage，但背景不足可增加FP。2.8.1 loader先get_indices抽case，trainer傳sampling_probabilities=None（均勻case抽樣），再決定每個batch位置是否force_fg；B0 probabilistic_oversampling=False，判斷為index>=round(batch*(1-p))。batch8、p=.33實際3/8位置要求foreground（37.5%），不是33%病例被抽中；空GT無可用foreground時回退隨機crop。
class_locations由preprocessed GT逐class抽樣位置，預設最多10000或至少1%覆蓋的抽樣規則，並非connected-lesion均勻抽樣；病灶大的component仍可能主導位置。force_fg的crop包含選定位置，但經spatial augmentation後未必保留，應統計最終patch前景存在率。
S1a=.50（4/8）、S1b=.75（6/8），均保留uniform cases。只修改training loader，validation loader的oversampling必須保持.33；不可只改trainer共用屬性而連validation一起改。現有patch已大於median cropped image尺寸，效果可能小；記錄實際crop coverage才能判斷。
S2獨立於S1：只讓1≤training GT volume<5mL病例權重2，其餘1，依loader case key順序正規化；仍用B0 patch oversampling。從原始header spacing換算training volume，不能使用validation ID或GT；保持validation split與loader不變。
Loader已有sampling_probabilities參數，最小custom trainer只接入training loader建構；不修改site-packages中的官方trainer。自訂類別與環境發現路徑需驗證；保留官方檔案hash。

## 5. Loss ablation

H4：hard voxel weighting或提高FN penalty可能提升recall，同時增加FP或不穩定。以下均為待驗證候選，不宣稱最佳。
L1a γ=2、L1b γ=1：Focal=mean[-(1-p_t)^γ log(p_t)]，用log_softmax/gather及float32 reduction，class alpha不啟用，避免額外class reweighting混因。γ=0必須等價CE。保留B0 negative soft Dice，Dice:Focal=1:1。γ=2常見於原始Focal研究但其object detection成效不保證適用DWI；γ=1為較弱modulation對照。
L2a：TI=(TP+s)/(TP+0.3FP+0.7FN+s)，s=1e-5；softmax lesion probability，逐sample空間加總後batch mean，背景不列Dice/Tversky class；loss=(1-TI)+CE，兩項1:1。保留CE以隔離Dice替換效果，loss相較B0有常數offset，曲線不能直接比大小。
L2b：loss=(1-TI)^(4/3)+CE，與L2a比較focal exponent。此處明定q>1偏重低TI樣本，不能照抄使用1/γ的函式參數。TI clamp[0,1]，分母clamp_min(1e-8)，float32累積；對exact0／1檢查finite gradients。不是宣稱完全復刻某篇Focal Tversky架構。
每個loss經相同DeepSupervisionWrapper套各有效輸出；2mm沿用4/7、2/7、1/7、0。padding -1先轉background；本dataset ignore_label=None，新增ignore支援時mask須一致套到所有components。
Boundary／Hausdorff loss、Transformer、attention、大幅architecture與heuristic postprocessing暫緩。

## 6. 固定 evaluation contract

在原始GT grid評估，固定final checkpoint、argmax、相同推論/TTA規則；不在fold0逐run調threshold。新增B0指標另存experiments/B0_baseline/evaluation_v2，不能改寫舊CSV/JSON。
Voxel/case：Dice mean、median、population SD(ddof0)、Q1/Q3、min、P10/P90（線性percentile）；HD95；volume MAE、median absolute error、mean/median signed error、mean/median signed percentage與absolute percentage error。Signed=Pred-GT；所有分母及有效n保留。Dice三位、mL兩位、百分比一位，原始數值保留完整precision。
雙空Dice N/A；GT>0 prediction空Dice0；GT空prediction非空Dice0。GT0 percentage N/A另報FP mL。HD95以GT/pred表面voxel centers、6-neighbor erosion界定表面、physical spacing算雙向最近距離，取兩方向各P95的max；雙空N/A、單空inf。報finite HD95分布與單空failure數，不以刪除inf後的平均代表全部病例。若affine非正交需使用完整world座標距離。
Robustness：Dice≥.8／≥.7／<.5／<.3、Pred/GT、severe under count（GT>0且ratio<.5）、catastrophic segmentation count（Dice<.3）、catastrophic miss（GT>0且lesion recall=0）。這些是本研究操作定義，不是臨床標準。
Lesion-level：binary mask用3D26-connectivity，不排除微小components；GT與prediction component有任意voxel overlap即eligible。最大matching cardinality優先、總IoU次之的一對一配對；merge不能同時算多個TP，split多出的prediction算FP；不另加distance匹配。TP=配對數，FN=未配對GT，FP=未配對prediction。輸出micro recall/precision/F1、每case count及空分母n；FN-only時recall0/F10、precision N/A，FP-only時precision0/F10、recall N/A，雙空三者N/A。
所有實驗固定case GT總體積分組[0,1)、[1,5)、[5,20)、[20,50)、[50,∞)mL；每組cases、valid Dice n、mean/median Dice、volume MAE、micro lesion recall。<1組含empty case但Dice不補1。Case總體積分層不能稱為individual connected-lesion size，另存component volume供後續研究。
B0 HD95及lesion recall目前資料未提供，須先重評50個既有mask才可比較；不從Dice或體積推估。

## 7. Experiment order 與 decision rules

先baseline protection及evaluator，再P1、P2、P3a、S1a、L1a、L2a各單因子。各run先短程20 warm-up+100 measured iterations，檢查memory/finite/geometry，不能用早期Dice淘汰。可行者各自完整1000 epochs。B0約13小時僅作參考，不以此乘所有不同resolution預算。
P3b、S1b、S2、L1b、L2b列secondary，主要候選完成後先看是否存在目標改善／FP tradeoff再追加，追加順序S1b→S2→L1b→L2b→P3b；所有變更先存新config/commit。不得未經結果就組合。
逐病例paired差異、case bootstrap10000次（seed2022）95%CI；subgroup在組內重抽，lesion micro指標用case抽樣避免把同case lesions視為獨立。報全部九項改善條件：mean/median Dice、1–5mL Dice、低分數、severe under、lesion recall、worst Dice、FP負擔、signed error距0。
先Pareto比較catastrophic miss、severe under與FP lesions/case、precision；無支配關係時保留tradeoff，不自動加權成單分數。優勝候選不得增加catastrophic miss或severe under，至少一項下降或small subgroup recall提高；若FP增加或precision下降則只能標tradeoff。再看small subgroup及overall Dice，全部未改善時保留B0。
單因子支持後按spacing→preprocess→sampling→loss逐步新增組合C1…C4，每步重新評估interaction，不自動把所有winner堆疊。原則上固定1000epochs；early stopping是未來獨立實驗。
主run random seed=2022（Python/NumPy/Torch/CUDA及worker種子需明確記錄），split沿用原始檔seed12345，不重新分折。B0歷史training seed若無完整紀錄標unknown，不把split seed當training seed。必要時新增同seed B0 replicate；優勝/B0補seed2023、2024配對重跑再談穩健性。CUDA不保證bitwise reproducibility。
Fold0為已用於開發的validation；定案後跑其餘folds，仍揭露模型選擇曾用fold0，不稱為完全獨立測試。嚴格泛化主張需要未使用外部資料或nested selection。

## 8. Multimodal stage（DWI-only定案後）

既有inventory：250例DWI/ADC/FLAIR/GT皆配對完整，DWI/ADC無記錄到幾何差異，FLAIR250例有差異；目前processed只存在Dataset001 DWI，不能稱ADC/FLAIR已完成preprocessing。
M0=固定improved DWI、M1=相同方法DWI+ADC、M2=再加FLAIR，各用新dataset ID並保持相同case split。ADC先做影像align/unit/range QA與每channel normalization；FLAIR先image-only配準到DWI，再做幾何及視覺QA，不能只改header。配準參數在training資料決定。通道變動會影響planning/mask，需揭露衍生configuration與channel-specific mask，避免原本DWI normalization被多channel union mask默默改變。

## 9. Version control、輸出與可見 Terminal

Baseline source commit=0aac38b；Git author使用Codex <codex@localhost>（未假冒使用者，未設定全域identity）。planning分支docs/improvement-plan。原始模型不入Git，由audit SHA256 manifest固定checkpoint、preprocessed/config/split、validation與metrics；commit不是大型artifact備份，原始檔仍須保存。
未來各方向從baseline加共同infra commit建立exp/spacing-1p5mm、exp/spacing-1mm、exp/clip-zscore、exp/foreground-sampling、exp/dice-focal、exp/focal-tversky。實驗ID包含variant及run suffix；資料夾experiments/<ID>/<run>/下config、commands、metrics、logs、notes，重型輸出不進Git。
experiment_config.json至少有ID、git hash/branch、dataset、trainer、plans/config/fold、split hash、random/worker seeds、spacing/patch/batch、loss/DS、sampling、preprocessing、environment、start/finish、status、evaluation checkpoint、input hashes。預跑config與完成metadata分開保存，既有完成結果只新增。
任何preprocess/planning/training/validation/predict前檢查stage目標輸出，不覆寫已有輸出；既有輸入可唯讀重用。衝突回報「Existing experiment output detected. No files were modified.」並另定新run ID。不自動刪舊checkpoint、不修改官方trainer或baseline plans。
使用者要求：每次實際執行開啟可見Terminal，stdout/stderr同步保存唯一log；命令列及開始/結束/exit code可見。長訓練顯示epoch、loss、pseudo-Dice、耗時與ETA，失敗保留Terminal。可見窗口不是訓練已啟動證據，以process/exit/log確認。
Comparison讀取各run metrics，新增outputs/comparisons/<comparison_id>/，不修改來源結果。

## 10. 後續實作驗收與來源

測試：baseline前後hash一致；split exactly equal且train/val不交集；normalization在更換GT後完全不變；empty/constant/NaN影像；S1驗證validation sampling未變、S2只有training權重；Focal γ0=CE；Tversky相同錯誤量FN penalty>FP；empty/perfect/extreme logits finite gradients；DS權重和1；空mask、merge/split、多component、anisotropic HD95與1/5/20/50mL邊界；輸出衝突零覆寫。
本地權威來源：MRI_DWI/Lib/site-packages/nnunetv2下experiment_planning/experiment_planners/default_experiment_planner.py、preprocessing/{cropping/cropping.py,normalization/default_normalization_schemes.py,preprocessors/default_preprocessor.py}、training/{dataloading/data_loader.py,nnUNetTrainer/nnUNetTrainer.py,loss/*}；baseline debug/log/plans及outputs/metrics/data/isles2022_inspection.json。
文獻依據：[Lin et al., Focal Loss](https://arxiv.org/abs/1708.02002)、[Salehi et al., Tversky loss](https://arxiv.org/abs/1706.05721)、[Tustison et al., N4ITK](https://pmc.ncbi.nlm.nih.gov/articles/PMC3071855/)。這些支持方法動機，不提供本專案改善幅度的證據；CLAHE/N4對本DWI的副作用為待驗證風險假設。
