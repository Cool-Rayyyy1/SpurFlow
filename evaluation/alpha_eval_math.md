# FLUX.2-Klein Softsign-01 Alpha 学生:评测时 alpha 的完整数学流程

对应脚本:`evaluation/run_flux2_klein_uedit_alpha_gedit_infer.sh`
配置链:`editflux2_klein_uedit_fixedeps_2nfe_k16_alpha_softsign_data_oss_pico.py`
推理路径:`ArcFlowEditAlphaImitation.forward_test`(继承自 `ArcFlowEditImitation`,
policy 为 `ArcFlowEditNewAlpha`),2 NFE,guidance=1.0(无 CFG),无 step2_direct_x0,
**两步都走动量积分**。

---

## 0. 记号与初始化

```
x_ref = VAE.encode(源图)                    # latent, 形状 C×H×W
eps   ~ N(0, I)                             # 固定路径噪声(fixedeps),整条轨迹共用
x(1)  = eps                                 # 起点

时间网格 (nfe=2, timestep_ratio=1):
  原始 t: 1 -> 1/2 -> 0,每段 0.5

时间弯曲 (sampler.py:46, shift s=3.2):
  sigma(t) = s*t / (1 + (s-1)*t)
  sigma(1) = 1,  sigma(0.5) = 1.6/2.1 ≈ 0.762,  sigma(0) = 0
```

## 1. 每个 NFE 步的网络输出

在 sigma_a ∈ {1, 0.762} 处各前向一次(`arcflux2_edit_alpha_softsign.py:317-340`),
对每个 latent 像素输出:

| 输出 | 形状(unpatchify 后) | 含义 |
|---|---|---|
| `deltax`  D_i, i=1..16 | (16, C, H, W) | GM 混合的 16 个残差分支 |
| `logweights` -> w = softmax_K | (16, 1, H, W) | 混合权重 π(逐像素) |
| `loggammas` gamma_i, i=2..16 | (15, 1, H, W) | 各分支时间衰减率(第 1 支恒为 0) |
| `alpha` | (1, 1, H, W) | 空间门控标量 |

**alpha 的产生**:proj_out_alpha 对每个 token 输出 4 个 logit
(对应 2×2 patch 内 4 个位置),经 Softsign-01 激活:

```
alpha = 0.5 * ( a_raw / (1 + |a_raw|) + 1 )      ∈ (0, 1),零 logit -> 0.5
```

然后 unpatchify 成逐 latent 像素的标量图。alpha 的语义:**该像素保留多少源图 latent**。

## 2. 学生的速度场(policy 定义)

`arcflow_edit_new_alpha.py:35-43`,段内时刻 sigma 的瞬时速度:

```
u(sigma) = eps  -  alpha ⊙ x_ref  -  Delta(sigma)

Delta(sigma) = Σ_{i=1..16}  w_i * D_i * exp( gamma_i * (sigma_a - sigma) )
```

评测时每步都是新预测(sigma_src = sigma_a),所以初始 decay = 1。
**alpha 在整个推理里唯一的作用**:把 vanilla 版速度中的 x_ref
换成 alpha ⊙ x_ref(逐像素缩放)。

## 3. 一步动量积分(闭式,`arcflow_edit.py:196-212` + `arcflow.py:55-75`)

从 sigma_a 积到 sigma_b,记 dt = sigma_a - sigma_b。三项分别积分:

- eps 与 alpha⊙x_ref 在段内是常数,积分精确;
- 混合项用 expm1 闭式积分(gamma_1 = 0 时积分因子 = 1):

```
x_b = x_a
      - eps * dt                                   # 噪声项
      + (alpha ⊙ x_ref) * dt                       # ← alpha 进入的地方
      + Σ_i  w_i * D_i * dt * (exp(gamma_i*dt) - 1) / (gamma_i*dt)   # 混合残差
```

## 4. 两步展开与终点

```
step 1: 在 (eps,  sigma=1)     前向 -> alpha(1), w(1), D(1), gamma(1)
        积分 sigma: 1 -> 0.762   (dt = 0.238)   得 x_1

step 2: 在 (x_1, sigma=0.762)  重新前向 -> alpha(2), w(2), D(2), gamma(2)
        积分 sigma: 0.762 -> 0   (dt = 0.762)   得 x_2

图像 = VAE.decode(x_2)
```

理想极限(网络预测自洽)下,从 1 积到 0 时 eps 项恰好抵消起点 x(1)=eps,
终点收敛到编辑分解:

```
x(0)  ≈  alpha ⊙ x_ref  +  Σ_i w_i * D_i * (gamma 积分因子)
         └─ 源图保留 ─┘     └───── 编辑残差 pred_delta ─────┘
```

即 **alpha 门控的源图保留 + GM 混合建模的编辑增量**。

---

## 与 qwen step2_alph_dino_gan 模型的推理差异

| | klein alpha(本脚本) | qwen step2_alph_dino_gan |
|---|---|---|
| diffusion 类 | ArcFlowEditAlphaImitation | ArcFlowEditImitationStep2GAN |
| 第 1 步 | 动量积分 | 动量积分(相同) |
| 第 2 步 | **仍是速度积分**,alpha 以 `alpha*x_ref*dt` 进入积分式 | **端点捷径**:直接 `x0 = alpha*x_ref + pred_delta` |

由于最后一段 dt 积满到 sigma=0,两者数学上基本等价:klein 版只是混合残差
多了 gamma 衰减的精确积分修正,alpha 的角色(端点处源图保留系数)一致。
