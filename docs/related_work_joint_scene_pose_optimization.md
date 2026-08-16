# Joint Scene + Pose Optimization: Related Work

Literature survey of methods that optimize a scene representation and camera poses
*together*, as competitors / reference points for the gravity-regularized bundle
adjustment in `siclib/pose_estimation.py`.

Compiled 2026-08. BARF is the canonical entry point; this document places it in the
wider landscape and extracts the transferable insights.

---

## 0. How our BA differs from this literature

Our estimator (`AbsolutePoseEstimator`) is a **correspondence-based, prior-regularized
BA**: 2D-3D matches drive a reprojection objective, and GeoCalib's single-image gravity
estimate enters as a soft penalty (`gravity_weight`, `poselib_gravity` /
`pycolmap_gravity`) with a per-image uncertainty gate (`max_uncertainty`).

Nearly everything below is instead a **photometric / rendering-based joint
optimization**: poses are free variables in a differentiable renderer and are updated by
backpropagating an image-reconstruction loss. That difference is the whole story — it
determines the failure modes, and it is why the two lines of work keep converging on the
same fixes (feature tracks, monocular priors, preconditioning).

The competitive threat to a classical BA is **not** BARF. It is the 2024–2026
feed-forward pointmap models (§6), which replace the initialization *and* the
optimization in one forward pass.

---

## 1. The founding family — photometric BA on neural fields

These treat pose as a learnable parameter of a NeRF and rely purely on the photometric
loss. They established the problem and diagnosed its core pathology.

| Method | Venue | Core idea |
|---|---|---|
| **NeRF--** (NeRFmm) | arXiv 2021 | First to drop COLMAP entirely: jointly learn poses + focal length with the radiance field. Only works on forward-facing scenes with small baselines. |
| **SC-NeRF** | ICCV 2021 | Self-calibrates a *generic* camera model — pinhole + radial distortion + a learned per-ray non-linear residual. Adds a **projected ray distance** loss on correspondences alongside the photometric term. |
| **BARF** | ICCV 2021 (oral) | The reference work. |
| **GARF** | ECCV 2022 | Replaces ReLU+positional-encoding with **Gaussian activations**; the smoother network removes the need for coarse-to-fine scheduling. |
| **SiNeRF** | BMVC 2022 | SIREN (sinusoidal) activations + **Mixed Region Sampling** — pick ray batches around SIFT keypoints instead of uniformly. Explicitly frames NeRFmm's problem as *systematic sub-optimality* of joint optimization. |
| **L2G-NeRF** | CVPR 2023 | **Local-to-global**: first allow free per-pixel warps driven by photometric loss, then fit a rigid/global transform to them (differentiable Procrustes). Over-parameterizing the pose makes the landscape easier, then projects back onto SE(3). |
| **Invertible Neural Warp for NeRF** | 2024 | Represents pose as an invertible neural network rather than an SE(3) element — an alternative over-parameterization to L2G. |

### BARF's insight (the one everyone cites)

Two claims, and the second is the load-bearing one:

1. Joint pose+scene optimization is exactly classical image alignment, lifted to 3D. The
   same theory applies.
2. **Positional encoding actively sabotages registration.** High-frequency encoding bands
   produce high-frequency gradients w.r.t. pose, which shatter the basin of attraction.
   BARF applies a smooth mask over the encoding bands, annealed low→high during training
   — a **dynamic low-pass filter** on the loss landscape.

Read (2) as the general principle: *the alignment signal must be smoothed early and
sharpened late.* Every family below rediscovers this in its own representation (Gaussian
blur pyramids in §5, activation smoothness in GARF/SiNeRF, densification schedules in
KeyGS).

### Where this family breaks

- Needs near-GT initialization; object-centric or forward-facing scenes only.
- Small camera motion. Fails on 360°, long trajectories, and large baselines.
- No global consistency — purely per-scene overfitting, no notion of a track graph.

---

## 2. Adding geometric priors to widen the basin

The direct response to §1's fragility: inject correspondences and monocular geometry, i.e.
move *back toward* what a classical BA already has.

- **SPARF** (CVPR 2023, Truong et al.) — the most relevant to us. Given as few as **3
  wide-baseline views with noisy poses**, it runs a pretrained dense matcher and adds a
  **multi-view correspondence loss**: reproject using NeRF-rendered depth + current poses,
  minimize the residual against the matches. Plus a depth-consistency loss. This is
  literally reprojection-error BA with the depth coming from the field instead of from
  triangulated points.
- **NoPe-NeRF** (CVPR 2023) — undistorts monocular depth maps (learned per-frame scale/shift)
  and uses them for an inter-frame point-cloud loss plus a surface-based photometric loss.
  Pose priors come from the mono-depth network rather than from matching.
- **NoPe-NeRF++** (CGF 2025) — explicitly bolts a **local-to-global** structure on:
  feature-matching relative-pose init → local joint optimization → **global BA over feature
  trajectories**. Notable as an admission that pure photometric joint optimization needs a
  real BA underneath it.
- **TrackNeRF** (ECCV 2024) — uses **feature tracks** (not just pairwise matches) for global
  consistency under sparse and noisy views. Same motivation as NoPe-NeRF++.
- **TD-NeRF** (2024) — truncated depth prior, addresses the scale ambiguity/local-minimum
  coupling of mono-depth supervision.
- **CT-NeRF / RA-NeRF** (2024–2025) — incremental optimization targeting **complex
  trajectories**, where the sequential assumption of earlier work fails.
- **Robust SG-NeRF** (2024) — scene-graph aided, explicitly handles **outlier poses** rather
  than assuming all inputs are merely noisy.

**Insight:** every serious attempt to scale photometric joint optimization ends up
re-adding correspondences, tracks, and a global optimization stage. The gap between "BARF"
and "BA" closes from the BARF side.

---

## 3. Conditioning and parameterization — the deepest insight in the area

**CamP: Camera Preconditioning for Neural Radiance Fields** (Park, Henzler, Mildenhall,
Barron, Martin-Brualla; SIGGRAPH Asia / TOG 2023) is the paper worth reading closely.

Argument: joint pose+scene optimization fails less because of the loss landscape's shape
than because camera parameters are **badly scaled and strongly correlated** with each
other. A millimeter of translation, a milliradian of rotation, and a pixel of focal length
have wildly different effects on the image, and their effects overlap. First-order
optimizers on such a parameterization crawl in some directions and diverge in others.

Fix: solve a cheap **proxy problem** (how much does each camera parameter move projected
points?), compute a **whitening transform** from it, and use that as a **preconditioner** on
the camera parameters. Cheap, representation-agnostic, and a large win.

Related: **"From Variance to Veracity: Unbundling and Mitigating Gradient Variance in
Differentiable Bundle Adjustment Layers"** (2024) — analyses why differentiable BA layers
produce high-variance gradients, and how to fix it.

**Why this matters for us:** this is the one place where the neural-rendering literature
found something a classical BA does *not* automatically get. Gauss-Newton / Levenberg-
Marquardt on reprojection error is already implicitly preconditioned by the Hessian
approximation — that is a structural advantage of our BA over every first-order method
in §1–§2, and a good thing to state explicitly. Conversely, CamP's whitening argument
applies directly to how we weight the gravity term: `gravity_weight = 50000` against a
reprojection residual in pixels is exactly the kind of arbitrary cross-parameter scaling
CamP formalizes. Deriving that weight from a proxy/uncertainty analysis rather than a
constant is a concrete, defensible improvement.

---

## 4. Progressive / scalable optimization

- **LocalRF** (CVPR 2023, Meuleman et al.) — for casually captured long videos. Two ideas:
  (a) **progressive** frame-by-frame insertion with joint pose estimation, so each new pose
  starts near its neighbours' converged solution; (b) **dynamically allocated local radiance
  fields** over temporal windows, since one global field cannot represent a long trajectory.
- **KeyGS** (2024) — keyframe-centric 3DGS for monocular sequences, with **coarse-to-fine
  frequency-aware densification** that prevents premature Gaussian splitting on high
  frequencies. The 3DGS analogue of BARF's encoding schedule.

**Insight:** progressive/incremental scheduling is the single most reliable robustness
trick in the whole area — which is, of course, exactly what incremental SfM has always
done.

---

## 5. 3D Gaussian Splatting joint optimization

The active frontier. Note that 3DGS *worsens* the optimization problem relative to NeRF.

### Why 3DGS pose optimization is harder

Gaussian primitives have **compact local support**. If the current pose puts a rendered
primitive more than a few standard deviations from where it belongs, the gradient is not
just small — it is *identically zero*. There is no smooth global field to carry a signal
across the gap. Additionally, even when gradients exist, there must be a monotonically
descending path, and primitives often have to cross empty space (occluding background)
to reach their correct positions. Vanilla 3DGS also provides **no analytic pose Jacobian**
at all — it assumes posed inputs — so each of these methods derives one.

### Optimization-based

| Method | Venue | Contribution |
|---|---|---|
| **CF-3DGS** | CVPR 2024 | COLMAP-free, sequential. Fits a *local* 3DGS per frame and uses it as a relative-pose estimator between adjacent views, then grows a global set. Works on 360° video. |
| **iComMa** | 2023 | Inverts 3DGS for pose estimation by combining a rendering-comparison loss with an explicit **matching** loss — the matching term supplies gradients where photometric ones vanish. |
| **InstantSplat** | 2024 | **DUSt3R** for coarse geometric init, then fast joint refinement of Gaussians + poses. Sparse unposed images → result in ~40 s. The "good init makes joint optimization easy" thesis. |
| **3R-GS** | 2025 | *Best-practice study.* Init from **MASt3R-SfM**, refine poses with an MLP pose refiner, regularize with **epipolar constraints**. Explicitly names the two failure modes: sensitivity to SfM init quality, and lack of global optimization capacity. Closest thing to a systematic ablation of what actually matters. |
| **PCR-GS** | ICCV 2025 | **Pose co-regularization** — DINO feature correspondence + wavelet-frequency regularization, for scenes with large camera motion. |
| **JOGS** | 2025 | Decouples the joint problem into **interleaved phases**: (a) update Gaussians with poses fixed via differentiable rendering, (b) update poses with a custom **3D optical flow** using geometric + photometric constraints. Alternating minimization rather than simultaneous gradient descent. |
| **GloSplat**, **Pi3DGS** | 2025–2026 | Joint pose-appearance optimization; Pi3-based init with 2D/3D gradient filters for stability. Both report beating COLMAP-based baselines. |
| **RegGS** | 2025 | Unposed sparse views via **3DGS registration** (Procrustes / MMD alignment of Gaussian mixtures) rather than gradient-based pose refinement. |
| **GSLoc** | 2024 | Localization, but the trick generalizes: **progressively decaying Gaussian blur** applied to both rendered and target images. Direct import of BARF's low-pass insight into 3DGS, at image level rather than encoding level. |

### GS-SLAM (joint tracking + mapping, online)

**MonoGS** / Gaussian Splatting SLAM (CVPR 2024) — derives the **analytic pose Jacobian** on
the SE(3) manifold and adds isotropic regularization. **SplaTAM** (CVPR 2024) — optimizes
pose and map by minimizing photometric + depth rendering losses. **Photo-SLAM** — ORB-SLAM3
for tracking, coarse-to-fine map optimization. **Splat-SLAM** (2025) — globally optimized
RGB-only with loop closure. **VGGT-SLAM** (2025) — dense RGB SLAM optimized on the SL(4)
manifold, on top of VGGT.

**Insight:** the SLAM community solved this by *not* trusting photometric pose gradients
alone — decoupled tracking, keyframing, loop closure, and global BA. Same conclusion as §2.

---

## 6. Feed-forward pose-free reconstruction — the actual competition

This is where the field moved, and it is the line that most directly threatens a
classical BA. Instead of *optimizing* poses, predict geometry and poses in one forward
pass.

**Backbones:**
- **DUSt3R** — dense pointmap in the first view's frame from an uncalibrated pair; pose,
  depth, correspondences recovered by post-processing. Global alignment stage for >2 views.
- **MASt3R / MASt3R-SfM** — adds matching heads and descriptors; enables real SfM.
- **VGGT** (CVPR 2025) — a single transformer ingests all views and emits cameras, depth,
  and pointmaps in one pass. Scales far better than pairwise DUSt3R.
- **CUT3R** — recurrent/streaming reformulation. **Fast3R**, **Spann3R**, **MV-DUSt3R** — related.
- **Pi3** — used as init by Pi3DGS and others.

**Pose-free splatting on top:**
- **NoPoSplat** (ICLR 2025) — predicts Gaussians directly in a **canonical space** from
  sparse unposed images, no iterative optimization at all.
- **Splatt3R** — zero-shot Gaussian splatting from uncalibrated pairs (DUSt3R + Gaussian heads).
- **AnySplat** (TOG 2025) — unconstrained image collections → Gaussians **and** intrinsics/
  extrinsics simultaneously; differentiable voxelization to aggregate.
- **SPFSplat / SPFSplatV2** (2025) — self-supervised pose-free from sparse views.
- **SHARE** (2025) — joint **shape and camera-ray** estimation into a pose-aware canonical
  volume, to absorb pose error rather than propagate it.

**Directly relevant to GeoCalib — worth reading first:**
**"G3T Up! Gravity Aligned Coordinate Frames Simplify Pointmap Processing"** (Nagoor Kani &
Snavely, arXiv 2605.27372, May 2026). Predicts pointmaps in an **upright, gravity-aligned
frame** instead of a camera-centric one. The argument is precisely ours: gravity-aligned
frames share a common vertical axis across viewpoints, so **the rotational DoF relating two
views drops from 3 to 1**. `G3T-Long` exploits this in a submap-based incremental pipeline
for large accuracy gains.

This is the strongest external validation of GeoCalib's premise *and* the most direct
competitor to a gravity-regularized BA — it bakes the gravity prior into the
representation rather than adding it as a penalty term. Position our work relative to it
explicitly.

Also: **MASt3R-Fusion** (2026) — integrates the feed-forward visual model with **IMU and
GNSS**, i.e. a hard gravity/orientation prior, for SLAM.

---

## 7. Classical and learned SfM/BA baselines

The other side of the comparison — what a modern non-rendering pipeline gives you.

- **GLOMAP** (ECCV 2024) — global SfM revisited; merges translation averaging and
  triangulation into a single **global positioning** step. Matches incremental COLMAP's
  accuracy at global-SfM speed, and beats COLMAP.
- **VGGSfM** (CVPR 2024) — fully differentiable end-to-end SfM: 2D tracks → camera init →
  point cloud → **differentiable BA layer**.
- **Detector-Free SfM** (CVPR 2024) — detector-free matching; succeeds in texture-poor
  scenes where keypoint-based SfM fails.
- **DROID-SLAM** — recurrent iterative updates of pose and per-pixel depth through a
  **dense BA layer**. Still the robustness reference point.
- **BA-Net** (ICLR 2019) — the original dense/learned BA; predicts a basis of depth maps and
  optimizes the damping factor.
- **DBARF** (CVPR 2023) — bundle-adjusts poses for **generalizable** NeRFs using a cost feature
  map as an implicit cost function. Unlike BARF/GARF it generalizes across scenes and needs
  no good initialization. The bridge between §1 and this section.
- **MegaSaM** (2024) — structure and motion from casual **dynamic** video.
- **Bundle Adjustment in the Eager Mode** (2024) — PyTorch-native BA; useful engineering
  reference if we want our BA differentiable and composable with learned priors.

---

## 8. Cross-cutting insights

Condensed, in rough order of how much they should influence our work.

1. **Smooth the objective early, sharpen it late.** BARF (encoding mask), GARF/SiNeRF
   (activation smoothness), GSLoc/SLAIM (Gaussian blur pyramid), KeyGS (densification
   schedule). Universally rediscovered. Our correspondence-based BA gets this partly for
   free through robust loss scaling (`loss_function_scale`, `max_reproj_error = 48.0` — a
   deliberately loose threshold that plays the same role as a wide basin).

2. **Preconditioning beats landscape engineering.** CamP: the dominant failure mode is
   correlated, badly scaled camera parameters, not the loss shape. Second-order methods on
   reprojection error are structurally advantaged here — argue this. And apply the same
   lens to our gravity weight.

3. **Photometric pose gradients are insufficient alone; everyone re-adds correspondences.**
   SPARF, SC-NeRF, TrackNeRF, iComMa, PCR-GS, 3R-GS, NoPe-NeRF++. The trajectory of this
   literature runs *toward* classical BA, not away from it. Strong framing for our related
   work section.

4. **Initialization dominates.** InstantSplat, 3R-GS, and Pi3DGS all report that a good
   feed-forward init makes joint refinement nearly trivial, and 3R-GS names init quality as
   one of its two failure modes. Joint optimization is a *refinement* technique, not a
   replacement for initialization.

5. **Alternate, don't co-descend.** JOGS interleaves (Gaussians | poses fixed) and (poses |
   Gaussians fixed). L2G-NeRF over-parameterizes then projects. Simultaneous gradient
   descent on both blocks is the weakest option.

6. **Global structure beats local consistency.** SC-NeRF's pairwise ray distances lack
   global constraints; NoPe-NeRF++ and TrackNeRF fix this with feature *tracks*; GLOMAP
   makes the same point for classical SfM. Pairwise → tracks → global is the recurring
   upgrade path.

7. **Gaussians have compact support, so gradients genuinely vanish** — not merely shrink.
   Any 3DGS pose-refinement comparison must account for this; it is why 3DGS methods lean
   harder on matching losses and good init than NeRF methods did.

8. **Priors as representation, not as penalty.** G3T's gravity-aligned frames reduce
   inter-view rotational DoF from 3 to 1 by construction. Compare against our soft-penalty
   formulation (`gravity_weight`, uncertainty gating) — the soft penalty is more flexible
   and degrades gracefully when GeoCalib is uncertain, which is a real advantage worth
   measuring, not assuming.

---

## 9. Positioning and open gaps

Where our gravity-regularized BA is genuinely differentiated:

- **Prior-as-soft-constraint with calibrated uncertainty.** `max_uncertainty` gating means a
  bad gravity estimate is discarded rather than propagated. G3T and MASt3R-Fusion bake the
  prior in hard; none of the §1–§5 methods use a *learned, uncertainty-aware* geometric
  prior on the camera at all. This is the clearest novelty axis.
- **Single-image prior, no sequence assumption.** CF-3DGS, LocalRF, KeyGS, and the GS-SLAM
  family all assume video / temporal ordering. GeoCalib's prior is per-image and works on
  unordered collections.
- **Second-order optimization on reprojection error**, versus first-order photometric
  descent — the CamP argument, in our favour.

Gaps worth closing before publication:

- No comparison against a feed-forward pointmap baseline (VGGT / MASt3R-SfM / Pi3) with and
  without a gravity prior. This is the obvious reviewer question in 2026.
- The `gravity_weight` constant is unjustified. A CamP-style whitening/proxy derivation, or
  weighting by GeoCalib's predicted uncertainty, would be principled and is cheap to do.
- No experiment on large-baseline / low-overlap pairs, which is exactly where a gravity
  prior should help most and where photometric joint optimization is known to collapse.
- G3T (May 2026) needs an explicit discussion — hard gravity-aligned representation vs. our
  soft gravity constraint.

---

## 10. Suggested reading order

1. **BARF** — the framing and the low-pass insight.
2. **CamP** — the deepest optimization insight; directly informs our weighting.
3. **SPARF** — closest to us in spirit (correspondence-driven, sparse, noisy poses).
4. **3R-GS** — best-practice ablation for the 3DGS era; tells you what actually matters.
5. **G3T Up!** — the gravity-prior competitor; read before writing related work.
6. **GLOMAP** + **VGGT** — the two baselines any reviewer will ask about.

---

## References

- BARF: Bundle-Adjusting Neural Radiance Fields — arXiv:2104.06405, ICCV 2021
- NeRF--: Neural Radiance Fields Without Known Camera Parameters — arXiv:2102.07064
- Self-Calibrating Neural Radiance Fields (SC-NeRF) — arXiv:2108.13826, ICCV 2021
- GARF: Gaussian Activated Radiance Fields — arXiv:2204.05735
- SiNeRF: Sinusoidal Neural Radiance Fields — arXiv:2210.04553, BMVC 2022
- L2G-NeRF: Local-to-Global Registration for Bundle-Adjusting NeRF — CVPR 2023
- Invertible Neural Warp for NeRF — arXiv:2407.12354
- NoPe-NeRF: Optimising NeRF with No Pose Prior — arXiv:2212.07388, CVPR 2023
- NoPe-NeRF++: Local-to-Global Optimization of NeRF with No Pose Prior — arXiv:2511.17322, CGF 2025
- SPARF: Neural Radiance Fields from Sparse and Noisy Poses — arXiv:2211.11738, CVPR 2023
- TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks — arXiv:2408.10739
- TD-NeRF: Truncated Depth Prior for Joint Camera Pose and NeRF Optimization — arXiv:2405.07027
- CT-NeRF: Incremental Optimizing NeRF and Poses with Complex Trajectory — arXiv:2404.13896
- RA-NeRF: Robust NeRF Reconstruction with Accurate Camera Pose Estimation — arXiv:2506.15242
- Robust SG-NeRF: Robust Scene Graph Aided Neural Surface Reconstruction — arXiv:2411.13620
- CamP: Camera Preconditioning for Neural Radiance Fields — arXiv:2308.10902, TOG 2023
- From Variance to Veracity: Gradient Variance in Differentiable BA Layers — arXiv:2406.07785
- Progressively Optimized Local Radiance Fields (LocalRF) — arXiv:2303.13791, CVPR 2023
- KeyGS: Keyframe-Centric Gaussian Splatting — arXiv:2412.20767
- COLMAP-Free 3D Gaussian Splatting (CF-3DGS) — arXiv:2312.07504, CVPR 2024
- iComMa: Inverting 3DGS for Camera Pose Estimation — arXiv:2312.09031
- InstantSplat — unbounded pose-free Gaussian splatting in 40 seconds
- 3R-GS: Best Practice in Optimizing Camera Poses Along with 3DGS — arXiv:2504.04294
- PCR-GS: COLMAP-Free 3DGS via Pose Co-Regularizations — arXiv:2507.13891, ICCV 2025
- JOGS: Joint Optimization of Pose Estimation and 3D Gaussian Splatting — arXiv:2510.26117
- GloSplat: Joint Pose-Appearance Optimization — arXiv:2603.04847
- Pi3DGS: Robust Joint Optimization of Camera Poses and 3DGS from Uncalibrated Images
- RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration — arXiv:2507.08136
- GSLoc: Visual Localization with 3D Gaussian Splatting — arXiv:2410.06165
- SLAIM: Robust Dense Neural SLAM for Online Tracking and Mapping — arXiv:2404.11419
- Gaussian Splatting SLAM (MonoGS) — CVPR 2024
- SplaTAM — CVPR 2024
- Splat-SLAM: Globally Optimized RGB-only SLAM with 3D Gaussians — CVPRW 2025
- VGGT-SLAM: Dense RGB SLAM Optimized on the SL(4) Manifold — arXiv:2505.12549
- DUSt3R / MASt3R / MASt3R-SfM
- VGGT: Visual Geometry Grounded Transformer — CVPR 2025
- CUT3R, Fast3R, Pi3 — feed-forward pointmap models
- Review of Feed-forward 3D Reconstruction: From DUSt3R to VGGT — arXiv:2507.08448
- NoPoSplat: No Pose, No Problem — ICLR 2025
- Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs
- AnySplat: Feed-forward 3DGS from Unconstrained Views — TOG 2025
- SPFSplat / SPFSplatV2 — arXiv:2508.01171, arXiv:2509.17246
- SHARE: Pose-free 3D Gaussian Splatting via Shape-Ray Estimation — arXiv:2505.22978
- G3T Up! Gravity Aligned Coordinate Frames Simplify Pointmap Processing — arXiv:2605.27372
- MASt3R-Fusion: feed-forward visual model with IMU and GNSS
- Global Structure-from-Motion Revisited (GLOMAP) — arXiv:2407.20219, ECCV 2024
- VGGSfM: Visual Geometry Grounded Deep Structure From Motion — CVPR 2024
- Detector-Free Structure from Motion — arXiv:2306.15669, CVPR 2024
- DROID-SLAM — NeurIPS 2021
- BA-Net: Dense Bundle Adjustment Network — ICLR 2019
- DBARF: Deep Bundle-Adjusting Generalizable NeRF — arXiv:2303.14478, CVPR 2023
- MegaSaM: Structure and Motion from Casual Dynamic Videos — arXiv:2412.04463
- Bundle Adjustment in the Eager Mode — arXiv:2409.12190
- A Survey on Pose-Free Neural Radiance Fields and 3D Gaussian Splatting — TechRxiv, 2026
