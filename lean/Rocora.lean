/-
# Rocora: fixed-point curvature and probing algebra

Retained mathematical development for the established Hessian decomposition
H = S + R, compositional second derivatives, and coded sign probing of a fixed
symmetric matrix. The core positivity and residual linearity statements apply
under their explicit hypotheses and in exact arithmetic.

This file does not establish a novel decomposition, monotone residual decay,
zero core-sampling noise, improved neural-network updates, TPU speed, or
nonconvex optimizer convergence. The experimental question and numerical
implementation are described separately in theory.md.

Names and namespace were changed from the original development to Rocora.
-/
import Mathlib

open Finset Matrix ContinuousLinearMap

noncomputable section

namespace Rocora

variable {R : Type*} [CommRing R] {n m p q : ℕ}

/-! ## §1 The carrier algebra -/

/-- The **curvature carrier** of a map `f : Rⁿ → Rᵐ` at a point: the Jacobian `J`
together with the adjoint-curvature map `H`, where `H λ = Σ_k λ_k ∇²f_k = ∇²⟨λ, f⟩`. -/
structure Carrier (R : Type*) [CommRing R] (n m : ℕ) where
  J : Matrix (Fin m) (Fin n) R
  H : (Fin m → R) → Matrix (Fin n) (Fin n) R
  H_add : ∀ a b, H (a + b) = H a + H b
  H_smul : ∀ (c : R) (a), H (c • a) = c • H a
  H_symm : ∀ lam, (H lam).IsSymm

theorem Carrier.ext {C₁ C₂ : Carrier R n m}
    (hJ : C₁.J = C₂.J) (hH : ∀ lam, C₁.H lam = C₂.H lam) : C₁ = C₂ := by
  obtain ⟨J₁, H₁, _, _, _⟩ := C₁
  obtain ⟨J₂, H₂, _, _, _⟩ := C₂
  simp only at hJ hH
  subst hJ
  obtain rfl : H₁ = H₂ := funext hH
  rfl

/-- The zero adjoint carries zero curvature. -/
theorem Carrier.H_zero (C : Carrier R n m) : C.H 0 = 0 := by
  have h := C.H_smul 0 0
  simpa using h

/-- Pullback composition: the carrier of `g ∘ f` from the carriers of `g` and `f`. -/
def comp (Cg : Carrier R m p) (Cf : Carrier R n m) : Carrier R n p where
  J := Cg.J * Cf.J
  H := fun lam => Cf.Jᵀ * Cg.H lam * Cf.J + Cf.H (Cg.Jᵀ *ᵥ lam)
  H_add := by
    intro a b
    rw [mulVec_add, Cg.H_add, Cf.H_add, Matrix.mul_add, Matrix.add_mul]
    abel
  H_smul := by
    intro c a
    rw [mulVec_smul, Cg.H_smul, Cf.H_smul, Matrix.mul_smul, Matrix.smul_mul, smul_add]
  H_symm := by
    intro lam
    apply Matrix.IsSymm.add ?_ (Cf.H_symm _)
    unfold Matrix.IsSymm
    rw [Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose,
      (Cg.H_symm lam).eq, Matrix.mul_assoc]

@[inherit_doc] scoped infixl:65 " ⋆ " => comp

/-- The identity carrier: `J = 1`, `H = 0`. -/
def idC (R : Type*) [CommRing R] (n : ℕ) : Carrier R n n where
  J := 1
  H := fun _ => 0
  H_add := by simp
  H_smul := by simp
  H_symm := fun _ => Matrix.isSymm_zero

@[simp] theorem comp_J (Cg : Carrier R m p) (Cf : Carrier R n m) :
    (Cg ⋆ Cf).J = Cg.J * Cf.J := rfl

@[simp] theorem comp_H (Cg : Carrier R m p) (Cf : Carrier R n m) (lam : Fin p → R) :
    (Cg ⋆ Cf).H lam = Cf.Jᵀ * Cg.H lam * Cf.J + Cf.H (Cg.Jᵀ *ᵥ lam) := rfl

/-- **Associativity**: the carrier of a deep composition is bracketing-independent. -/
theorem comp_assoc (Ch : Carrier R p q) (Cg : Carrier R m p) (Cf : Carrier R n m) :
    (Ch ⋆ Cg) ⋆ Cf = Ch ⋆ (Cg ⋆ Cf) := by
  apply Carrier.ext
  · simp [Matrix.mul_assoc]
  · intro lam
    simp only [comp_H, comp_J, Matrix.transpose_mul, Matrix.mul_add, Matrix.add_mul,
      Matrix.mul_assoc, mulVec_mulVec]
    abel

theorem id_comp (Cf : Carrier R n m) : idC R m ⋆ Cf = Cf := by
  apply Carrier.ext
  · simp [idC]
  · intro lam
    simp [idC, one_mulVec]

theorem comp_id (Cf : Carrier R n m) : Cf ⋆ idC R n = Cf := by
  apply Carrier.ext
  · simp [idC]
  · intro lam
    simp [idC, Matrix.transpose_one]

/-! ## §2 Layer carriers, the loss Hessian, and the split -/

/-- Carrier of an affine layer `x ↦ Ax + b`: zero curvature. -/
def affineCarrier (A : Matrix (Fin m) (Fin n) R) : Carrier R n m where
  J := A
  H := fun _ => 0
  H_add := by simp
  H_smul := by simp
  H_symm := fun _ => Matrix.isSymm_zero

/-- Carrier of a pointwise nonlinearity at a point: `J = diag φ'`, `H λ = diag (λ ⊙ φ″)`.
Storage `2n` scalars — linear, never quadratic. -/
def diagCarrier (d h : Fin n → R) : Carrier R n n where
  J := diagonal d
  H := fun lam => diagonal (lam * h)
  H_add := fun a b => by
    ext i j
    by_cases hij : i = j <;> simp [hij, add_mul]
  H_smul := fun c a => by
    ext i j
    by_cases hij : i = j <;> simp [hij]
  H_symm := fun lam => isSymm_diagonal _

/-- Carrier of the quadratic form `x ↦ ½ xᵀAx` (`A` symmetric) at `x₀`. -/
def quadCarrier (A : Matrix (Fin n) (Fin n) R) (hA : A.IsSymm) (x₀ : Fin n → R) :
    Carrier R n 1 where
  J := Matrix.of fun _ j => (A *ᵥ x₀) j
  H := fun lam => lam 0 • A
  H_add := fun a b => by simp [add_smul]
  H_smul := fun c a => by simp [smul_smul]
  H_symm := fun lam => by
    unfold Matrix.IsSymm
    rw [Matrix.transpose_smul, hA.eq]

/-- The Hessian encoded by a scalar carrier: evaluate `H` at `λ = 1`. -/
def hess (C : Carrier R n 1) : Matrix (Fin n) (Fin n) R := C.H 1

theorem hess_symm (C : Carrier R n 1) : (hess C).IsSymm := C.H_symm 1

/-- The **structural core** of a loss over features: `S = J_fᵀ Λ J_f`
(`Λ` = loss Hessian at the features). -/
def structCore (Cf : Carrier R n m) (Lam : Matrix (Fin m) (Fin m) R) :
    Matrix (Fin n) (Fin n) R :=
  Cf.Jᵀ * Lam * Cf.J

/-- The **residual curvature** of a loss over features: `R = ℋ_f(∇ℓ)`
(feature curvature weighted by the loss gradient). -/
def residual (Cf : Carrier R n m) (g : Fin m → R) : Matrix (Fin n) (Fin n) R :=
  Cf.H g

/-- **The split (exact).** For a loss layer `Cℓ` over features `Cf`, the loss Hessian
decomposes exactly — not approximately — as structural core + residual:
`∇²(ℓ∘f) = J_fᵀ (∇²ℓ) J_f + ℋ_f(∇ℓ)`. -/
theorem split_exact (Cl : Carrier R m 1) (Cf : Carrier R n m) :
    hess (Cl ⋆ Cf) = structCore Cf (Cl.H 1) + residual Cf (Cl.Jᵀ *ᵥ 1) := rfl

/-- **The residual is linear in the loss gradient** — the mechanism behind vanishing
probe variance near critical points (§8). -/
theorem residual_smul (Cf : Carrier R n m) (c : R) (g : Fin m → R) :
    residual Cf (c • g) = c • residual Cf g :=
  Cf.H_smul c g

@[simp] theorem residual_zero (Cf : Carrier R n m) : residual Cf 0 = 0 :=
  Cf.H_zero

/-- Any pullback `Jᵀ M J` through `J : m × n` has rank at most `m`: the structural core
is a *width-rank* matrix. -/
theorem rank_core_le [StrongRankCondition R]
    (M : Matrix (Fin m) (Fin m) R) (J : Matrix (Fin m) (Fin n) R) :
    (Jᵀ * M * J).rank ≤ m := by
  calc (Jᵀ * M * J).rank ≤ min (Jᵀ * M).rank J.rank := Matrix.rank_mul_le _ _
    _ ≤ J.rank := min_le_right _ _
    _ ≤ Fintype.card (Fin m) := J.rank_le_card_height
    _ = m := Fintype.card_fin m

@[simp] theorem quadCarrier_hess (A : Matrix (Fin n) (Fin n) R) (hA : A.IsSymm)
    (x₀ : Fin n → R) : hess (quadCarrier A hA x₀) = A := by
  simp [hess, quadCarrier]

/-- **Range of `hess` = the symmetric matrices**: carriers realize every symmetric
matrix and only those. Exact curvature costs `dim Sym(n)` degrees of freedom on
unstructured functions; all compression must come from compositional structure. -/
theorem range_hess : Set.range (hess : Carrier R n 1 → _) = {A | A.IsSymm} := by
  apply Set.eq_of_subset_of_subset
  · rintro A ⟨C, rfl⟩
    exact hess_symm C
  · intro A hA
    exact ⟨quadCarrier A hA 0, quadCarrier_hess A hA 0⟩

/-! ## §3 Matrix-free evaluation: the Hessian as a program, not a matrix -/

/-- One-composition Hessian-vector product from the factors only. -/
def hvp₂ (Cg : Carrier R m p) (Cf : Carrier R n m) (lam : Fin p → R) (v : Fin n → R) :
    Fin n → R :=
  Cf.Jᵀ *ᵥ (Cg.H lam *ᵥ (Cf.J *ᵥ v)) + Cf.H (Cg.Jᵀ *ᵥ lam) *ᵥ v

theorem hvp₂_eq (Cg : Carrier R m p) (Cf : Carrier R n m) (lam : Fin p → R)
    (v : Fin n → R) : hvp₂ Cg Cf lam v = (Cg ⋆ Cf).H lam *ᵥ v := by
  simp [hvp₂, add_mulVec, mulVec_mulVec, Matrix.mul_assoc]

theorem column_recovery (A : Matrix (Fin n) (Fin n) R) (j : Fin n) :
    A *ᵥ Pi.single j 1 = fun i => A i j := by
  ext i; simp

theorem entry_recovery (A : Matrix (Fin n) (Fin n) R) (i j : Fin n) :
    Pi.single i 1 ⬝ᵥ (A *ᵥ Pi.single j 1) = A i j := by
  rw [column_recovery]
  simp

/-- Compose a list of layer carriers, head = outermost. -/
def compChain : List (Carrier R n n) → Carrier R n n
  | [] => idC R n
  | c :: t => c ⋆ compChain t

def chainJ : List (Carrier R n n) → Matrix (Fin n) (Fin n) R
  | [] => 1
  | c :: t => c.J * chainJ t

/-- **Sum-of-pullbacks normal form** of a chain's curvature. -/
def chainH : List (Carrier R n n) → (Fin n → R) → Matrix (Fin n) (Fin n) R
  | [], _ => 0
  | c :: t, lam => (chainJ t)ᵀ * c.H lam * chainJ t + chainH t (c.Jᵀ *ᵥ lam)

theorem compChain_J (L : List (Carrier R n n)) : (compChain L).J = chainJ L := by
  induction L with
  | nil => rfl
  | cons c t ih => simp [compChain, chainJ, ih]

theorem compChain_H (L : List (Carrier R n n)) (lam : Fin n → R) :
    (compChain L).H lam = chainH L lam := by
  induction L generalizing lam with
  | nil => rfl
  | cons c t ih => simp [compChain, chainH, compChain_J, ih]

def chainJv : List (Carrier R n n) → (Fin n → R) → Fin n → R
  | [], v => v
  | c :: t, v => c.J *ᵥ chainJv t v

def chainVJ : List (Carrier R n n) → (Fin n → R) → Fin n → R
  | [], u => u
  | c :: t, u => chainVJ t (u ᵥ* c.J)

theorem chainJv_eq (L : List (Carrier R n n)) (v : Fin n → R) :
    chainJv L v = chainJ L *ᵥ v := by
  induction L with
  | nil => simp [chainJv, chainJ]
  | cons c t ih => simp [chainJv, chainJ, ih, mulVec_mulVec]

theorem chainVJ_eq (L : List (Carrier R n n)) (u : Fin n → R) :
    chainVJ L u = u ᵥ* chainJ L := by
  induction L generalizing u with
  | nil => simp [chainVJ, chainJ]
  | cons c t ih => simp [chainVJ, chainJ, ih, vecMul_vecMul]

/-- **The exact Hessian-vector program**: per layer one forward `mulVec`, one local
curvature hit, one backward `vecMul`; every intermediate is a vector. -/
def chainHv : List (Carrier R n n) → (Fin n → R) → (Fin n → R) → Fin n → R
  | [], _, _ => 0
  | c :: t, lam, v => chainVJ t (c.H lam *ᵥ chainJv t v) + chainHv t (lam ᵥ* c.J) v

theorem chainHv_eq (L : List (Carrier R n n)) (lam v : Fin n → R) :
    chainHv L lam v = chainH L lam *ᵥ v := by
  induction L generalizing lam with
  | nil => simp [chainHv, chainH]
  | cons c t ih =>
      simp only [chainHv, chainH, chainVJ_eq, chainJv_eq, ih, add_mulVec, mulVec_mulVec,
        ← mulVec_transpose, ← Matrix.mul_assoc]

theorem chain_entry_recovery (L : List (Carrier R n n)) (lam : Fin n → R) (i j : Fin n) :
    Pi.single i 1 ⬝ᵥ chainHv L lam (Pi.single j 1) = chainH L lam i j := by
  rw [chainHv_eq]
  simp

/-! ## §4 Analytic soundness: the second-order chain rule -/

variable {E F G : Type*}
  [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]
  [NormedAddCommGroup G] [NormedSpace ℝ G]

/-- The second derivative of a composition, assembled from the parts — the abstract
form of the carrier composition law. -/
def sndComp (g'x : F →L[ℝ] G) (f'x : E →L[ℝ] F)
    (g''x : F →L[ℝ] F →L[ℝ] G) (f''x : E →L[ℝ] E →L[ℝ] F) :
    E →L[ℝ] E →L[ℝ] G :=
  (compL ℝ E F G g'x).comp f''x + ((compL ℝ E F G).flip f'x).comp (g''x.comp f'x)

@[simp] theorem sndComp_apply (g'x : F →L[ℝ] G) (f'x : E →L[ℝ] F)
    (g''x : F →L[ℝ] F →L[ℝ] G) (f''x : E →L[ℝ] E →L[ℝ] F) (v w : E) :
    sndComp g'x f'x g''x f''x v w = g'x (f''x v w) + g''x (f'x v) (f'x w) := by
  simp [sndComp]

variable {f : E → F} {g : F → G} {f' : E → E →L[ℝ] F} {g' : F → F →L[ℝ] G}
  {f'' : E →L[ℝ] E →L[ℝ] F} {g'' : F →L[ℝ] F →L[ℝ] G} {x : E}

theorem chain_fst (hf' : ∀ y, HasFDerivAt f (f' y) y) (hg' : ∀ z, HasFDerivAt g (g' z) z)
    (y : E) : HasFDerivAt (g ∘ f) ((g' (f y)).comp (f' y)) y :=
  (hg' (f y)).comp y (hf' y)

/-- **The second-order chain rule** (coordinate-free): the derivative map of `g ∘ f` is
differentiable with derivative `sndComp` — the exact analytic content of the carrier
composition law. -/
theorem chain_snd (hf' : ∀ y, HasFDerivAt f (f' y) y)
    (hf'' : HasFDerivAt f' f'' x) (hg'' : HasFDerivAt g' g'' (f x)) :
    HasFDerivAt (fun y => (g' (f y)).comp (f' y))
      (sndComp (g' (f x)) (f' x) g'' f'') x := by
  have hc : HasFDerivAt (fun y => g' (f y)) (g''.comp (f' x)) x := hg''.comp x (hf' x)
  exact hc.clm_comp hf''

theorem snd_symm (hf' : ∀ y, HasFDerivAt f (f' y) y) (hf'' : HasFDerivAt f' f'' x)
    (v w : E) : f'' v w = f'' w v :=
  second_derivative_symmetric hf' hf'' v w

theorem affine_fst (A : E →L[ℝ] F) (b : F) (y : E) :
    HasFDerivAt (fun z => A z + b) A y :=
  A.hasFDerivAt.add_const b

theorem affine_snd (A : E →L[ℝ] F) (x : E) :
    HasFDerivAt (fun _ : E => A) (0 : E →L[ℝ] E →L[ℝ] F) x :=
  hasFDerivAt_const _ _

theorem quad_fst (B : E →L[ℝ] E →L[ℝ] F) (y : E) :
    HasFDerivAt (fun z => B z z) (B y + B.flip y) y := by
  simpa using B.hasFDerivAt.clm_apply (hasFDerivAt_id y)

theorem quad_snd (B : E →L[ℝ] E →L[ℝ] F) (x : E) :
    HasFDerivAt (fun y => B y + B.flip y) (B + B.flip) x := by
  have h : (fun y => B y + B.flip y) = fun y => (B + B.flip) y := by
    ext y; simp
  rw [h]
  exact (B + B.flip).hasFDerivAt

/-- Every symmetric continuous bilinear form is exactly the second derivative of its
half-quadratic, at every point. -/
theorem half_quad_exact (B : E →L[ℝ] E →L[ℝ] F) (hB : ∀ v w, B v w = B w v) :
    (∀ y, HasFDerivAt (fun z => (2⁻¹ : ℝ) • B z z) (B y) y) ∧
    (∀ x : E, HasFDerivAt (fun y : E => B y) B x) := by
  have hflip : B.flip = B := by
    ext v w
    exact hB w v
  constructor
  · intro y
    have h := (quad_fst B y).const_smul (2⁻¹ : ℝ)
    rw [hflip] at h
    have h2 : (2⁻¹ : ℝ) • (B y + B y) = B y := by
      rw [← two_smul ℝ (B y), smul_smul]
      norm_num
    rwa [h2] at h
  · intro x
    exact B.hasFDerivAt

/-- Exact scalar-loss decomposition, coordinate-free:
`(ℓ∘f)''(v,w) = ℓ''(f'v, f'w) + ℓ'(f''(v,w))`. -/
theorem loss_snd {ℓ : F → ℝ} {ℓ' : F → F →L[ℝ] ℝ} {ℓ'' : F →L[ℝ] F →L[ℝ] ℝ}
    (hf' : ∀ y, HasFDerivAt f (f' y) y) (hℓ' : ∀ z, HasFDerivAt ℓ (ℓ' z) z)
    (hf'' : HasFDerivAt f' f'' x) (hℓ'' : HasFDerivAt ℓ' ℓ'' (f x)) :
    (∀ y, HasFDerivAt (ℓ ∘ f) ((ℓ' (f y)).comp (f' y)) y) ∧
    HasFDerivAt (fun y => (ℓ' (f y)).comp (f' y)) (sndComp (ℓ' (f x)) (f' x) ℓ'' f'') x ∧
    ∀ v w, sndComp (ℓ' (f x)) (f' x) ℓ'' f'' v w
      = ℓ'' (f' x v) (f' x w) + ℓ' (f x) (f'' v w) := by
  refine ⟨chain_fst hf' hℓ', chain_snd hf' hf'' hℓ'', fun v w => ?_⟩
  rw [sndComp_apply]
  ring

/-! ## §5 The bridge: matrix carriers exactly shadow the analysis -/

/-- A matrix as a continuous linear map on coordinate space. -/
def toCLM (A : Matrix (Fin m) (Fin n) ℝ) :
    (Fin n → ℝ) →L[ℝ] (Fin m → ℝ) :=
  LinearMap.toContinuousLinearMap
    { toFun := fun v => A *ᵥ v
      map_add' := mulVec_add A
      map_smul' := fun c v => mulVec_smul A c v }

@[simp] theorem toCLM_apply (A : Matrix (Fin m) (Fin n) ℝ) (v : Fin n → ℝ) :
    toCLM A v = A *ᵥ v := by
  simp [toCLM]

theorem toCLM_mul (A : Matrix (Fin p) (Fin m) ℝ) (B : Matrix (Fin m) (Fin n) ℝ) :
    toCLM (A * B) = (toCLM A).comp (toCLM B) := by
  ext v
  simp [mulVec_mulVec]

theorem adj_pairing (A : Matrix (Fin p) (Fin m) ℝ) (lam : Fin p → ℝ) (u : Fin m → ℝ) :
    lam ⬝ᵥ (A *ᵥ u) = (Aᵀ *ᵥ lam) ⬝ᵥ u := by
  rw [dotProduct_mulVec, ← mulVec_transpose]

theorem conj_pairing (A : Matrix (Fin m) (Fin n) ℝ) (M : Matrix (Fin m) (Fin m) ℝ)
    (v w : Fin n → ℝ) :
    (A *ᵥ v) ⬝ᵥ (M *ᵥ (A *ᵥ w)) = v ⬝ᵥ ((Aᵀ * M * A) *ᵥ w) := by
  rw [← mulVec_mulVec, ← mulVec_mulVec, dotProduct_mulVec v, vecMul_transpose]

/-- `C` is **exact** for `f` at `x`: `J` is the true derivative, `H` reproduces the true
second derivative through every adjoint direction. -/
structure ExactAt (C : Carrier ℝ n m) (f : (Fin n → ℝ) → (Fin m → ℝ))
    (f' : (Fin n → ℝ) → (Fin n → ℝ) →L[ℝ] (Fin m → ℝ))
    (f'' : (Fin n → ℝ) →L[ℝ] (Fin n → ℝ) →L[ℝ] (Fin m → ℝ))
    (x : Fin n → ℝ) : Prop where
  fst : ∀ y, HasFDerivAt f (f' y) y
  snd : HasFDerivAt f' f'' x
  J_eq : f' x = toCLM C.J
  H_eq : ∀ lam v w, lam ⬝ᵥ f'' v w = v ⬝ᵥ (C.H lam *ᵥ w)

/-- **Soundness of `⋆`**: exact carriers compose to exact carriers — the algebra
computes true second derivatives of actual compositions. -/
theorem ExactAt.comp {Cg : Carrier ℝ m p} {Cf : Carrier ℝ n m}
    {g : (Fin m → ℝ) → (Fin p → ℝ)} {f : (Fin n → ℝ) → (Fin m → ℝ)}
    {g' : (Fin m → ℝ) → (Fin m → ℝ) →L[ℝ] (Fin p → ℝ)}
    {f' : (Fin n → ℝ) → (Fin n → ℝ) →L[ℝ] (Fin m → ℝ)}
    {g'' : (Fin m → ℝ) →L[ℝ] (Fin m → ℝ) →L[ℝ] (Fin p → ℝ)}
    {f'' : (Fin n → ℝ) →L[ℝ] (Fin n → ℝ) →L[ℝ] (Fin m → ℝ)}
    {x : Fin n → ℝ}
    (hg : ExactAt Cg g g' g'' (f x)) (hf : ExactAt Cf f f' f'' x) :
    ExactAt (Cg ⋆ Cf) (g ∘ f) (fun y => (g' (f y)).comp (f' y))
      (sndComp (g' (f x)) (f' x) g'' f'') x where
  fst := fun y => (hg.fst (f y)).comp y (hf.fst y)
  snd := chain_snd hf.fst hf.snd hg.snd
  J_eq := by
    show (g' (f x)).comp (f' x) = toCLM (Cg.J * Cf.J)
    rw [hg.J_eq, hf.J_eq, toCLM_mul]
  H_eq := by
    intro lam v w
    rw [sndComp_apply, dotProduct_add]
    have termA : lam ⬝ᵥ (g' (f x)) (f'' v w) = v ⬝ᵥ (Cf.H (Cg.Jᵀ *ᵥ lam) *ᵥ w) := by
      rw [hg.J_eq, toCLM_apply, adj_pairing]
      exact hf.H_eq _ v w
    have termB : lam ⬝ᵥ g'' ((f' x) v) ((f' x) w)
        = v ⬝ᵥ ((Cf.Jᵀ * Cg.H lam * Cf.J) *ᵥ w) := by
      rw [hg.H_eq lam, hf.J_eq, toCLM_apply, toCLM_apply, conj_pairing]
    rw [termA, termB]
    show _ = v ⬝ᵥ ((Cf.Jᵀ * Cg.H lam * Cf.J + Cf.H (Cg.Jᵀ *ᵥ lam)) *ᵥ w)
    rw [add_mulVec, dotProduct_add, add_comm]

theorem ExactAt.affine (A : Matrix (Fin m) (Fin n) ℝ) (b : Fin m → ℝ) (x : Fin n → ℝ) :
    ExactAt (affineCarrier A) (fun z => toCLM A z + b) (fun _ => toCLM A) 0 x where
  fst := fun y => affine_fst (toCLM A) b y
  snd := affine_snd (toCLM A) x
  J_eq := rfl
  H_eq := by
    intro lam v w
    simp [affineCarrier]

/-- Two independent routes agree: algebraic symmetry of `H` matches analytic symmetry
of `f''` for any exact carrier. -/
theorem ExactAt.H_consistent {C : Carrier ℝ n m} {f f' f''} {x : Fin n → ℝ}
    (h : ExactAt C f f' f'' x) (lam : Fin m → ℝ) (v w : Fin n → ℝ) :
    v ⬝ᵥ (C.H lam *ᵥ w) = w ⬝ᵥ (C.H lam *ᵥ v) := by
  rw [← h.H_eq, ← h.H_eq, snd_symm h.fst h.snd]

/-! ## §6 The witness: every symmetric matrix is an exact Hessian -/

/-- The bilinear form `⟨v, A w⟩` as a continuous bilinear map. -/
def matBilin (A : Matrix (Fin n) (Fin n) ℝ) :
    (Fin n → ℝ) →L[ℝ] (Fin n → ℝ) →L[ℝ] ℝ :=
  LinearMap.toContinuousLinearMap <|
    (LinearMap.toContinuousLinearMap.toLinearMap.comp
      (LinearMap.mk₂ ℝ (fun v w => v ⬝ᵥ (A *ᵥ w))
        (fun a b c => add_dotProduct a b _)
        (fun r a c => smul_dotProduct r a _)
        (fun a b c => by rw [mulVec_add, dotProduct_add])
        (fun r a b => by rw [mulVec_smul, dotProduct_smul])))

@[simp] theorem matBilin_apply (A : Matrix (Fin n) (Fin n) ℝ) (v w : Fin n → ℝ) :
    matBilin A v w = v ⬝ᵥ (A *ᵥ w) := by
  simp [matBilin]

theorem matBilin_symm {A : Matrix (Fin n) (Fin n) ℝ} (hA : A.IsSymm) (v w : Fin n → ℝ) :
    matBilin A v w = matBilin A w v := by
  rw [matBilin_apply, matBilin_apply, dotProduct_mulVec, ← mulVec_transpose, hA.eq,
    dotProduct_comm]

theorem matBilin_basis (A : Matrix (Fin n) (Fin n) ℝ) (i j : Fin n) :
    matBilin A (Pi.single i 1) (Pi.single j 1) = A i j := by
  rw [matBilin_apply]
  simp

/-- For symmetric `A`, `q(x) = ½⟨x, Ax⟩` has second derivative whose basis values are
exactly the entries of `A`: unstructured exact curvature is `dim Sym(n)`-hard. -/
theorem matrix_quad_witness {A : Matrix (Fin n) (Fin n) ℝ} (hA : A.IsSymm) :
    (∀ y, HasFDerivAt (fun z : Fin n → ℝ => (2⁻¹ : ℝ) • (z ⬝ᵥ (A *ᵥ z)))
      (matBilin A y) y) ∧
    (∀ x : Fin n → ℝ, HasFDerivAt (fun y : Fin n → ℝ => matBilin A y) (matBilin A) x) ∧
    (∀ i j, matBilin A (Pi.single i 1) (Pi.single j 1) = A i j) := by
  have hq : (fun z : Fin n → ℝ => (2⁻¹ : ℝ) • (z ⬝ᵥ (A *ᵥ z)))
      = fun z => (2⁻¹ : ℝ) • matBilin A z z := by
    ext z
    rw [matBilin_apply]
  have h := half_quad_exact (matBilin A) (matBilin_symm hA)
  exact ⟨by rw [hq]; exact h.1, h.2, matBilin_basis A⟩

/-! ## §7 Tomographic probing: what structure cannot give, measurement can -/

/-- A sign vector: every entry `±1`. -/
def IsSign (v : Fin n → ℝ) : Prop := ∀ i, v i = 1 ∨ v i = -1

lemma IsSign.mul_self {v : Fin n → ℝ} (hv : IsSign v) (i : Fin n) : v i * v i = 1 := by
  rcases hv i with h | h <;> rw [h] <;> norm_num

/-- Sign vector of a boolean pattern (a point of the finite probability space). -/
def sgn (s : Fin n → Bool) : Fin n → ℝ := fun j => if s j then 1 else -1

lemma sgn_isSign (s : Fin n → Bool) : IsSign (sgn s) := by
  intro i; unfold sgn; split <;> simp

@[simp] lemma sgn_mul_self (s : Fin n → Bool) (i : Fin n) : sgn s i * sgn s i = 1 :=
  (sgn_isSign s).mul_self i

lemma mulVec_apply' (M : Matrix (Fin n) (Fin n) ℝ) (v : Fin n → ℝ) (i : Fin n) :
    M.mulVec v i = ∑ j, M i j * v j := rfl

/-- For sign probes, dividing the response by the probe IS multiplying by it. -/
theorem probe_div_eq_mul {v : Fin n → ℝ} (hv : IsSign v) (p : Fin n → ℝ) (i : Fin n) :
    p i / v i = p i * v i := by
  rcases hv i with h | h <;> rw [h] <;> simp [div_neg]

/-- **Master identity**: `v_i (Mv)_i = M_ii + Σ_{j≠i} M_ij v_i v_j` — the diagonal plus
an explicit interference term. Everything below is about killing that term. -/
theorem master_identity (M : Matrix (Fin n) (Fin n) ℝ) {v : Fin n → ℝ}
    (hv : IsSign v) (i : Fin n) :
    v i * M.mulVec v i = M i i + ∑ j ∈ univ.erase i, M i j * (v i * v j) := by
  rw [mulVec_apply', Finset.mul_sum, ← Finset.sum_erase_add _ _ (mem_univ i)]
  have hdiag : v i * (M i i * v i) = M i i := by
    calc v i * (M i i * v i) = v i * v i * M i i := by ring
    _ = M i i := by rw [hv.mul_self i, one_mul]
  have hsum : ∑ j ∈ univ.erase i, v i * (M i j * v j)
      = ∑ j ∈ univ.erase i, M i j * (v i * v j) :=
    Finset.sum_congr rfl fun j _ => by ring
  rw [hsum, hdiag, add_comm]

/-- **One probe can never suffice** (n ≥ 2): for every probe there is a symmetric
`E ≠ 0` with `Ev = 0` and nonzero diagonal — invisible to the probe, visible in the
quantity sought. -/
theorem one_probe_insufficient (hn : 2 ≤ n) (v : Fin n → ℝ) :
    ∃ E : Matrix (Fin n) (Fin n) ℝ,
      E.IsSymm ∧ E ≠ 0 ∧ E.mulVec v = 0 ∧ ∃ i, E i i ≠ 0 := by
  have h0 : 0 < n := by omega
  have h1 : 1 < n := by omega
  obtain ⟨w, hdot, j, hj⟩ : ∃ w : Fin n → ℝ, (∑ x, w x * v x) = 0 ∧ ∃ j, w j ≠ 0 := by
    by_cases hv : v = 0
    · refine ⟨Pi.single ⟨0, h0⟩ 1, ?_, ⟨0, h0⟩, by simp⟩
      simp [hv]
    · obtain ⟨i₀, hi₀⟩ := Function.ne_iff.mp hv
      simp only [Pi.zero_apply] at hi₀
      obtain ⟨j₀, hj₀⟩ : ∃ j₀ : Fin n, j₀ ≠ i₀ :=
        Fintype.exists_ne_of_one_lt_card (by rwa [Fintype.card_fin]) i₀
      refine ⟨(Pi.single i₀ (v j₀) + Pi.single j₀ (-(v i₀)) : Fin n → ℝ), ?_, j₀, ?_⟩
      · show (Pi.single i₀ (v j₀) + Pi.single j₀ (-(v i₀)) : Fin n → ℝ) ⬝ᵥ v = 0
        rw [add_dotProduct, single_dotProduct, single_dotProduct]
        ring
      · have hz : (Pi.single i₀ (v j₀) : Fin n → ℝ) j₀ = 0 := Pi.single_eq_of_ne hj₀ _
        have happ : (Pi.single i₀ (v j₀) + Pi.single j₀ (-(v i₀)) : Fin n → ℝ) j₀
            = -(v i₀) := by
          simp [hz]
        rw [happ]
        exact neg_ne_zero.mpr hi₀
  refine ⟨vecMulVec w w, ?_, ?_, ?_, j, ?_⟩
  · show (vecMulVec w w)ᵀ = vecMulVec w w
    ext a b
    simp [vecMulVec_apply, transpose_apply, mul_comm]
  · intro hE
    have hzero : vecMulVec w w j j = 0 := by rw [hE]; rfl
    rw [vecMulVec_apply] at hzero
    exact (mul_ne_zero hj hj) hzero
  · funext i
    rw [mulVec_apply']
    have hterm : ∀ x, vecMulVec w w i x * v x = w i * (w x * v x) := by
      intro x; rw [vecMulVec_apply]; ring
    rw [Finset.sum_congr rfl fun x _ => hterm x, ← Finset.mul_sum, hdot, mul_zero]
    rfl
  · rw [vecMulVec_apply]
    exact mul_ne_zero hj hj

/-- **Cycle identity** over `m` probes: interference of pair `(i,j)` is weighted by the
code coherence `Σ_k v^k_i v^k_j` — a designed quantity, not noise. -/
theorem cycle_identity (M : Matrix (Fin n) (Fin n) ℝ)
    {V : Fin m → Fin n → ℝ} (hV : ∀ k, IsSign (V k)) (i : Fin n) :
    ∑ k, V k i * M.mulVec (V k) i
      = m * M i i + ∑ j ∈ univ.erase i, M i j * (∑ k, V k i * V k j) := by
  have expand : ∀ k, V k i * M.mulVec (V k) i
      = M i i + ∑ j ∈ univ.erase i, M i j * (V k i * V k j) :=
    fun k => master_identity M (hV k) i
  rw [Finset.sum_congr rfl fun k _ => expand k, Finset.sum_add_distrib]
  congr 1
  · rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  · rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun j _ => by rw [← Finset.mul_sum]

/-- **Exact diagonal recovery**: codes orthogonal across every interacting pair kill the
interference deterministically — no expectation, no error term. -/
theorem diag_recovery_exact (M : Matrix (Fin n) (Fin n) ℝ)
    {V : Fin m → Fin n → ℝ} (hV : ∀ k, IsSign (V k)) (i : Fin n)
    (hm : (m : ℝ) ≠ 0)
    (horth : ∀ j ∈ univ.erase i, M i j ≠ 0 → ∑ k, V k i * V k j = 0) :
    (∑ k, V k i * M.mulVec (V k) i) / m = M i i := by
  rw [cycle_identity M hV i]
  have hz : ∑ j ∈ univ.erase i, M i j * (∑ k, V k i * V k j) = 0 := by
    apply Finset.sum_eq_zero
    intro j hj
    by_cases h : M i j = 0
    · rw [h, zero_mul]
    · rw [horth j hj h, mul_zero]
  rw [hz, add_zero, mul_comm, mul_div_assoc, div_self hm, mul_one]

/-- **Full-matrix recovery**: with all-pairs orthogonal codes, every entry is
reconstructed exactly: `M = (1/m) Σ_k p^k (v^k)ᵀ`. -/
theorem full_recovery (M : Matrix (Fin n) (Fin n) ℝ)
    {V : Fin m → Fin n → ℝ} (hs : ∀ k, IsSign (V k)) (hm : (m : ℝ) ≠ 0)
    (horth : ∀ j l : Fin n, j ≠ l → ∑ k, V k j * V k l = 0)
    (i j : Fin n) :
    (∑ k, M.mulVec (V k) i * V k j) / m = M i j := by
  have hexp : ∀ k, M.mulVec (V k) i * V k j = ∑ l, M i l * (V k l * V k j) := by
    intro k
    rw [mulVec_apply', Finset.sum_mul]
    exact Finset.sum_congr rfl fun l _ => by ring
  rw [Finset.sum_congr rfl fun k _ => hexp k, Finset.sum_comm]
  have hpull : ∀ l, (∑ k, M i l * (V k l * V k j)) = M i l * (∑ k, V k l * V k j) := by
    intro l; rw [← Finset.mul_sum]
  rw [Finset.sum_congr rfl fun l _ => hpull l]
  have hsplit : (∑ l, M i l * (∑ k, V k l * V k j)) = M i j * m := by
    rw [Finset.sum_eq_single j]
    · rw [Finset.sum_congr rfl fun k _ => (hs k).mul_self j, Finset.sum_const,
        Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
    · intro l _ hlj
      rw [horth l j hlj, mul_zero]
    · intro hj; exact absurd (mem_univ j) hj
  rw [hsplit, mul_div_assoc, div_self hm, mul_one]

/-- **Full recovery costs `m ≥ n` probes** (Gram rank): the dense regime is provably
expensive; efficiency lives in structured/split recovery. -/
theorem full_recovery_cost (hm : 0 < m) {V : Fin m → Fin n → ℝ}
    (hs : ∀ k, IsSign (V k))
    (horth : ∀ j l : Fin n, j ≠ l → ∑ k, V k j * V k l = 0) : n ≤ m := by
  set A : Matrix (Fin m) (Fin n) ℝ := Matrix.of (fun k j => V k j) with hA
  have hm' : (m : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hm.ne'
  have hGram : Aᵀ * A = (m : ℝ) • (1 : Matrix (Fin n) (Fin n) ℝ) := by
    ext j l
    rw [Matrix.mul_apply]
    by_cases hjl : j = l
    · subst hjl
      have hone : ∀ k, Aᵀ j k * A k j = 1 := fun k => (hs k).mul_self j
      rw [Finset.sum_congr rfl fun k _ => hone k, Finset.sum_const, Finset.card_univ,
        Fintype.card_fin]
      simp
    · have hVV : ∀ k, Aᵀ j k * A k l = V k j * V k l := fun k => rfl
      rw [Finset.sum_congr rfl fun k _ => hVV k, horth j l hjl]
      simp [Matrix.one_apply_ne hjl]
  have hu : IsUnit ((m : ℝ) • (1 : Matrix (Fin n) (Fin n) ℝ)) := by
    refine isUnit_iff_exists.mpr ⟨(m : ℝ)⁻¹ • 1, ?_, ?_⟩
    · rw [smul_mul_smul_comm, mul_inv_cancel₀ hm', one_mul, one_smul]
    · rw [smul_mul_smul_comm, inv_mul_cancel₀ hm', one_mul, one_smul]
  have hrank : (Aᵀ * A).rank = n := by
    rw [hGram, Matrix.rank_of_isUnit _ hu, Fintype.card_fin]
  calc n = (Aᵀ * A).rank := hrank.symm
  _ ≤ Aᵀ.rank := Matrix.rank_mul_le_left _ _
  _ ≤ Fintype.card (Fin m) := Matrix.rank_le_card_width _
  _ = m := Fintype.card_fin m

/-- The two parity probes: all-ones and coordinate-parity alternation. -/
def parityProbes (n : ℕ) : Fin 2 → Fin n → ℝ :=
  fun k j => if k = 0 then 1 else (-1 : ℝ) ^ (j : ℕ)

lemma parityProbes_isSign : ∀ k, IsSign (parityProbes n k) := by
  intro k i
  unfold parityProbes
  split
  · left; rfl
  · rcases Nat.even_or_odd (i : ℕ) with h | h
    · left; exact h.neg_one_pow
    · right; exact h.neg_one_pow

/-- **Two probes recover any banded (tridiagonal) diagonal exactly**: adjacent
coordinates get opposite-parity codes. With `one_probe_insufficient` (whose witness is a
2×2 block), the probe complexity of banded recovery is exactly 2. -/
theorem banded_two_probe (M : Matrix (Fin n) (Fin n) ℝ)
    (htri : ∀ i j : Fin n, ((i : ℕ) + 2 ≤ (j : ℕ) ∨ (j : ℕ) + 2 ≤ (i : ℕ)) → M i j = 0)
    (i : Fin n) :
    (∑ k, parityProbes n k i * M.mulVec (parityProbes n k) i) / ((2 : ℕ) : ℝ) = M i i := by
  apply diag_recovery_exact M parityProbes_isSign i (by norm_num)
  intro j hj hMij
  have hji : j ≠ i := Finset.ne_of_mem_erase hj
  have hne : (j : ℕ) ≠ (i : ℕ) := fun h => hji (Fin.ext h)
  have hadj : (j : ℕ) = (i : ℕ) + 1 ∨ (i : ℕ) = (j : ℕ) + 1 := by
    by_contra hc
    exact hMij (htri i j (by omega))
  have e0 : ∀ x : Fin n, parityProbes n 0 x = 1 := fun x => if_pos rfl
  have e1 : ∀ x : Fin n, parityProbes n 1 x = (-1 : ℝ) ^ (x : ℕ) :=
    fun x => if_neg (by decide)
  rw [Fin.sum_univ_two, e0 i, e0 j, e1 i, e1 j, one_mul, ← pow_add]
  have hodd : Odd ((i : ℕ) + (j : ℕ)) := by
    rw [Nat.odd_iff]; omega
  rw [hodd.neg_one_pow]
  ring

/-! ### Character sums over the boolean cube (the probability backbone) -/

lemma sum_sgn_mul_of_ne {i j : Fin n} (hij : i ≠ j) :
    ∑ s : Fin n → Bool, sgn s i * sgn s j = 0 := by
  have hinv : Function.Involutive (fun s : Fin n → Bool => Function.update s j (!(s j))) := by
    intro s
    funext x
    by_cases hx : x = j
    · subst hx; simp
    · simp [Function.update_of_ne hx]
  have key : ∀ s : Fin n → Bool,
      sgn (Function.update s j (!(s j))) i * sgn (Function.update s j (!(s j))) j
        = -(sgn s i * sgn s j) := by
    intro s
    have hi : sgn (Function.update s j (!(s j))) i = sgn s i := by
      unfold sgn; rw [Function.update_of_ne hij]
    have hjv : sgn (Function.update s j (!(s j))) j = -(sgn s j) := by
      unfold sgn; rw [Function.update_self]
      cases hsj : s j <;> simp
    rw [hi, hjv]; ring
  have hswap := Equiv.sum_comp (hinv.toPerm _)
    (fun s : Fin n → Bool => sgn s i * sgn s j)
  simp only [Function.Involutive.coe_toPerm] at hswap
  have hneg : ∑ s : Fin n → Bool, sgn s i * sgn s j
      = -∑ s : Fin n → Bool, sgn s i * sgn s j := by
    conv_lhs => rw [← hswap]
    rw [Finset.sum_congr rfl fun s _ => key s, Finset.sum_neg_distrib]
  linarith

lemma sum_sgn_sq (i : Fin n) :
    ∑ s : Fin n → Bool, sgn s i * sgn s i = (2 : ℝ) ^ n := by
  rw [Finset.sum_congr rfl fun s _ => sgn_mul_self s i, Finset.sum_const, Finset.card_univ]
  simp

lemma sum_sgn_mul (i j : Fin n) :
    ∑ s : Fin n → Bool, sgn s i * sgn s j = if i = j then (2 : ℝ) ^ n else 0 := by
  by_cases h : i = j
  · subst h; rw [if_pos rfl]; exact sum_sgn_sq i
  · rw [if_neg h]; exact sum_sgn_mul_of_ne h

/-- Cube Parseval (degree 1): distinct characters do not interact. -/
lemma sum_sq_sgn_weighted (i : Fin n) (a : Fin n → ℝ) :
    ∑ s : Fin n → Bool, (∑ j ∈ univ.erase i, a j * sgn s j) ^ 2
      = (2 : ℝ) ^ n * ∑ j ∈ univ.erase i, a j ^ 2 := by
  have hexpand : ∀ s : Fin n → Bool,
      (∑ j ∈ univ.erase i, a j * sgn s j) ^ 2
        = ∑ j ∈ univ.erase i, ∑ l ∈ univ.erase i, (a j * a l) * (sgn s j * sgn s l) := by
    intro s
    rw [sq, Finset.sum_mul_sum]
    exact Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun l _ => by ring
  rw [Finset.sum_congr rfl fun s _ => hexpand s, Finset.sum_comm]
  have hinner : ∀ j ∈ univ.erase i,
      (∑ s : Fin n → Bool, ∑ l ∈ univ.erase i, (a j * a l) * (sgn s j * sgn s l))
        = (2 : ℝ) ^ n * a j ^ 2 := by
    intro j hjmem
    rw [Finset.sum_comm]
    have hl : ∀ l ∈ univ.erase i,
        (∑ s : Fin n → Bool, (a j * a l) * (sgn s j * sgn s l))
          = (a j * a l) * (if j = l then (2 : ℝ) ^ n else 0) := by
      intro l _
      rw [← Finset.mul_sum, sum_sgn_mul]
    rw [Finset.sum_congr rfl hl, Finset.sum_eq_single j]
    · rw [if_pos rfl]; ring
    · intro l _ hlj
      rw [if_neg fun h => hlj h.symm, mul_zero]
    · intro hj'
      exact absurd hjmem hj'
  rw [Finset.sum_congr rfl hinner, ← Finset.mul_sum]

/-- **Hutchinson**: over uniform sign probes, `E[v ⊙ Mv] = diag M`. -/
theorem hutchinson_unbiased (M : Matrix (Fin n) (Fin n) ℝ) (i : Fin n) :
    ∑ s : Fin n → Bool, sgn s i * M.mulVec (sgn s) i = (2 : ℝ) ^ n * M i i := by
  have expand : ∀ s : Fin n → Bool,
      sgn s i * M.mulVec (sgn s) i = ∑ j, M i j * (sgn s i * sgn s j) := by
    intro s
    rw [mulVec_apply', Finset.mul_sum]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [Finset.sum_congr rfl fun s _ => expand s, Finset.sum_comm]
  have hj : ∀ j, (∑ s : Fin n → Bool, M i j * (sgn s i * sgn s j))
      = M i j * (if i = j then (2 : ℝ) ^ n else 0) := by
    intro j; rw [← Finset.mul_sum, sum_sgn_mul]
  rw [Finset.sum_congr rfl fun j _ => hj j]
  simp only [mul_ite, mul_zero]
  rw [Finset.sum_ite_eq]
  simp [mul_comm]

/-! ### Coded cycles with a random global sign flip -/

section Coded

variable (C : Fin m → Fin n → ℝ)

/-- Probe `k` of the coded cycle after the global flip `s`: `v^k = sgn(s) ⊙ C_k`. -/
def flip (s : Fin n → Bool) (k : Fin m) : Fin n → ℝ := fun x => sgn s x * C k x

lemma flip_isSign (hC : ∀ k, IsSign (C k)) (s : Fin n → Bool) (k : Fin m) :
    IsSign (flip C s k) := by
  intro i
  unfold flip
  rcases sgn_isSign s i with h | h <;> rcases hC k i with g | g <;>
    rw [h, g] <;> norm_num

/-- Deviation of the coded cycle from `m·M_ii` in closed form: pure interference,
weighted by coherences and the random signs. -/
lemma coded_error (hC : ∀ k, IsSign (C k)) (M : Matrix (Fin n) (Fin n) ℝ)
    (i : Fin n) (s : Fin n → Bool) :
    (∑ k, flip C s k i * M.mulVec (flip C s k) i) - m * M i i
      = ∑ j ∈ univ.erase i, (M i j * (∑ k, C k i * C k j)) * (sgn s i * sgn s j) := by
  rw [cycle_identity M (flip_isSign C hC s) i]
  have hterm : ∀ j ∈ univ.erase i,
      M i j * (∑ k, flip C s k i * flip C s k j)
        = (M i j * (∑ k, C k i * C k j)) * (sgn s i * sgn s j) := by
    intro j _
    have hk : ∀ k, flip C s k i * flip C s k j
        = (C k i * C k j) * (sgn s i * sgn s j) := by
      intro k; unfold flip; ring
    rw [Finset.sum_congr rfl fun k _ => hk k, ← Finset.sum_mul]
    ring
  rw [Finset.sum_congr rfl hterm]
  ring

/-- **Unbiasedness for arbitrary codes**: the global flip alone de-biases the cycle. -/
theorem coded_unbiased (hC : ∀ k, IsSign (C k)) (M : Matrix (Fin n) (Fin n) ℝ)
    (i : Fin n) :
    ∑ s : Fin n → Bool, (∑ k, flip C s k i * M.mulVec (flip C s k) i)
      = (2 : ℝ) ^ n * (m * M i i) := by
  have hs : ∀ s : Fin n → Bool,
      (∑ k, flip C s k i * M.mulVec (flip C s k) i)
        = m * M i i
          + ∑ j ∈ univ.erase i, (M i j * (∑ k, C k i * C k j)) * (sgn s i * sgn s j) := by
    intro s
    have h0 := coded_error C hC M i s
    linarith
  rw [Finset.sum_congr rfl fun s _ => hs s, Finset.sum_add_distrib]
  have hconst : (∑ _s : Fin n → Bool, (m : ℝ) * M i i) = (2 : ℝ) ^ n * (m * M i i) := by
    rw [Finset.sum_const, Finset.card_univ]
    simp
  have hzero : (∑ s : Fin n → Bool,
      ∑ j ∈ univ.erase i, (M i j * (∑ k, C k i * C k j)) * (sgn s i * sgn s j)) = 0 := by
    rw [Finset.sum_comm]
    apply Finset.sum_eq_zero
    intro j hj
    have hji : j ≠ i := Finset.ne_of_mem_erase hj
    rw [← Finset.mul_sum, sum_sgn_mul, if_neg fun h => hji h.symm, mul_zero]
  rw [hconst, hzero, add_zero]

/-- **Variance factorization (central)**: the second moment of the coded cycle is
EXACTLY `2^n · Σ_{j≠i} M_ij² (Σ_k C_ki C_kj)²` — curvature strength × squared designed
coherence, per pair. -/
theorem coded_variance (hC : ∀ k, IsSign (C k)) (M : Matrix (Fin n) (Fin n) ℝ)
    (i : Fin n) :
    ∑ s : Fin n → Bool,
        ((∑ k, flip C s k i * M.mulVec (flip C s k) i) - m * M i i) ^ 2
      = (2 : ℝ) ^ n * ∑ j ∈ univ.erase i, (M i j) ^ 2 * (∑ k, C k i * C k j) ^ 2 := by
  have herr : ∀ s : Fin n → Bool,
      ((∑ k, flip C s k i * M.mulVec (flip C s k) i) - m * M i i) ^ 2
        = (∑ j ∈ univ.erase i, (M i j * (∑ k, C k i * C k j)) * sgn s j) ^ 2 := by
    intro s
    rw [coded_error C hC M i s]
    have hfac : ∑ j ∈ univ.erase i, (M i j * (∑ k, C k i * C k j)) * (sgn s i * sgn s j)
        = sgn s i * ∑ j ∈ univ.erase i, (M i j * (∑ k, C k i * C k j)) * sgn s j := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [hfac, mul_pow]
    have h1 : sgn s i ^ 2 = 1 := by rw [sq, sgn_mul_self]
    rw [h1, one_mul]
  rw [Finset.sum_congr rfl fun s _ => herr s]
  refine (sum_sq_sgn_weighted i fun j => M i j * (∑ k, C k i * C k j)).trans ?_
  congr 1
  refine Finset.sum_congr rfl fun j _ => ?_
  show (M i j * (∑ k, C k i * C k j)) ^ 2 = (M i j) ^ 2 * (∑ k, C k i * C k j) ^ 2
  ring

/-- Normalized unbiasedness of the cycle-mean estimator. -/
theorem coded_estimator_unbiased (hm : (m : ℝ) ≠ 0) (hC : ∀ k, IsSign (C k))
    (M : Matrix (Fin n) (Fin n) ℝ) (i : Fin n) :
    ∑ s : Fin n → Bool, (∑ k, flip C s k i * M.mulVec (flip C s k) i) / m
      = (2 : ℝ) ^ n * M i i := by
  rw [← Finset.sum_div, coded_unbiased C hC M i, mul_comm (m : ℝ) (M i i),
    ← mul_assoc, mul_div_assoc, div_self hm, mul_one]

/-- Normalized variance with the empirical coherence `C̄ = (Σ_k C_ki C_kj)/m`. -/
theorem coded_estimator_variance (hm : (m : ℝ) ≠ 0) (hC : ∀ k, IsSign (C k))
    (M : Matrix (Fin n) (Fin n) ℝ) (i : Fin n) :
    ∑ s : Fin n → Bool,
        ((∑ k, flip C s k i * M.mulVec (flip C s k) i) / m - M i i) ^ 2
      = (2 : ℝ) ^ n * ∑ j ∈ univ.erase i, (M i j) ^ 2 * ((∑ k, C k i * C k j) / m) ^ 2 := by
  have hstep : ∀ s : Fin n → Bool,
      ((∑ k, flip C s k i * M.mulVec (flip C s k) i) / m - M i i) ^ 2
        = ((∑ k, flip C s k i * M.mulVec (flip C s k) i) - m * M i i) ^ 2
            / m ^ 2 := by
    intro s
    have hd : (∑ k, flip C s k i * M.mulVec (flip C s k) i) / m - M i i
        = ((∑ k, flip C s k i * M.mulVec (flip C s k) i) - m * M i i) / m := by
      field_simp
    rw [hd, div_pow]
  have hRHS : ∑ j ∈ univ.erase i, (M i j) ^ 2 * ((∑ k, C k i * C k j) / m) ^ 2
      = (∑ j ∈ univ.erase i, (M i j) ^ 2 * (∑ k, C k i * C k j) ^ 2) / m ^ 2 := by
    rw [Finset.sum_div]
    exact Finset.sum_congr rfl fun j _ => by rw [div_pow, mul_div_assoc]
  rw [Finset.sum_congr rfl fun s _ => hstep s, ← Finset.sum_div,
    coded_variance C hC M i, hRHS, mul_div_assoc]

end Coded

/-- Hutchinson variance — the `m = 1`, all-ones-code degenerate case: every pair
collides; this is the baseline coded cycles improve on. -/
theorem hutchinson_variance (M : Matrix (Fin n) (Fin n) ℝ) (i : Fin n) :
    ∑ s : Fin n → Bool, (sgn s i * M.mulVec (sgn s) i - M i i) ^ 2
      = (2 : ℝ) ^ n * ∑ j ∈ univ.erase i, (M i j) ^ 2 := by
  have herr : ∀ s : Fin n → Bool,
      (sgn s i * M.mulVec (sgn s) i - M i i) ^ 2
        = (∑ j ∈ univ.erase i, M i j * sgn s j) ^ 2 := by
    intro s
    rw [master_identity M (sgn_isSign s) i]
    have hfac : M i i + (∑ j ∈ univ.erase i, M i j * (sgn s i * sgn s j)) - M i i
        = sgn s i * ∑ j ∈ univ.erase i, M i j * sgn s j := by
      have hswap : ∑ j ∈ univ.erase i, M i j * (sgn s i * sgn s j)
          = ∑ j ∈ univ.erase i, sgn s i * (M i j * sgn s j) :=
        Finset.sum_congr rfl fun j _ => by ring
      rw [hswap, ← Finset.mul_sum]
      ring
    rw [hfac, mul_pow]
    have h1 : sgn s i ^ 2 = 1 := by rw [sq, sgn_mul_self]
    rw [h1, one_mul]
  rw [Finset.sum_congr rfl fun s _ => herr s]
  exact sum_sq_sgn_weighted i fun j => M i j

/-- **Zero variance under orthogonal codes**: exactness restated probabilistically. -/
theorem orthogonal_zero_variance {C : Fin m → Fin n → ℝ}
    (hC : ∀ k, IsSign (C k)) (M : Matrix (Fin n) (Fin n) ℝ) (i : Fin n)
    (horth : ∀ j ∈ univ.erase i, M i j ≠ 0 → ∑ k, C k i * C k j = 0) :
    ∑ s : Fin n → Bool,
        ((∑ k, flip C s k i * M.mulVec (flip C s k) i) - m * M i i) ^ 2 = 0 := by
  rw [coded_variance C hC M i]
  have hz : ∑ j ∈ univ.erase i, (M i j) ^ 2 * (∑ k, C k i * C k j) ^ 2 = 0 := by
    apply Finset.sum_eq_zero
    intro j hj
    by_cases h : M i j = 0
    · rw [h]; ring
    · rw [horth j hj h]; ring
  rw [hz, mul_zero]

/-- **Coded never exceeds Hutchinson**: every coherence satisfies `|C̄| ≤ 1`. -/
theorem coded_le_hutchinson {C : Fin m → Fin n → ℝ}
    (hC : ∀ k, IsSign (C k)) (hm : 0 < m) (M : Matrix (Fin n) (Fin n) ℝ) (i : Fin n) :
    ∑ j ∈ univ.erase i, (M i j) ^ 2 * ((∑ k, C k i * C k j) / m) ^ 2
      ≤ ∑ j ∈ univ.erase i, (M i j) ^ 2 := by
  apply Finset.sum_le_sum
  intro j _
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have habs : |∑ k, C k i * C k j| ≤ m := by
    calc |∑ k, C k i * C k j| ≤ ∑ k, |C k i * C k j| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ _k : Fin m, (1 : ℝ) := Finset.sum_congr rfl fun k _ => by
        rw [abs_mul]
        rcases hC k i with h | h <;> rcases hC k j with g | g <;> rw [h, g] <;> norm_num
    _ = m := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
  have hdiv : |(∑ k, C k i * C k j) / m| ≤ 1 := by
    rw [abs_div, abs_of_pos hmR, div_le_one hmR]
    exact habs
  have hsq : ((∑ k, C k i * C k j) / m) ^ 2 ≤ 1 := by
    nlinarith [sq_abs ((∑ k, C k i * C k j) / m), abs_nonneg ((∑ k, C k i * C k j) / m)]
  calc (M i j) ^ 2 * ((∑ k, C k i * C k j) / m) ^ 2
      ≤ (M i j) ^ 2 * 1 := mul_le_mul_of_nonneg_left hsq (sq_nonneg _)
  _ = (M i j) ^ 2 := mul_one _

/-! ## §8 The fusion: probe the residual, not the Hessian -/

/-- A **positive core**: symmetric with nonnegative quadratic form (real PSD). -/
def PosCore (M : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  M.IsSymm ∧ ∀ x : Fin n → ℝ, 0 ≤ x ⬝ᵥ (M *ᵥ x)

/-- **The structural core is a positive core** whenever the loss Hessian is
(any convex loss): its diagonal is a sound nonnegative preconditioner base. -/
theorem core_posCore (Cf : Carrier ℝ n m) {Lam : Matrix (Fin m) (Fin m) ℝ}
    (hsym : Lam.IsSymm) (hpsd : ∀ z : Fin m → ℝ, 0 ≤ z ⬝ᵥ (Lam *ᵥ z)) :
    PosCore (structCore Cf Lam) := by
  constructor
  · unfold Matrix.IsSymm structCore
    rw [Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose,
      hsym.eq, Matrix.mul_assoc]
  · intro x
    have h := conj_pairing Cf.J Lam x x
    unfold structCore
    rw [← h]
    exact hpsd (Cf.J *ᵥ x)

/-- Positive cores have nonnegative diagonals — the base never flips sign. -/
theorem posCore_diag_nonneg {M : Matrix (Fin n) (Fin n) ℝ} (h : PosCore M) (i : Fin n) :
    0 ≤ M i i := by
  have hq := h.2 (Pi.single i 1)
  rwa [entry_recovery M i i] at hq

/-- **Split estimator, exactness form**: exact core diagonal plus a residual probe cycle
whose codes are orthogonal on the *residual's* interference graph recovers the exact
Hessian diagonal. The codes only need to resolve `R` — typically far sparser than `H`. -/
theorem split_probe_exact (Cl : Carrier ℝ m 1) (Cf : Carrier ℝ n m)
    {V : Fin q → Fin n → ℝ} (hV : ∀ k, IsSign (V k)) (i : Fin n)
    (hq : (q : ℝ) ≠ 0)
    (horth : ∀ j ∈ univ.erase i, residual Cf (Cl.Jᵀ *ᵥ 1) i j ≠ 0 →
      ∑ k, V k i * V k j = 0) :
    structCore Cf (Cl.H 1) i i
      + (∑ k, V k i * (residual Cf (Cl.Jᵀ *ᵥ 1)).mulVec (V k) i) / q
      = hess (Cl ⋆ Cf) i i := by
  rw [diag_recovery_exact _ hV i hq horth, split_exact]
  simp [Matrix.add_apply]

/-- **Split estimator, unbiasedness**: with a random global flip, exact core + probed
residual is unbiased for the exact Hessian diagonal under ANY codes. -/
theorem split_probe_unbiased (Cl : Carrier ℝ m 1) (Cf : Carrier ℝ n m)
    {C : Fin q → Fin n → ℝ} (hC : ∀ k, IsSign (C k)) (i : Fin n) (hq : (q : ℝ) ≠ 0) :
    ∑ s : Fin n → Bool,
        (structCore Cf (Cl.H 1) i i
          + (∑ k, flip C s k i * (residual Cf (Cl.Jᵀ *ᵥ 1)).mulVec (flip C s k) i) / q)
      = (2 : ℝ) ^ n * hess (Cl ⋆ Cf) i i := by
  rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ,
    coded_estimator_unbiased C hq hC _ i, split_exact]
  simp [Matrix.add_apply, mul_add]

/-- **The split estimator's variance is the residual's alone** — the structural core
contributes exactly zero variance. -/
theorem split_variance_residual_only (Cl : Carrier ℝ m 1) (Cf : Carrier ℝ n m)
    {C : Fin q → Fin n → ℝ} (hC : ∀ k, IsSign (C k)) (i : Fin n) (hq : (q : ℝ) ≠ 0) :
    ∑ s : Fin n → Bool,
        ((structCore Cf (Cl.H 1) i i
            + (∑ k, flip C s k i * (residual Cf (Cl.Jᵀ *ᵥ 1)).mulVec (flip C s k) i) / q)
          - hess (Cl ⋆ Cf) i i) ^ 2
      = (2 : ℝ) ^ n * ∑ j ∈ univ.erase i,
          (residual Cf (Cl.Jᵀ *ᵥ 1) i j) ^ 2 * ((∑ k, C k i * C k j) / q) ^ 2 := by
  have hcancel : ∀ s : Fin n → Bool,
      (structCore Cf (Cl.H 1) i i
          + (∑ k, flip C s k i * (residual Cf (Cl.Jᵀ *ᵥ 1)).mulVec (flip C s k) i) / q)
        - hess (Cl ⋆ Cf) i i
      = (∑ k, flip C s k i * (residual Cf (Cl.Jᵀ *ᵥ 1)).mulVec (flip C s k) i) / q
          - residual Cf (Cl.Jᵀ *ᵥ 1) i i := by
    intro s
    rw [split_exact]
    simp [Matrix.add_apply]
  rw [Finset.sum_congr rfl fun s _ => by rw [hcancel s]]
  exact coded_estimator_variance C hq hC _ i

/-- **Variance scales with the squared loss gradient.** Scaling the adjoint by `c`
scales every probe-variance term by `c²`: as training drives the loss gradient down,
the split estimator's variance vanishes quadratically. Probing the full Hessian has no
such mechanism — its structural variance is gradient-independent. -/
theorem split_variance_scaling (Cf : Carrier ℝ n m) (g : Fin m → ℝ) (c : ℝ)
    {C : Fin q → Fin n → ℝ} (hC : ∀ k, IsSign (C k)) (i : Fin n) (hq : (q : ℝ) ≠ 0) :
    ∑ s : Fin n → Bool,
        ((∑ k, flip C s k i * (residual Cf (c • g)).mulVec (flip C s k) i) / q
          - residual Cf (c • g) i i) ^ 2
      = c ^ 2 * ∑ s : Fin n → Bool,
          ((∑ k, flip C s k i * (residual Cf g).mulVec (flip C s k) i) / q
            - residual Cf g i i) ^ 2 := by
  rw [coded_estimator_variance C hq hC _ i, coded_estimator_variance C hq hC _ i,
    residual_smul]
  have hL : ∀ j, ((c • residual Cf g) i j) ^ 2 = c ^ 2 * (residual Cf g i j) ^ 2 := by
    intro j
    have hentry : (c • residual Cf g) i j = c * residual Cf g i j := rfl
    rw [hentry]
    ring
  calc (2 : ℝ) ^ n * ∑ j ∈ univ.erase i,
        ((c • residual Cf g) i j) ^ 2 * ((∑ k, C k i * C k j) / q) ^ 2
      = (2 : ℝ) ^ n * ∑ j ∈ univ.erase i,
          c ^ 2 * ((residual Cf g i j) ^ 2 * ((∑ k, C k i * C k j) / q) ^ 2) := by
        congr 1
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [hL j]
        ring
    _ = c ^ 2 * ((2 : ℝ) ^ n * ∑ j ∈ univ.erase i,
          (residual Cf g i j) ^ 2 * ((∑ k, C k i * C k j) / q) ^ 2) := by
        rw [← Finset.mul_sum]
        ring

/-- **Deterministic exactness at critical residual**: when the loss gradient vanishes,
the split estimator equals the exact Hessian diagonal for EVERY flip and ANY codes —
even a single probe. (Probing the full Hessian instead retains its full structural
variance in exactly this regime: see `coded_estimator_variance` applied to `H = S`.) -/
theorem split_exact_at_critical (Cl : Carrier ℝ m 1) (Cf : Carrier ℝ n m)
    (hcrit : Cl.Jᵀ *ᵥ 1 = 0)
    {C : Fin q → Fin n → ℝ} (i : Fin n) (s : Fin n → Bool) :
    structCore Cf (Cl.H 1) i i
      + (∑ k, flip C s k i * (residual Cf (Cl.Jᵀ *ᵥ 1)).mulVec (flip C s k) i) / q
      = hess (Cl ⋆ Cf) i i := by
  rw [hcrit, residual_zero]
  simp only [zero_mulVec, Pi.zero_apply, mul_zero, Finset.sum_const_zero, zero_div,
    add_zero]
  rw [split_exact, hcrit, residual_zero]
  simp [Matrix.add_apply]

/-- **Directional trust decomposition**: the measured ray curvature `vᵀHv` splits as
`vᵀSv + vᵀRv` with `vᵀSv ≥ 0` guaranteed — negative measured curvature is always
attributable to the residual. -/
theorem directional_split (Cl : Carrier ℝ m 1) (Cf : Carrier ℝ n m)
    (hsym : (Cl.H 1).IsSymm) (hpsd : ∀ z : Fin m → ℝ, 0 ≤ z ⬝ᵥ (Cl.H 1 *ᵥ z))
    (v : Fin n → ℝ) :
    v ⬝ᵥ (hess (Cl ⋆ Cf) *ᵥ v)
      = v ⬝ᵥ (structCore Cf (Cl.H 1) *ᵥ v) + v ⬝ᵥ (residual Cf (Cl.Jᵀ *ᵥ 1) *ᵥ v)
    ∧ 0 ≤ v ⬝ᵥ (structCore Cf (Cl.H 1) *ᵥ v) := by
  constructor
  · rw [split_exact, add_mulVec, dotProduct_add]
  · exact (core_posCore Cf hsym hpsd).2 v

/-! ## §9 Optimizer payoffs -/

/-- Separable quadratic `f(z) = Σ_i (h_i/2)z_i² + b_i z_i`. -/
def quadObj (h b : Fin n → ℝ) (z : Fin n → ℝ) : ℝ :=
  ∑ i, (h i / 2 * z i ^ 2 + b i * z i)

/-- The exact-diagonal Newton step `x ← x − ∇f(x)/h`. -/
def newtonStep (h b x : Fin n → ℝ) : Fin n → ℝ :=
  fun i => x i - (h i * x i + b i) / h i

lemma newtonStep_eq (h b x : Fin n → ℝ) (hp : ∀ i, 0 < h i) (i : Fin n) :
    newtonStep h b x i = -(b i) / h i := by
  have hi : h i ≠ 0 := ne_of_gt (hp i)
  unfold newtonStep
  field_simp
  ring

/-- **One-step global minimization**: the exact diagonal of a diagonal PD quadratic
preconditions to the global minimum from any start — what exact recovery buys. -/
theorem diag_newton_one_step (h b : Fin n → ℝ) (hp : ∀ i, 0 < h i) (x z : Fin n → ℝ) :
    quadObj h b (newtonStep h b x) ≤ quadObj h b z := by
  unfold quadObj
  apply Finset.sum_le_sum
  intro i _
  have hi : (0 : ℝ) < h i := hp i
  have hi' : h i ≠ 0 := ne_of_gt hi
  rw [newtonStep_eq h b x hp i]
  have key : h i / 2 * (-(b i) / h i) ^ 2 + b i * (-(b i) / h i)
      = -(b i ^ 2 / (2 * h i)) := by
    field_simp
    ring
  rw [key]
  have hdiff : (h i / 2 * z i ^ 2 + b i * z i) - (-(b i ^ 2 / (2 * h i)))
      = (h i * z i + b i) ^ 2 / (2 * h i) := by
    field_simp
    ring
  have hnn : (0 : ℝ) ≤ (h i * z i + b i) ^ 2 / (2 * h i) := by positivity
  linarith

/-- **Directional Newton optimality**: along a ray the loss is
`φ(η) = c − aη + (κ/2)η²` with `a = g·v`, `κ = vᵀHv` — both exact from one product.
For `κ > 0` the measured step `η* = a/κ` is the ray's global minimizer. -/
theorem directional_newton_optimal (a c : ℝ) {κ : ℝ} (hκ : 0 < κ) (η : ℝ) :
    c - a * (a / κ) + κ / 2 * (a / κ) ^ 2 ≤ c - a * η + κ / 2 * η ^ 2 := by
  have hκ' : κ ≠ 0 := ne_of_gt hκ
  have hdiff : (c - a * η + κ / 2 * η ^ 2)
      - (c - a * (a / κ) + κ / 2 * (a / κ) ^ 2)
      = κ / 2 * (η - a / κ) ^ 2 := by
    field_simp
    ring
  nlinarith [sq_nonneg (η - a / κ), hκ]

end Rocora
