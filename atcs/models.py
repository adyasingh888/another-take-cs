"""
The programme's actual interfaces, written as structural models.

Each one is the real wiring of a handoff in the bench-to-bedside chain, so
`observability` can decide it before anyone fits anything. The docstrings carry
the biology; the code carries only which quantity each measurement constrains.
"""
from __future__ import annotations

from .observability import Observation, StructuralModel

# ---------------------------------------------------------------------------
# The organ -> cytosol interface. This is where the chain breaks, and the reason
# is structural rather than statistical.
# ---------------------------------------------------------------------------
ORGAN_TO_CYTOSOL = StructuralModel(
    name="organ-associated material -> parenchymal-cell cytosolic exposure",
    unknowns=(
        "f_parenchymal",      # fraction of organ-associated material in target cells
        "f_macrophage",       # Kupffer / resident macrophage compartment
        "f_endothelial",      # sinusoidal endothelium
        "f_extracellular",    # interstitium and residual blood
        "phi_intact",         # fraction still carrier-associated rather than released
        "k_escape",           # endosomal escape rate into cytosol
    ),
    observations=(
        Observation(
            "whole_organ_homogenate",
            frozenset({"f_parenchymal", "f_macrophage", "f_endothelial",
                       "f_extracellular"}),
            method="ICP-MS or radiolabel on tissue homogenate",
            note=("the measurement the field actually makes: it reports the SUM and "
                  "says nothing about the split")),
    ),
    # NO simplex constraint is declared, and the omission is deliberate.
    #
    # The four fractions are fractions OF organ-associated material, so "they sum
    # to one" is the same equation as "the homogenate measures their total" -- a
    # definition restated, not independent information. Declaring both looked like
    # two equations and made the system appear one degree of freedom better than it
    # is; the module's structural deduplication caught the double count, which is
    # the whole reason equations are compared as unknown-sets rather than counted.
    #
    # A simplex IS informative when the fractions are of something else -- e.g.
    # corona composition as a share of total adsorbed protein, where the total is
    # measured separately. The distinction is whether the normalising total is an
    # independent observation or the same one.
    constraints=(),
)

#: Measurements that could be added. Each is a real assay with a real cost.
CYTOSOL_CANDIDATES = (
    Observation("flow_sorted_parenchymal",
                frozenset({"f_parenchymal"}),
                method="collagenase dissociation + FACS + ICP-MS on sorted cells",
                note="directly measures the parenchymal share; destroys spatial context"),
    Observation("macrophage_depletion_arm",
                frozenset({"f_macrophage"}),
                method="clodronate liposome pre-treatment, paired arms",
                note="infers the macrophage share by difference; perturbs the biology"),
    Observation("perfusion_before_harvest",
                frozenset({"f_extracellular"}),
                method="saline perfusion, reported as a protocol step",
                note="removes residual blood; costs nothing and is often omitted"),
    Observation("quantitative_imaging_colocalisation",
                frozenset({"f_parenchymal", "f_endothelial"}),
                method="confocal or MALDI imaging with cell-type markers",
                note="separates parenchymal from endothelial; relative unless calibrated"),
    Observation("cargo_release_assay",
                frozenset({"phi_intact"}),
                method="dual-label carrier/cargo with separate quantification",
                note="distinguishes carrier-associated from released cargo"),
    Observation("functional_cytosolic_reporter",
                frozenset({"k_escape"}),
                method="cytosol-restricted reporter, e.g. RISC loading or GAL9 recruitment",
                note="the only direct read of productive escape"),
)


# ---------------------------------------------------------------------------
# The cross-study offset. Same formalism, and it explains why the anchor works.
# ---------------------------------------------------------------------------
CROSS_STUDY_OFFSET = StructuralModel(
    name="formulation effect vs study offset, from one study's measurements",
    unknowns=("beta_formulation", "alpha_study"),
    observations=(
        Observation("test_article_measurement",
                    frozenset({"beta_formulation", "alpha_study"}),
                    method="organ delivery of the test formulation in this study",
                    note=("every published value is the SUM of a formulation effect and "
                          "a study offset; one equation, two unknowns")),
    ),
)

#: What a reference arm does, stated as an equation rather than as intuition.
OFFSET_CANDIDATES = (
    Observation("reference_arm_same_run",
                frozenset({"alpha_study"}),
                method="a known reference formulation measured in the same analytical run",
                note=("pins the study offset because the reference's formulation effect "
                      "is known from prior characterisation; this is the whole content "
                      "of reference anchoring")),
    Observation("second_formulation_same_run",
                frozenset({"beta_formulation", "alpha_study"}),
                method="another novel formulation in the same run",
                note=("adds an equation AND an unknown, so it does not close the system "
                      "-- which is why 'more formulations per paper' never fixed this")),
)


# ---------------------------------------------------------------------------
# Loop B: patient state. Included to show the formalism refuses symmetrically.
# ---------------------------------------------------------------------------
PATIENT_STATE = StructuralModel(
    name="per-patient exposure vs responsiveness, from routine measurements",
    unknowns=("exposure_patient", "sensitivity_patient", "assay_bias_site"),
    observations=(
        Observation("biomarker_response",
                    frozenset({"exposure_patient", "sensitivity_patient",
                               "assay_bias_site"}),
                    method="routine clinical biomarker at scheduled visits",
                    note=("one observable confounds how much drug reached the target, "
                          "how responsive this patient is, and the site's assay bias")),
    ),
)

PATIENT_CANDIDATES = (
    Observation("plasma_pk",
                frozenset({"exposure_patient"}),
                method="plasma concentration-time sampling",
                note="pins systemic exposure; still not target-tissue exposure"),
    Observation("site_reference_standard",
                frozenset({"assay_bias_site"}),
                method="a shared reference material run at every site",
                note=("the clinical analogue of the reference anchor -- the SAME fix, "
                      "which is the one genuine structural parallel between the loops")),
)


REGISTRY: dict[str, tuple[StructuralModel, tuple[Observation, ...]]] = {
    "organ_to_cytosol": (ORGAN_TO_CYTOSOL, CYTOSOL_CANDIDATES),
    "cross_study_offset": (CROSS_STUDY_OFFSET, OFFSET_CANDIDATES),
    "patient_state": (PATIENT_STATE, PATIENT_CANDIDATES),
}
