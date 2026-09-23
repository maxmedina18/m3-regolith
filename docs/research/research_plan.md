# M³ Apollo 17 research plan: the first four weeks

Prepared 16 September 2026. This is a proposed research protocol, not a report of completed data analysis.

## Research objective and scope

**First research question:** Can we reproduce the documented Apollo 17 FeO procedure, establish exactly which calibrated reflectances it uses, and evaluate the resulting estimates against appropriately matched lunar-soil measurements?

The two-week minimum outcome is a documented literature audit, a verified reader for one Apollo 17 observation, and a reproducible data-quality report. The four-week target adds an audited FeO implementation, a comparison with reference soil compositions, and an explicit account of calibration sensitivity and unresolved reproduction gaps. A demonstrated failure or a precisely documented inability to reproduce a published result is a scientifically useful outcome.

The long-term chain remains:

**M³ reflectance → documented preprocessing → spectral observables → Fe/mineralogical inference → validation → maps → conditional ISRU assessment.**

For the first month, quantitative FeO and qualitative spectral evidence are separate products. Mineral abundances, mineral classifications, and construction suitability require additional validation. PCA, GMM, random forests, and other ML methods are deferred.

The schedule assumes approximately 10–15 focused hours per week and a short weekly mentor meeting. These are planning assumptions, not literature requirements. At lower availability, preserve the scientific gates and extend the schedule.

## 1. What the supplied documents establish

I reviewed the five shorter supplied papers and the relevant spectroscopy, maturity, and modeling sections of Fischer’s 216-page dissertation. This is not a claim of a complete line-by-line review of the dissertation. Its scanned equations require visual checking before any later implementation.

| Source | Evidence relevant to this project | Limits on what we may conclude |
|---|---|---|
| **[A] Trevino et al. (2026), LPSC #1102**, `1102.pdf`, pp. 1–2 | Applies a two-wavelength FeO relationship to M³ Level 2 pixels around Apollo 17; reports extreme predictions and a revised origin/regression; Figure 2 is the reproduction target. | A two-page conference abstract is an incomplete methods specification. A visually improved map does not establish independent predictive accuracy. |
| **[B] Fischer (1995)**, `PhFFisher.pdf`, especially Chapter 1, printed pp. 1–22, and Chapter 3, printed pp. 93–98 and 113–128 | Explains composition–maturity ambiguity, absorption depth, spectral sampling, laboratory comparisons, and an optical alteration model. Chapter 1 reproduces Fischer and Pieters (1994), *Icarus* 111, 475–488. | Its mature-highland band-depth relations and maturity thresholds must not be transferred indiscriminately to Apollo 17 mare or pyroclastic material. The filename says “Fisher”; the author is **Fischer**. |
| **[C] Martirosyan & Luss (2006), LPSC #1896**, `37LPS_1896.pdf`, pp. 1–2 | Reports consolidation of JSC-1 with a titanium–boron reaction system; examines a propagating thermal front, product structure, and hardness. | This is a particular simulant/additive experiment, not an empirical FeO-to-construction-performance calibration. |
| **[D] Martirosyan & Luss (2008), LPSC #1689**, `NANOENERGETIC FABRICATION OF DENSE CERAMICS FOR LUNAR EXPLORATION.pdf`, pp. 1–2 | Compares nano- and microscale reactants in a Ti–B/JSC-1 system, reporting changes in propagation and temperature. | The experimental section specifies an argon environment despite discussion of lunar vacuum requirements. Do not describe these experiments as an on-Moon or universal vacuum demonstration. |
| **[E] Hobosyan & Martirosyan (2012), LPSC #1019**, `43rdLPSC_1019.pdf`, pp. 1–2 | Investigates activated regolith–Al mixtures and links activation to consolidation, porosity, and thermal behavior. | The later journal paper is the more detailed source; distinguish ignition from stable propagation. |
| **[F] Hobosyan & Martirosyan (2014 online; 2015 journal issue)**, `Consolidation of Lunar Regolith Hobosyan Martirosyan ASCE 2014.pdf`, pp. 1–9 | Gives thermodynamic analysis and experiments for JSC-1A with Al or Mg and PTFE; measures propagation, porosity, thermal conductivity, and hardness. | Neither hardness nor sample appearance alone establishes structural qualification. Its experimental variations do not isolate lunar FeO abundance as the sole predictor of performance. |

**Established within the supplied experiments:** particular mixtures can consolidate regolith simulant under specified conditions. **Author proposal in [A]:** orbital FeO information can help select candidate feedstocks. **Our inference:** useful feedstock screening will require a validated link between orbital observables, the actual mineral/chemical feedstock, and process-specific laboratory outcomes. That link is not demonstrated by an FeO map alone. [A, pp. 1–2; C–E, pp. 1–2; F, Experimental Methods and Results and Discussion.]

## 2. Physics to understand before coding an estimator

### A spectrum is a light measurement, not a chemical assay

A pixel spectrum records wavelength-dependent light from an area of terrain. Sunlight interacts with particles through scattering and absorption; some light escapes toward the instrument. A darker pixel may reflect composition, illumination, particle properties, or surface weathering. An empirical chemical estimate therefore requires a calibration that separates or controls these effects.

Radiance is the instrument-facing light measurement. Reflectance products normalize it according to a defined radiometric and geometric convention. For the archived M³ Level 2 V1.0 dataset, “reflectance” means a unitless radiance factor normalized to incidence 30° and emission 0°. It is not an arbitrary per-spectrum normalization. [G]

### Why iron-bearing minerals produce absorptions

Ferrous iron, Fe²⁺, in a mineral’s lattice has electronic energy levels influenced by surrounding atoms. Photons with suitable energies can be absorbed in transitions between these levels, reducing reflected light over wavelength bands. This explains why mineral chemistry and lattice environment affect band position and shape. Grain properties and mixtures also affect the spectrum. The common lunar absorption around 1 µm is associated with ferrous iron, but its depth depends on more than iron abundance. [B, Chapter 1, printed pp. 10–12; H, Introduction.]

Fischer discusses pyroxene absorptions in both the approximately 1 and 2 µm regions and olivine absorption in the 1 µm region. Those are interpretive clues, not sufficient rules for assigning a mineral to a mixed orbital pixel. In particular, a weak or absent feature is not proof that the corresponding material is absent. [B, Chapter 1, printed pp. 10–12.]

### Why maturity matters

Space weathering changes soil optical properties: reflectance commonly decreases, the continuum becomes redder, and absorptions weaken. Here “redder” means reflectance increases more strongly toward longer wavelengths; it does not mean a visible red rock. Fischer’s compositionally similar Apollo 16 soils illustrate these changes. Consequently, a deeper band does not uniquely imply greater FeO. [B, Chapter 1, printed pp. 3–12, Figure 1.]

Do not interpret an optical maturity index as a direct calendar age. Do not assume a highland-specific maturity discriminator applies to a mare scene. Fischer explicitly discusses limits for mare and pyroclastic material. [B, Chapter 1, printed pp. 15–22; I.]

### Three quantities that must remain distinct

- **Reflectance:** a wavelength-dependent optical quantity with a specified calibration and geometry.
- **FeO wt.%:** a bulk chemical abundance convention used by the calibration/reference analyses. It is not a measured fraction of discrete FeO mineral grains. Track whether a chemical source reports FeO, total iron as FeO, elemental Fe, or Fe₂O₃; they are not interchangeable labels.
- **Mineral abundance:** proportions of particular mineral phases. The same bulk iron abundance can be distributed among different phases, with different optical and processing behavior.

This distinction is essential when comparing the spectroscopy papers with the JSC-1A chemistry and thermodynamic systems in [F, pp. 2–3]. Do not silently substitute total iron for ferrous iron or turn a chemical FeO map into an ilmenite map.

### Continuum removal has a specific purpose

For a chosen continuum estimate C(λ), continuum-removed reflectance is R(λ)/C(λ). The usual band-depth definition at the band center is:

\[
BD=1-\frac{R(\lambda_b)}{C(\lambda_b)}.
\]

Here λ_b is the selected/fitted absorption center, not automatically 950 nm. This is an absorption measurement definition, not a new FeO calibration. Fischer explains why a two-channel ratio combines band and slope information and why fixed continuum anchors can bias measurements. [B, Chapter 1, printed pp. 19–21.]

**Keep a separate absolute-reflectance branch for FeO.** Continuum removal or normalization to unity at one wavelength alters the quantities entering the empirical FeO relation. A useful plot transformation is not automatically permissible estimator preprocessing.

## 3. Reproduction audit of the 2026 Apollo 17 study

### What is specified

[A] identifies Level 2 data, GDAL for binary reading, per-pixel calculation without averaging, reflectances near 750 and 950 nm, initial calibration constants, and a later revised origin plus regression. It does not uniquely identify the image or fully specify the revised analysis.

The published initial constants in [A, p. 1] are A = 17.427, B = −7.565, x₀ = 0.08, y₀ = 1.19. They also appear in the FeO algorithm paper by Lucey, Blewett, and Jolliff (2000). [J, §3.1, equations 1–2.]

**Bibliographic discrepancy:** reference [6] of [A] gives page 20,377, which identifies *Imaging of lunar surface maturity* [I]. The detailed iron/titanium algorithm paper begins at page 20,297 [J]. Read both; record the discrepancy rather than silently repairing the bibliography.

### The angle convention requires an explicit decision

For the original calibration, define x = R₇₅₀ − 0.08 and y = R₉₅₀/R₇₅₀ − 1.19. A quadrant-aware implementation of the signed-angle prescription is:

\[
\theta_{Fe}=-\operatorname{atan2}(y,x),\qquad
\widehat{FeO}=17.427\,\theta_{Fe}-7.565,
\]

with θ in radians. This is a software translation to audit against [J], not a resolution of [A]’s ambiguity. [J] explicitly requires an inverse tangent spanning −π to +π. [A] describes a 0-to-2π ArcTan, places a minus sign outside it, and later describes constraining θ to positive angles. The order of sign changes and wrapping is therefore not sufficiently specified for exact replication.

Test coordinate quadrants, axes, points near the adopted origin, zero/invalid reflectances, and the branch cut. A one-argument arctangent can lose quadrant information. Conversely, blindly wrapping an angle into a positive interval can create extreme FeO values. Preserve such outputs as diagnostic results with invalidity flags; do not hide them by clipping to an attractive range.

### Information to request from the authors or mentor

1. Exact M³ product ID(s), acquisition mode/date, archive version, ancillary files, and Figure 2 crop/order/orientation.
2. Actual channels and wavelengths, plus any interpolation or band-response matching.
3. Complete processing order: correction state, ground-truth correction choice, filters, masks, and the meaning of “continuum normalization.”
4. Exact angle code, argument order, units, sign, and wrapping operation.
5. Revised x₀, y₀, A, and B, and the operational rule for choosing the revised origin.
6. Calibration sample/station identities, chemical sources, spatial extraction regions, regression weights, and any independent validation observations.
7. Numerical Figure 2 products or reference pixel values, units, no-data convention, and plotting limits.

No messages have been sent to the authors. This is a list for the student’s mentor discussion.

Maintain three named analyses: **original-calibration diagnostic**, **2026 reproduction with confirmed settings**, and **new sensitivity experiments**. If the exact scene is unavailable, an alternate Apollo 17 observation is a methodological replication attempt, not a reproduction of Figure 2. If revised constants are missing, Figure 2c remains unreproduced.

### Preprocessing claims must be checked against the archive

The V1.0 Level 2 archive documents solar-irradiance correction, statistical polishing, thermal removal, photometric correction, and degraded-channel flagging. Its optional ground-truth correction was deliberately not applied; calibration files are supplied for user application. Continuum removal is not a listed standard archive step. [G]

Thus [A]’s general preprocessing paragraph is not an executable recipe. Audit the exact product rather than adding every named operation. Isaacson et al. show that ground-truth correction can materially affect weak 1 µm features and differs with operational conditions; its corrected wavelength range also warrants caution when interpreting longer-wavelength relationships. [H]

## 4. Major research steps and decision gates

All implementation choices and gates below are **proposed for this project**. They are not claims about unreported methods in [A].

### Step 1 — Establish the physical and evidential basis

- **Scientific objective:** Define the observable, target quantity, reference evidence, and limits of the published claim.
- **Physical reasoning:** Composition, weathering, particle properties, and observing conditions can affect the same spectral measurements.
- **Required background:** Units; mass versus modal abundance; absorption and scattering; basic regression; calibration versus validation.
- **Data required:** [A]–[F], M³ documentation, [H]–[J]; no full image archive yet.
- **Mathematical method:** Dimensional checks and explicit definitions of reflectance, band ratio, continuum, angle, and regression target. Sketch the ratio–reflectance plane.
- **Implementation plan:** Build a literature ledger with source/page, claim, evidence type, calibration population, limitations, and project consequence. Build the missing-methods list above.
- **Validation:** Give a ten-minute explanation to the mentor of why band depth, FeO, and mineral fraction differ. Explain how weathering can mimic composition.
- **Failure modes:** Treating conference claims as independently validated results; transferring highland-only relations to mare soils; confusing units or chemical conventions.
- **Expected output:** Two-page physics note, evidence ledger, and reproduction specification with explicit unknowns.
- **Move on when:** Every planned operation has a source or is labeled as our own experimental choice; unresolved details have explicit effects on the reproduction claim.

### Step 2 — Read and locate one M³ observation correctly

- **Scientific objective:** Demonstrate that a selected pixel is the intended ground location and contains the intended calibrated spectrum.
- **Physical reasoning:** A well-written algorithm applied to the wrong location, band order, or data quantity is scientifically meaningless.
- **Required background:** Image cubes; array axes; binary types and interleave; wavelength units; lunar coordinates and map projections.
- **Data required:** One Level 2 observation covering Apollo 17, its label/header, wavelength information, matching location/geometry products and supplemental files, calibration references, and browse/context image. Use the author’s observation if available.
- **Mathematical method:** Map array indices to wavelength and location using product metadata; retain native pixel geometry; select a geographic region using actual coordinates.
- **Implementation plan:** Write a reader that returns data, wavelengths, coordinates, metadata, and masks with named axes. Save a download manifest, checksums, product/version IDs, URLs, acquisition mode/date, and local file roles. Read small windows before the full cube.
- **Validation:** Compare dimensions and expected storage with labels; verify wavelengths and units; independently inspect several fixed pixel/band values; overlay the footprint on lunar context imagery and inspect multiple recognizable landmarks.
- **Failure modes:** RDN mistaken for RFL; reversed axes; zero-based versus one-based band indices; incorrect byte order; mismatched ancillary acquisition; longitude convention errors; treating a located swath as a regular projected raster.
- **Expected output:** Scene inventory, footprint plot, a near-750-nm image, and spectra from clearly identified pixels.
- **Move on when:** Data values agree with an independent read at selected checks and geolocation agrees within the documented uncertainty. Unexplained offsets or metadata conflicts halt chemical interpretation.

### Step 3 — Establish valid reflectances and a minimal preprocessing baseline

- **Scientific objective:** Identify which measurements support inference and what corrections were already applied.
- **Physical reasoning:** Low signal, artifacts, illumination geometry, and correction errors can dominate ratios or weak bands.
- **Required background:** Signal/noise; missing values; photometric versus thermal effects; multiplicative calibration; error propagation.
- **Data required:** Step 2 products plus applicable quality flags, archive limitations, observing conditions, and any required ground-truth calibration tables.
- **Mathematical method:** Apply documented fill/degraded-channel masks; inspect finite values, distributions, spatial stripes, and wavelength discontinuities. Treat any extra quality threshold as a justified analysis choice.
- **Implementation plan:** Preserve immutable archive data. Save separate masks by reason and a processing ledger. Start with no extra smoothing, destriping, interpolation over gaps, or continuum removal. If evidence warrants an optional correction, add a separately named branch and record its source/version.
- **Validation:** Plot before/after spectra at the same pixels; report count and area removed by each mask; test behavior on invalid inputs; compare suitable overlap observations if available.
- **Failure modes:** Double correction; wrong warm/cold correction; masking pixels because their inferred chemistry looks inconvenient; smoothing away shallow bands; assuming all finite pixels are reliable; replacing missing reflectances with zeros.
- **Expected output:** Audited reflectance subset, mask maps, correction-state table, and quality report.
- **Move on when:** Every operation and exclusion is reproducible, key FeO channels are usable, and suspected artifacts are either resolved or explicitly retained as limitations. Do not claim a quantitative uncertainty distribution without evidence for it.

### Step 4 — Reproduce the FeO computation and its failure behavior

- **Scientific objective:** Separate coding correctness, faithful reproduction, and validity of calibration transfer to M³.
- **Physical reasoning:** The angular parameter combines brightness and spectral ratio relative to an empirical reference point. Its calibration depends on the measurement system and sample population.
- **Required background:** Radians; atan2; linear regression; numerical conditioning; empirical calibration domains.
- **Data required:** Valid absolute reflectances, actual wavelength centers, original calibration [J], and all available author settings for [A].
- **Mathematical method:** Implement the audited relation in Section 3. For the revised version use supplied coefficients only. Match the publication’s wavelength extraction if known; otherwise compare explicitly labeled nearest-channel and interpolation alternatives as sensitivity experiments, without asserting exact reproduction.
- **Implementation plan:** Use a pure function returning theta, estimated FeO, and reason flags. Store constants with citations in configuration. Export R₇₅₀, R₉₅₀/R₇₅₀, theta, quadrant, and distance from the reference origin alongside the FeO result. Keep native-pixel estimates before mapping.
- **Validation:** Hand-check a small synthetic table and selected real pixels; verify sign/quadrant behavior independently; compare author-provided numerical values if obtained. Reproduce a published station-calibration plot where underlying values are available; identify digitized figure values as approximate.
- **Failure modes:** Wrong sign/angle branch; percent reflectance versus fractional reflectance; normalization before estimation; choosing revised constants to make a plausible-looking map; assuming close channel centers imply interchangeable instrument bandpasses.
- **Expected output:** Original-calibration diagnostic maps, scatterplots, anomaly inventory, and—only if fully specified—a separately labeled revised-calibration reproduction.
- **Move on when:** Numerical checks pass and all departures from the source are listed. Exact numerical agreement uses a tolerance tied to input precision/reference rounding, not an invented chemical-accuracy threshold. Missing author information limits the reproduction claim but does not prevent diagnosis.

### Step 5 — Validate against matched soil evidence and quantify sensitivity

- **Scientific objective:** Determine how well predictions represent independently characterized material and identify what limits that assessment.
- **Physical reasoning:** An orbital pixel averages terrain; a laboratory rock or tiny soil aliquot may not represent that footprint. Registration and mixed terrain matter.
- **Required background:** Sampling support; bias; MAE/RMSE; grouped validation; spatial dependence; uncertainty versus sensitivity.
- **Data required:** Source-traceable Apollo 17 soil/station chemistry, sample type and fraction, coordinate provenance, uncertainty where available, and matching image regions. A published map can provide a secondary comparison if its provenance and effective resolution are known.
- **Mathematical method:** Compare estimates and chemistry at matched support; compute residuals, mean bias, MAE, and RMSE in FeO percentage points. Do not count neighboring pixels or multiple aliquots from one station as independent samples. Compare averaging reflectances before the nonlinear estimator with averaging pixel estimates; they are different estimands.
- **Implementation plan:** Build a reference table with IDs, sources/pages, composition convention, coordinates/frame, material type, and calibration/test role. Predefine extraction regions and sensitivity choices. Report changes from band selection, correction branch, mask, and plausible registration/footprint alternatives.
- **Validation:** Show predicted-versus-reference scatter, residuals by station/terrain/quality, and sample counts at the independent-unit level. If refitting, choose origins and coefficients using training groups only; hold out whole stations or sites where feasible. Historical calibration stations test consistency, not independent predictive accuracy.
- **Failure modes:** Using basalt rock chemistry as a soil-pixel target; reusing fit data as “validation”; selecting a best-fitting pixel after inspecting chemistry; leakage from preprocessing selection; comparing different spatial resolutions without matching support.
- **Expected output:** Traceable reference table, validation plots, uncertainty/sensitivity budget, and explicit statement of independence and sample-size limitations.
- **Move on when:** Matching and validation design are defensible and conclusions reflect the available support. If independent references are inadequate, report limited consistency evidence. Proceed to new calibration claims only after independent assessment becomes possible.

### Step 6 — Measure spectral features as a small, separate extension

- **Scientific objective:** Learn what information the full spectra add and whether features survive reasonable analysis choices.
- **Physical reasoning:** Band positions and shapes reflect mineral/chemical information, but continuum placement and mixed spectra can bias estimates.
- **Required background:** Continuum fitting; numerical interpolation; distinction between band minimum and fitted center; laboratory-to-orbital spectral sampling.
- **Data required:** A small set of quality-controlled M³ spectra and documented laboratory reference spectra with sample identity, grain size, viewing geometry, and instrument information.
- **Mathematical method:** On the feature branch, calculate continuum-removed spectra and band depth using the definition above; test candidate continuum anchors and, if justified, a local center fit. Use wavelength coordinates rather than band numbers. Report undefined/unstable parameters explicitly.
- **Implementation plan:** Implement only after the baseline is archived. Begin with a few independently chosen terrain regions and reference spectra. Use synthetic continua/absorptions to check recovery and broad features to inspect sampling effects. Fixed test anchors are experimental choices, not published M³ processing constants.
- **Validation:** Inspect the continuum and fitted region over each spectrum; perturb defensible anchors and channel selections; quantify how much parameters change. Spectral agreement with a laboratory reference is qualitative unless a validated mixture/calibration model supports a stronger claim.
- **Failure modes:** Assigning minerals from one index; fitting a noise minimum; confusing a shifted raw minimum with a band center; treating continuum removal as a complete weathering correction; transferring Fischer’s highland relation to mare pixels.
- **Expected output:** Small spectral atlas, parameter table, and stability report, with tentative interpretations clearly labeled.
- **Move on when:** Features are stable enough for the intended comparison, with an agreed resolution/uncertainty criterion. This is optional in week 4; unfinished FeO auditing takes priority. No ML is required.

### Step 7 — Produce interpretable maps and define the ISRU evidence gap

- **Scientific objective:** Communicate where estimates are supported and what additional evidence feedstock assessment needs.
- **Physical reasoning:** A map can conceal missing coverage, extrapolation, and support mismatch. Process behavior depends on the material and the chosen reaction system.
- **Required background:** Map projection/resampling; spatial footprint; chemical versus mineral maps; heat release versus reaction kinetics and final properties.
- **Data required:** Native-pixel estimates, quality and sensitivity layers, verified geolocation, reference locations, and the experimental evidence table from [C]–[F].
- **Mathematical method:** Map only supported quantities. Explicitly choose the resampling/aggregation rule; categorical flags require categorical treatment. Do not interpolate a coarse validation source into artificial fine-resolution truth.
- **Implementation plan:** Export an FeO estimate layer with units/calibration, quality mask, extrapolation/instability flags, coverage, and separate sensitivity layers. A georeferenced format must use a verified lunar reference system. Keep native products for checking.
- **Validation:** Check map orientation and landmark alignment after export; verify selected exported values against native pixels; ensure excluded areas are missing rather than zero and that display limits do not conceal numerical anomalies.
- **Failure modes:** Calling FeO “mineral abundance”; interpreting smooth resampled pixels as added information; labeling high-FeO areas “best building sites”; generalizing a simulant experiment to all regolith.
- **Expected output:** Map plates and a one-page ISRU evidence-gap memo. The memo should identify required mineralogy, oxidation-state/chemical information, particle properties, additives, environment, processing conditions, and mechanical/thermal outcomes.
- **Move on when:** A reader can distinguish measurement, calibration inference, unvalidated extrapolation, and hypothesis. A later ISRU ranking requires process-specific laboratory validation; the first month does not supply it.

## 5. Exactly what to do during weeks 1–4

| Week | Read and understand | Reproduce and code | Validate and document | Deliverable and gate |
|---|---|---|---|---|
| **1: physical basis and methods audit** | Read [A] completely, including Figure 2 and references. Read [B] Preface, then Chapter 1 printed pp. 1–22 and Figures 1, 4, 7, 9. Read [J] §3.1, Table 1, and Figure 1. Read the Level 2 dataset description [G] and M³ SIS instrument/product overviews. | Redraw the reasoning behind the ratio–reflectance diagram. Create literature and data-manifest templates; write small synthetic angle/unit checks. Do not fit a new calibration. | Explain reflectance, FeO, band depth, maturity, and validation independence to the mentor. Record missing scene/settings, angle ambiguity, and citation discrepancy. | Physics note; evidence ledger; reproduction specification; author-information checklist. Gate: scientific target and unresolved assumptions are explicit. |
| **2: verified data and baseline** | Read matching SIS §§2.5.3.2–2.5.3.3, §2.5.4 including Limitations, §2.6.3, and the detailed Level 2 product specifications. Read [H] Introduction and correction development/evaluation. | Obtain one observation plus metadata/ancillaries; implement loading, coordinate handling, wavelength lookup, masks, and plotting. Inspect at least six documented example spectra spanning different brightness/quality conditions. Finish the original-calibration function and synthetic checks if time permits. | Compare two reading paths or an independent pixel inspection; check the footprint and correction history; record mask counts. Freeze a small reproducible subset and baseline configuration. | Scene inventory, reader, spectrum sheet, footprint, quality report. **Two-week stopping point:** a defensible data foundation, not a claim of completed Figure 2 reproduction. |
| **3: Apollo 17 FeO reproduction attempt** | Re-read [A] Methodology/Results and [J] §3.1. Inspect original reference chemistry and coordinates, not only secondary summaries. | Run the original-calibration diagnostic on native pixels. Plot ratio–reflectance space, theta, FeO, and anomaly locations. Reproduce confirmed author settings separately. Implement revised coefficients only if supplied. | Compare numerical reference values when available. Build station matching, inspect registration, and distinguish calibration reuse from independent tests. Record every undocumented choice. | FeO diagnostic package; reproduction comparison; source-traceable station table; first residual plots. Gate: correct arithmetic and honest reproduction status, not visual similarity alone. |
| **4: validation, sensitivity, and a research decision** | Read [F] fully, prioritizing Experimental Methods and Results; read [C]–[E] fully for comparison. Revisit [B] Chapter 1 printed pp. 18–22; optionally Chapter 3 printed pp. 113–128 for model assumptions/uncertainties. | Run a small predeclared sensitivity matrix: wavelength selection, correction state where defensible, masks, and spatial matching. Export maps plus quality layers. Only if the FeO work is settled, build the small continuum-removal feature experiment. | Report bias/MAE/RMSE with independent-unit counts and limitations; document uncertainty sources without inventing noise levels. Compare process studies in an evidence table. Re-run the analysis from configuration in a clean process. | Four-to-six-page research memo, map plates, reproducible run record, and a decision: continue calibration, repair data/validation, or begin a bounded spectral-feature study. |

Suggested weekly allocation: 3–4 hours reading/notes, 5–7 hours implementation/analysis, 2–3 hours validation and reporting. In week 1, favor reading; in weeks 3–4, favor validation. The time budget is adjustable; the gates are the important part.

**If author information does not arrive:** use week 3 for the original-calibration diagnostic and week 4 for the transfer/sensitivity analysis. Title the result “Apollo 17 FeO calibration audit and partial replication,” with Figure 2c explicitly unresolved. Do not manufacture a revised origin or regress to a regional average just to complete the schedule.

## 6. Minimal software/repository design

Suggested structure for the future project; this plan does not claim these modules have already been implemented:

```text
m3-lunar-mapper/
  README.md
  pyproject.toml
  environment.lock
  configs/
    apollo17_baseline.yaml
    sensitivity_cases.yaml
  data/
    manifests/                 # URLs, IDs, versions, checksums, file roles
    raw/                       # immutable downloaded products; ignore in Git
    reference/                 # chemistry, coordinates, laboratory spectra + sources
    interim/                   # cached subsets; reproducible, not manually edited
  src/m3mapper/
    io.py                      # product/label/header reading; named dimensions
    metadata.py                # acquisition, units, wavelengths, provenance
    geometry.py                # pixel location, coordinate checks, ROI matching
    quality.py                 # separate masks with reason codes
    corrections.py             # only documented, explicitly enabled operations
    wavelengths.py             # wavelength selection and declared interpolation
    feo.py                     # cited calibration and angle convention
    continuum.py               # separate feature-analysis branch
    features.py                # depth/center measurements and validity flags
    validation.py              # station comparison, residuals, grouped evaluation
    mapping.py                 # verified export and display
    provenance.py              # input/config/code/environment record
    cli.py                     # repeatable entry points
  tests/
    fixtures/                  # tiny synthetic cubes; non-sensitive sample tables
    test_binary_layout.py
    test_units_and_wavelengths.py
    test_angle_and_invalid_inputs.py
    test_geolocation_and_matching.py
    test_feature_recovery.py    # only when feature extraction is implemented
  notebooks/                   # exploration and narrative; import src functions
  docs/
    literature_ledger.csv
    physics_notes.md
    reproduction_spec.md
    decision_log.md
    uncertainty_budget.md
  results/<run_id>/
    manifest.json
    figures/
    tables/
    maps/
    report.md
```

Use NumPy/Pandas and plotting tools you already know. Choose a reader that demonstrably supports the actual products; GDAL is the reader identified in [A]. Additional abstractions are optional. Do not spend the first month building an interactive platform.

The baseline configuration records exact product IDs, region definition, wavelength units and selection, correction state, masks, calibration constants and source, angle convention, chemical target convention, and export settings. Unconfirmed author parameters remain null/unset; the reproduction command should fail clearly if they are required.

Scientific functions should return values **and validity flags**, avoid modifying inputs, and preserve units/provenance. Notebooks should call these functions rather than contain the sole implementation. Never silently substitute a new calibration, correction, wavelength, or coordinate system.

Tests should catch scientifically meaningful mistakes: wrong interleave, percent/fraction mismatch, radians/degrees, reversed atan2 arguments, lost quadrants, missing-data propagation, coordinate mismatch, and feature distortion. Software correctness is necessary but does not validate mineralogy.

## 7. What the first-month report should contain

1. **Question and scope:** the observable, chemical target, study area, and level of reproduction attempted.
2. **Sources and data:** exact publications, product IDs, versions, source chemistry, and sample/coordinate provenance.
3. **Methods:** correction audit, masks, wavelength handling, formula/angle convention, and native-to-map transformation.
4. **Results:** calibrated spectra; ratio–reflectance plot; FeO and quality maps; station comparison; sensitivity results.
5. **Failure analysis:** anomalous values, unresolved method details, spatial mismatches, unstable estimates, and excluded data with reasons.
6. **Claims and limits:** what was reproduced, what was merely consistent, what was independently validated, and what remains hypothesis.
7. **Next experiment:** the single uncertainty whose resolution would most improve the science.

Maintain an uncertainty budget separating instrument/radiometric effects, correction choices, wavelength/bandpass differences, reference chemistry, spatial support/geolocation, and calibration/model transfer. Spatial variability within a region is not automatically instrument noise. Sensitivity to processing choices is not automatically a confidence interval. If a probabilistic error model is not supported, report a sensitivity envelope and missing uncertainty information explicitly.

A later ML study is justified only by a defined scientific hypothesis—such as additional spectral information improving held-out FeO prediction—tested against this simple baseline with sufficient independent labeled support. Unsupervised clusters may reveal spectral structure, but they do not become mineral labels without external evidence.

## 8. Reading references and access

### Supplied primary context

- **[A]** Trevino, A., Pereiro, Y. V., Zyskin, M., & Martirosyan, K. S. (2026). *From Orbital Spectroscopy to Lunar Bricks: Combustion Synthesis Guided by M3 Mineral Mapping.* 57th LPSC, abstract 1102. Supplied `1102.pdf`; [public abstract](https://www.hou.usra.edu/meetings/lpsc2026/pdf/1102.pdf).
- **[B]** Fischer, E. M. (1995). *Quantitative Compositional Analysis of the Lunar Surface from Reflectance Spectroscopy: Iron, Aluminum, and a Model for Removing the Optical Effects of Space Weathering.* Brown University PhD dissertation. Supplied `PhFFisher.pdf`. For the main chapters, PDF page = printed Arabic page + 22. Read printed pp. 1–22 as PDF pp. 23–44; Chapter 1 Figures 1–9 follow the captions and table. Chapter 3 printed pp. 113–128 are PDF pp. 135–150. The Preface is PDF pp. 12–17.
- **[C]** Martirosyan, K. S., & Luss, D. (2006). *Combustion Synthesis of Ceramic Composites from Lunar Soil Simulant.* 37th LPSC, abstract 1896. Supplied `37LPS_1896.pdf`.
- **[D]** Martirosyan, K. S., & Luss, D. (2008). *Nanoenergetic Fabrication of Dense Ceramics for Lunar Exploration Program.* 39th LPSC, abstract 1689. Supplied nanoenergetic paper; [public abstract](https://www.lpi.usra.edu/meetings/lpsc2008/pdf/1689.pdf).
- **[E]** Hobosyan, M. A., & Martirosyan, K. S. (2012). *Sintering of Regolith by Activated Thermites: A Novel Approach for Lunar In Situ Resource Utilization.* 43rd LPSC, abstract 1019. Supplied `43rdLPSC_1019.pdf`; [public abstract](https://www.lpi.usra.edu/meetings/lpsc2012/pdf/1019.pdf).
- **[F]** Hobosyan, M. A., & Martirosyan, K. S. *Consolidation of Lunar Regolith Simulant by Activated Thermite Reactions.* Online publication 2014; *Journal of Aerospace Engineering* 28(4), 04014105, 2015 issue. Supplied nine-page paper; [DOI](https://doi.org/10.1061/(ASCE)AS.1943-5525.0000447).

### Additional primary documentation and literature

- **[G]** NASA PDS, [M³ Level 2 Reflectance V1.0 dataset description](https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=CH1-ORB-L-M3-4-L2-REFLECTANCE-V1.0). Read Overview, Processing, Data, and Coordinate System. Then obtain the **matching archive version** of *M³ Data Product Software Interface Specification*, especially §§2.1–2.4, 2.5.3.2–2.5.3.3, 2.5.4, 2.6.3, and detailed Level 2 specifications. The [indexed SIS copy](https://planetarydata.jpl.nasa.gov/img/data/m3/CH1M3_0003/DOCUMENT/DPSIS.PDF) has Level 2 processing starting on printed p. 41 and Limitations on p. 60; other versions differ. The full SIS was not retrieved during this review because of fetch failures; section references were checked against its indexed contents. Verify the copy accompanying your selected data.
- **[H]** Isaacson et al. (2013). [*Development, importance, and effect of a ground truth correction for the Moon Mineralogy Mapper reflectance data set*](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/jgre.20048), *JGR: Planets*, 118, 369–381. Prioritize the introduction, correction rationale, and comparison of operational conditions.
- **[I]** Lucey, Blewett, Taylor, & Hawke (2000). [*Imaging of lunar surface maturity*](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/1999JE001110), *JGR*, 105(E8), 20377–20386. Read for maturity context and to resolve [A]’s reference [6].
- **[J]** Lucey, Blewett, & Jolliff (2000). [*Lunar iron and titanium abundance algorithms based on final processing of Clementine ultraviolet-visible images*](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/1999JE001117), *JGR*, 105(E8), 20297–20305. Prioritize Table 1, §3.1, equations 1–2, and Figure 1. Its station table includes compositionally diverse Apollo 17 soils; do not substitute a single regional basalt range for station references. Legacy coordinates must be reconciled with the current image frame.
- For reference selection, use the [LPI Lunar Sample Atlas](https://www.lpi.usra.edu/lunar/samples/atlas/) as an entry point, then cite the specific sample documentation and original chemical analyses used. An atlas landing page is not a substitute for a traceable calibration table.

The central research discipline is to preserve the boundary between an observed spectrum, a calibrated inference, an independently tested result, and a proposed ISRU consequence.
