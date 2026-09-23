# Equation register and research reading guide

Reviewed 17 September 2026. This is an interpretation of the supplied papers, not a validated implementation. Printed pages of the key FeO, aluminium and weathering equations were visually checked. Source PDFs are now in /Users/max/Desktop/M3 UGR/Preliminary Readings. Page numbers below are printed pages except where PDF page is specified.

## Targets and units

Fe, FeO-equivalent, Al, Al2O3-equivalent, and mineral abundance must have separate variable names. Oxide-equivalent chemistry does not demonstrate a separate free oxide mineral, and elemental abundance does not demonstrate extractable metallic material. Weight percent is mass of a constituent divided by total sample mass times 100.

Spectral radiance L_lambda is power per projected area, per solid angle, per wavelength interval, commonly W m^-2 sr^-1 micrometre^-1. Spectral irradiance E_lambda is incident power per area per wavelength interval, W m^-2 micrometre^-1. A steradian describes angular extent in three dimensions. Reference radiance has the same units as measured radiance. Radiance factor is their dimensionless ratio. Under the standard ideal perfectly diffuse reference at normal illumination, reference spectral radiance is E_lambda/pi and RADF=pi L_lambda/E_lambda. This is the reference definition, not the complete M3 Level 2 processing recipe. Thermal separation and photometric correction matter.

NASA's Level 2 description defines RADF with reference incidence 30 degrees and emission 0 degrees. Its archive ground-truth correction was disabled. Never substitute normalized spectra or MS2-relative values for RADF without the calibration explicitly requiring it.

## 2026 LPSC 1102: FeO estimator

Equation (1), p.1:

FeO_wt% = A theta + B

theta = -ArcTan(R950/R750 - yFe, R750 - xFe)

A=17.427; B=-7.565; xFe=0.08; yFe=1.19.

These are published empirical coefficients, not newly derived constants. Geometric interpretation: set X=R750 and Y=R950/R750. Translate the point by (0.08,1.19), calculate its angle, then apply a straight-line fit to composition. Brightness and spectral contrast jointly inform the estimate. R950/R750 is not automatically a continuum-removed band depth.

The text defines its two-argument ArcTan on 0 to 2pi yet puts a minus sign before it; p.2 also describes constraining theta to (0,2pi). These statements require clarification. NumPy atan2(y,x) returns a signed angle; wrapping before versus after the minus sign changes results. Under A=17.427, adding 2pi changes the prediction by about 109.5 percentage points. That is a mathematical sensitivity demonstration, not an observed lunar abundance. Do not silently select a convention or clip unphysical results.

Page 2 reports a revised origin and refitted A,B but does not give a complete numeric specification or fitting dataset. Exact reproduction of the revised result is pending those details and the observation ID, correction conventions and sample matching. The abstract's broad preprocessing description must not replace the actual PDS product specification.

## Fischer thesis (1995): features and aluminium

Chapter 1, discussion of band depth: D=(Rc-Rb)/Rc=1-Rb/Rc. Rb is reflectance at the absorption band center; Rc is the fitted continuum value there. The continuum is a model of the underlying spectral trend. D=0.2 means a depression 20% below that chosen continuum, not 20% iron. Its anchors and fitting method are part of the result. The thesis emphasizes exposure/maturity and soil-type restrictions when relating band depth to Fe-bearing material.

Chapter 2, p.56 (PDF p.78), Eq.(2):

Al(wt%) = 7.39 R_(0.76/MS2) + 1.65.

R_(0.76/MS2) is reflectance at 0.76 micrometres relative to reference area MS2 at 20-degree phase. It is not an unmodified M3 channel. The reported one-standard-deviation error is 2.08 wt% Al in that study; this is not an uncertainty we can attach to our M3 map. For illustration, an MS2-relative value of 1.2 produces 10.518 wt% Al. This synthetic example demonstrates arithmetic only.

Chapter 2 pp.53-57: the overall correlation partly represents mixing between mare and highland populations; within-population correlations are weak. A global correlation does not prove fine-scale prediction skill. The study resamples optical data to the X-ray support for comparison. This is a direct precedent for avoiding artificial high-resolution ground truth in our ML work.

The thesis gives approximate oxide conversion factors: Al2O3-equivalent =1.89 times elemental Al; FeO-equivalent =1.29 times elemental Fe (p.60). They express molar-mass accounting under the specified oxide reporting convention, not a mineral identification or metal-extraction yield. Its Fe-Al anticorrelation has a stated Fe/Mg caveat; do not impose it globally as a physical law.

## Fischer chapter 3: optical weathering model

These equations are later research candidates, not mandatory preprocessing. Let zeta denote the laboratory exposure index Is/FeO. Is measures a ferromagnetic signal associated with fine metallic iron; this index is neither orbital radiance I nor a direct exposure age in years.

Eq.(1), p.115:

ln[100 R_zeta(lambda)] = ln[100 R_zeta0(lambda)] exp[-q(lambda)(zeta-zeta0)].

It models how log reflectance changes with exposure state for compositionally comparable soil. q(lambda) sets the wavelength-dependent alteration rate. The factor 100 is part of the published convention; do not casually remove it inside a logarithm. Logs require positive inputs. A model of altered appearance is not proof that lost compositional information can be recovered uniquely.

Eq.(2), p.120, printed equation:

Is/FeO = exp[(1.182 - R(0.753)/R(0.889))/0.057].

The preceding prose on p.119 instead gives 0.899 micrometres. Record this discrepancy and consult the original bandpass definitions/cited publication before coding. The relation is calibrated to a particular spectral convention, has a mature-soil plateau treatment, and is not applicable to mare and pyroclastic material in the stated method.

Eq.(3), p.120:

slope = 0.171 + 0.011(FeO+TiO2).

FeO+TiO2 is in wt%. This estimates the slope of q, not directly the observed spectrum or FeO. The thesis notes the strong reported correlation is based on four points (p.123); that limits the evidence for generalization.

Eq.(4), p.121:

C(lambda)=m1 + m2/[sinh(lambda^-1)].

This is an empirical hyperbolic continuum fit, not inverse hyperbolic sine. Use the source wavelength convention when reproducing it; the formula is not invariant to changing numeric wavelength units. The thesis discusses imperfect fits at the ends of the spectrum.

Eq.(5), p.122:

q(lambda)=qf(lambda)-x[1-R_CR(lambda)].

It adds the effects of residual absorption structure to a featureless alteration function. R_CR is the continuum-removed spectrum. An additional convention is specified when the Modified Gaussian Model is used for the continuum; those branches cannot be mixed casually.

Eq.(6), p.123:

x=0.135+0.003(Is/FeO).

This empirical weight controls how much absorption structure enters q. It is not an elemental fraction. Learn the concepts now; do not implement the full model before validating simpler spectral measurements and its applicability.

## 2006 and 2008 Ti-B papers

The core reaction is Ti+2B -> TiB2. Their mixture expression combines this reactive component with JSC-1 simulant through a parameter alpha. Alpha controls the modeled reactive contribution; it should not be assumed to be an elemental abundance or universal mass fraction without checking the authors' definition. Released reaction energy heats and helps consolidate the regolith.

Delta H is reaction enthalpy: negative values conventionally indicate an exothermic reaction. The abstracts display awkward (-Delta H) sign notation; verify thermochemical conventions before numerical energy calculations. Their reported adiabatic-temperature values differ; do not merge them into a universal regolith constant. The 2008 work compares nano and micro precursors: size affects contact and kinetics, not a free creation of chemical energy.

Relevance: a possible downstream processing route and evidence that particle size/microstructure matter. Neither paper gives an M3-to-Fe or M3-to-Al calibration.

## 2012 activated-thermite abstract and ASCE consolidation paper

ASCE Eqs.(1)-(3) specify an approximate multi-oxide simulant composition and additions of n moles of Al or Mg. They are model input mixtures, not a universal composition of lunar soil. The 2012 and ASCE approximations have different CaO/MgO coefficients; retain source-specific compositions.

ASCE Eqs.(4)-(7), p.3, describe fluorocarbon reactions involving metal or oxides, forming fluorides/carbon-bearing products and releasing heat. Their role is to explain activation, additional heat and gaseous products. The gas marker (g) denotes a gaseous phase. These are not equations for interpreting reflected light.

Equation audit: ASCE Eq.(6), as printed, has SiO2 + (1/n)(C2F4)n -> SiF4 + CO. Carbon and oxygen do not balance as written. Treat it as a printed scheme needing clarification; do not feed it directly to a stoichiometric or energy solver. The 2012 abstract also explicitly presents reaction schemes rather than a fully specified reaction network.

Thermodynamics estimates equilibrium products and an ideal adiabatic temperature through conservation and free-energy minimization. Kinetics asks whether a reaction proceeds quickly enough and overcomes barriers. Heat loss and protective oxide layers can prevent propagation even when equilibrium calculations look favorable. Laboratory validation remains necessary.

## Techniques and interpretation vocabulary

- Regolith: loose surface debris; a simulant reproduces selected properties, not every lunar condition.
- Mare/highland: basaltic plains versus generally feldspathic highlands; broad geological populations, not pixel labels inferred solely from brightness.
- Mineral phase: a material with a particular composition/crystal structure; different phases can contain the same element.
- Mafic: rich in Mg/Fe-bearing silicate constituents. Plagioclase is an Al-bearing feldspar; its spectral identification is not a direct Al assay.
- Space weathering: exposure-related changes from the space environment that change optical properties; optical maturity is not a unique age.
- Photometric correction: adjustment for illumination/viewing geometry. Thermal correction: separation of emitted thermal radiation from reflected sunlight.
- Continuum removal: divide by a fitted background trend for feature measurement; preserve the original reflectance for methods requiring it.
- Band center/depth/area: position, relative depression and integrated extent of an absorption; each depends on definitions and processing.
- XRD: X-ray diffraction, identifying crystalline structure from diffraction patterns. Its plotted 2theta is a diffraction angle, unrelated to the FeO estimator's feature-space theta.
- SEM: electron microscopy of grain and pore morphology; EDX: characteristic X-rays used to assess elements. Neither is an M3 measurement.
- Sintering: particles bond/densify through heating, without requiring complete bulk melting.
- SHS/combustion synthesis: a reaction front supplies heat as it propagates; no atmospheric oxygen is necessarily required.
- Adiabatic: idealized absence of heat exchange with the environment. Tad is a modeled temperature under assumptions, not a guaranteed measured Tc.
- Porosity: fraction of volume occupied by voids. Hardness measures resistance to indentation, not automatically structural strength. Thermal conductivity, W/(m K), measures heat conduction and can vary with porosity.
- Gibbs energy: the thermodynamic potential minimized for equilibrium at fixed temperature and pressure, subject to elemental conservation; adiabatic problems also solve the energy balance.

## Final pipeline and next experiment

Archive selection → verified downloads → correct cube/metadata loading → quality diagnosis → matched geometry and calibration → spectral features → separate Fe and Al estimators → independent validation → maps with uncertainty → process-specific ISRU scenarios → laboratory testing.

Next build a small, documented equation notebook using synthetic inputs. Show the FeO feature-space angle and branch issue, the Al reference normalization and its reported scatter, and a continuum/band-depth example. Mark all synthetic values as educational. Then validate the existing cube reader independently and investigate its anomalies before applying chemistry. Seek the 2026 revised fit specifications and resolve the thesis band discrepancy. Keep the existing system architecture, with separate Fe and Al branches and explicit calibration metadata.

Additional primary documentation:
- NASA PDS L2 description: https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=CH1-ORB-L-M3-4-L2-REFLECTANCE-V1.0
- M3 Data Product SIS (radiance units and product structure): https://planetarydata.jpl.nasa.gov/img/data/m3/CH1M3_0003/DOCUMENT/DPSIS.PDF
- M3 instrument paper, Green et al. (2011): https://planetarydata.jpl.nasa.gov/img/data/m3/CH1M3_0003/EXTRAS/PUBLICATIONS/2011JE003797.pdf
