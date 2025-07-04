# MASTER THESIS - INTRODUCING ROUGHNESS INTO HETEROGENEOUS AUTOREGRESSIVE VOLATIITY MODELS - SOURCE CODE

## ABSTRACT
This research introduces two novel methodologies for modelling financial market volatility by simultaneously capturing its empirical characteristics, specifically the long- memory and rough properties, 
both of which were well-documented features in econometrics literature. Our approaches extend upon the popular Heterogeneous Autoregressive (HAR) framework, where key aspects from the rough volatility 
paradigm are incorporated into our model developments. The first model considers a straightforward integration of the rough fractional stochastic volatility (RFSV) dynamics into the HAR cascade structure 
by employing conditional expectations of fractional Brownian motion (fBm) processes as inputs for the three-factor HAR specification. The second model enhances the HAR framework through state-space 
modelling, Kalman filtering, and a Markovian approximation of fBm, to capture rough and persistent behaviours of volatility while accounting for measurement errors within realised volatility (RV) estimates. 
These models are eventually evaluated through extensive simulation and empirical analysis and were found to demonstrate competitive forecasting performance relative to contemporary benchmarks and 
effectively reproduce empirical features of volatility adequately. Beyond its predictive capability, the second framework also provided further insights into uncovering the origins of volatility roughness. 
Ultimately, this work contributes to the growing literature on volatility modelling by offering empirically robust and flexible tools, while also providing a complementary perspective on understanding the 
dynamics of financial volatility more comprehensively.

Keywords: realised volatility, heterogeneous autoregressive, rough volatility, Kalman filter, fractional Brownian motion.

## REPOSITORY DESCRIPTION
- folders:
  - data: source data described in Section 3.3.1 and Section 4.4.1
  - isa_result: Results for Section 4.4.2
  - osa_result: Results for Section 4.4.3
- files:
  - main.py: main codes
  - analysis.py: compiled analysis codes
  - dp.py: data processing codes
  - fbmsim.py: fbm simulation codes
  - hark.py: HARK model and HARK-red model codes
  - rhark.py: rough-HARK model codes
  - mle.py: MLE estimation (HARK, HARK-red, rough-HARK) codes
  - ls.py: OLS estimation and prediction (rough-HAR, HAR, log-rough-HAR, log-HAR) codes
  - rfsv.py: RFSV model codes
  - sspred: State-space model prediction codes
  - nmse.py: numerical standard errors computation codes
  - hurst.py: structure-function estimation codes
  - loss.py: loss functions codes
  - visualisation.py: plots and figures codes.
