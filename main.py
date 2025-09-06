from time import time
import analysis

def main():
    
    # Empirical analysis for Chapter 3
    # analysis.p1_empirical_analysis()
    
    # Simulation study for Chapter 4
    # analysis.p2_simulation_analysis()

    # Empirical analysis for Chapter 4
    analysis.p2_empirical_analysis()
    # analysis.p2_empirical_analysis_extended()

    return 0

if __name__ == '__main__':
    start_time = time()
    main()
    end_time = time()
    print(f"Elapsed time: {end_time - start_time} seconds")


#
# Estimated Params:  [-0.14950154  0.30446621  0.1829766   0.0026374  -0.42253119  0.47841441]
# LL:  -480910.6486342164 - (-481788.4457) = 877.797
# AIC:  961833.2972684328
# SE:  [0. 0. 0. 0. 0. 0.]
# RHARK Out-sample rmse: 0.931245680742323
# RHARK Out-sample qlike: 0.34624779247196874
# RHARK Out-sample mae: 0.32157007427782486