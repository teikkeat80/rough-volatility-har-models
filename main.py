from time import time
import analysis

def main():
    
    # Empirical analysis for Chapter 3
    analysis.p1_empirical_analysis()
    
    # Simulation study for Chapter 4
    # analysis.p2_simulation_analysis()

    # # Empirical analysis for Chapter 4
    # analysis.p2_empirical_analysis()
    # analysis.p2_empirical_analysis_extended()

    return 0

if __name__ == '__main__':
    start_time = time()
    main()
    end_time = time()
    print(f"Elapsed time: {end_time - start_time} seconds")