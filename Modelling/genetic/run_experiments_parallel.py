import sys
sys.path.append('./')
import itertools

from Utilities.imports import *

from Modelling.genetic.stochastic_vs_efficient import *
from Modelling.genetic.ga_utils import *


if __name__ == "__main__":
    left_l, right_l, danger, predator_memory, predator_risk_factor, reproduction, mutation_rate, n_agents, max_agents, n_generations = restore_params()
    save_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\maze_solvers\\EVOLUTION"


    # Run experiments
    SF_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    EF_values = [0.0, 0.05, 0.1, 0.15,  0.2, 0.25, 0.3, 0.35, 0.4]
    repeats = 3

    n_exps = len(SF_values) * len(EF_values)
    curexp = 1

    print("starting experiment")
    ch = MplColorHelper("Greens", 0, 1)
    f, ax = create_figure(subplots=False, facecolor=white, figsize=(16, 16))

    com_results = {k:[] for k in list(get_params("m1").keys())}
    com_results['pR'] = []
    pR_values = []
    for SF in SF_values: # varying stochasticity pressure
        for EF in EF_values: # varying efficiency pressure
            mean_pR = 0
            print("Running {}/{}".format(curexp, n_exps))
            curexp += 1

            # run scenes in parallel
            scenes = [Scene(**get_params("m1", predator_risk_factor=SF, risk=EF)) for i in range(repeats)]
            results = run_scenes_in_parallel(scenes)
            
            # store results
            mean_pR = np.mean([np.mean(traces.predator_bias.values[-100:]) for traces, params in results])
            pR_values.append((SF, EF, mean_pR))
            ax.scatter(SF, EF, color=ch.get_rgb(mean_pR), s=1000*mean_pR, label="SF{},EF{},p(R):{}".format(SF, EF, round(mean_pR, 2)))
            
            # save to file
            params = results[0][1]
            params['pR'] = mean_pR
            params = {k:[v] for k, v in params.items()}
            pd.DataFrame.from_dict(params).to_pickle(os.path.join(save_fld, "experiments", "SF_{}_EF_{}.pkl".format(SF,EF)))

            # Store all results
            for k,v in params.items():
                com_results[k].append(v[0])
    pd.DataFrame.from_dict(com_results).to_pickle(os.path.join(save_fld, "comres_.pkl".format(SF,EF)))

    
    # interpolate and plot
    ax.set(title="varying both pressures in M1", xlabel="stochasticity pressure", ylabel="efficiency pressure", facecolor=[.2, .2, .2],)
    ax.legend()

    # organize Data...
    x = np.array([p[0] for p in pR_values])
    y = np.array([p[1] for p in pR_values])
    z = np.array([p[2] for p in pR_values])

    # Fit a 3rd order, 2d polynomial
    m = polyfit2d(x,y,z)

    # Evaluate it on a grid...
    nx, ny = 20, 20
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx), 
                            np.linspace(y.min(), y.max(), ny))
    zz = polyval2d(xx, yy, m)

    # Plot
    f,ax = create_figure(subplots=False, facecolor=white, figsize=(16, 16))
    ax.imshow(zz, extent=(0, x.max(), 0, y.max()), cmap="Greens")
    ax.scatter(x, y, c=z, cmap="Greens", lw=1, edgecolor=black)

    CS = ax.contour(xx, yy, zz)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set(xlabel="stochasticity pressure", ylabel="efficiency pressure")


    plt.show()