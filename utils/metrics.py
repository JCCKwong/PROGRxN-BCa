import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.metrics import concordance_index_censored
from lifelines import CRCSplineFitter
from lifelines.utils import CensoringType
from dcurves import dca

def c_index(X, Y, model):
    c_index_progrxn_main = round(concordance_index_censored(Y['Progression'], Y['Time'], model.predict(X))[0],2)
    c_index_progrxn_pred = model.predict(X)

    # 95% CI for metric, selected below
    n_bootstraps = 100
    rng_seed = 42
    bootstrapped_progrxn = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(X), len(X))
        if len(np.unique(Y)) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        bootstrapped_progrxn.append(concordance_index_censored(Y['Progression'][indices],
                                                               Y['Time'][indices], c_index_progrxn_pred[indices])[0])

    bootstrapped_progrxn_sorted = np.array(bootstrapped_progrxn)
    bootstrapped_progrxn_sorted.sort()

    # Computing the lower and upper bound of the 95% confidence interval
    progrxn_confidence_lower = round(bootstrapped_progrxn_sorted[int(0.025 * len(bootstrapped_progrxn_sorted))],2)
    progrxn_confidence_upper = round(bootstrapped_progrxn_sorted[int(0.975 * len(bootstrapped_progrxn_sorted))],2)

    return c_index_progrxn_main, progrxn_confidence_lower, progrxn_confidence_upper

def calibration_dca(X, Y, model):
    # Adapted from Lifelines source code (https://github.com/CamDavidsonPilon/lifelines)
    def ccl(p):
        return np.log(-np.log(1 - p))

    cal_dca_fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

    surv = model.predict_survival_function(X, return_array=False)
    predictions = pd.Series(1 - np.array([fn(5) for fn in surv]), name=5)
    model_5yr = pd.Series(1 - np.array([fn(5) for fn in surv]), name='model')

    # Create new dataset with the predictions
    predictions_df = pd.DataFrame({"ccl_at_5": ccl(predictions), "Time": Y['Time'], "Event": Y['Progression']})

    predictions_5yr = pd.DataFrame({'PROGRxN': model_5yr,
                                    'ttcancer': Y['Time'],  # for dcurves, Time has to be 'ttcancer'
                                    'Event': Y['Progression']
                                    })

    # Calculate net benefit using dca from dcurves package
    dca_results_5yr = dca(data=predictions_5yr,
                          outcome='Event',
                          modelnames=['PROGRxN'],
                          models_to_prob=None,
                          thresholds=np.arange(0, 0.50, 0.01),
                          time_to_outcome_col="ttcancer",
                          time=5,
                          nper=1000
                          )

    # Fit new dataset to flexible spline model
    n_knots = 3
    regressors = {"beta_": ["ccl_at_5"], "gamma0_": "1", "gamma1_": "1", "gamma2_": "1"}

    # This model is from examples/royson_crowther_clements_splines.py
    crc = CRCSplineFitter(n_baseline_knots=n_knots)  # , penalizer=0.00001) #0.000001
    crc.fit_right_censoring(predictions_df, "Time", "Event", regressors=regressors)

    # Predict new model at values 0 to 1, but remember to ccl it!
    x = np.linspace(np.clip(predictions.min() - 0.01, 0, 1), np.clip(predictions.max() + 0.01, 0, 1), 100)
    y = 1 - crc.predict_survival_function(pd.DataFrame({"ccl_at_5": ccl(x)}), times=5).T.squeeze()

    ax[0].plot(x, y, label="PROGRxN-BCa", color="blue", lw=3)
    ax[0].set_xlabel("Predicted probability of progression at 5 years (%)")
    ax[0].set_ylabel("Observed probability of progression at 5 years (%)")

    # Plot Perfect Calibration line
    ax[0].plot([0, 1], [0, 1], c="grey", ls="--", label="Perfect Calibration")
    ax[0].set_xlim([0, 0.5])
    ax[0].set_xticks(np.arange(0, 0.6, step=0.1))
    ax[0].set_xticklabels(np.arange(0, 60, step=10))
    ax[0].set_ylim([0, 0.5])
    ax[0].set_yticks(np.arange(0, 0.6, step=0.1))
    ax[0].set_yticklabels(np.arange(0, 60, step=10))
    ax[0].grid(which='major', axis='both', alpha=0.3)
    ax[0].legend(loc='upper left', fancybox=True, shadow=False, ncol=1)
    ax[0].set_title("Calibration")

    modelnames = ['PROGRxN', 'all', 'none']
    color_names = ['blue', 'black', 'grey']
    styles = ['-', '-', '--']
    widths = [3, 2, 2]
    for modelname, color_name, style, width in zip(modelnames, color_names, styles, widths):
        single_model_df = dca_results_5yr[dca_results_5yr["model"] == modelname]
        ax[1].plot(single_model_df["threshold"] * 100,
                single_model_df["net_benefit"],
                color=color_name,
                lw=width,
                linestyle=style
                )

    ax[1].set_ylim([-0.05, 0.20])
    ax[1].set_yticks(np.arange(-0.05, 0.21, step=0.05))
    ax[1].set_xticks(np.arange(0, 51, step=10))
    ax[1].set_xlim([0, 50])
    ax[1].grid(which='major', axis='both', alpha=.3)
    ax[1].set_xlabel("Threshold probability at 5 years (%)")
    ax[1].set_ylabel("Net benefit")
    ax[1].legend(['PROGRxN-BCa', 'Treat All', 'Treat None'], loc='upper right', fancybox=True, shadow=False, ncol=1)
    ax[1].set_title('Decision Curve Analysis')

    return cal_dca_fig