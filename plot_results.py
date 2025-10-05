import functions as fct
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns

# Functions to plot results
#===========================================================================================================

# This is the simple plot function
def plot_simple_fit(covariate, 
                    response, 
                    fixed_bandwith, 
                    sigma_2):
    
    plt.figure(figsize=(20, 6))
    n_smp = len(covariate)
    beta_fct_opti = fct.estimate_parameters(covariate,                      # Estimate parameters for fixed
                                            response,
                                            fixed_bandwith, 
                                            p=2)
    p = np.linspace(0, 1, 100)                                              # To plot
    beta_values_fixed = beta_fct_opti(p)
    plt.scatter(covariate,                                                  # Scatter plot              
                response, 
                marker='o', 
                facecolors='none', 
                edgecolors='lightblue', 
                alpha=0.5, 
                label="Samples")
    
    h_values = np.arange(0.01, 0.99, step=0.08)                             # Values for bandwith to test
    norm = plt.Normalize(vmin=h_values.min(), vmax=h_values.max())          # For nice fade
    cmap = plt.cm.viridis
    
    for h in h_values:
        beta_fct_default = fct.estimate_parameters(covariate,               # Estimate for values of h
                                                   response, 
                                                   h, 
                                                   p=2)
        beta_values_default = beta_fct_default(p)
        plt.plot(p, beta_values_default[:, 0],                              # Plot for this h
                 color=cmap(norm(h)), 
                 alpha=0.6)
    
    plt.plot(p, beta_values_fixed[:, 0],                                    # Plot the fixed one
             color="red", 
             label=rf"fit with $h_{{bw}}={fixed_bandwith}$")

    m = lambda x: np.sin(1 / (x / 3 + 0.1))                                 # Truth m
    plt.plot(p, m(p), label=r"$m$ (truth)", color="darkorange")             # Plot the truth

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)                        # For the colorbar
    plt.colorbar(sm, ax=plt.gca(), label='bandwidth h')
    plt.xlabel("x / covariate")
    plt.ylabel("fit / response")
    plt.legend()
    plt.title(
        rf"Regression fits of {n_smp} samples of $m(X)+\epsilon$, "
        rf"$X \sim \mathrm{{Beta}}(\alpha={alpha}, \beta={beta})$, "
        rf"$\sigma^2(\epsilon)={sigma_2}$")

    plt.savefig('simple_plot_bandwith.png')                                 # Save name




# This is for the impact of alpha and beta
def plot_alpha_beta_impact(alphas, 
                           betas, 
                           error_variance, 
                           default_bandwith, 
                           number_of_blocks):
    
    res_h_IMSE = np.zeros((len(alphas), len(betas)))                        # For storing values
    res_sigma_2 = np.zeros((len(alphas), len(betas)))                       # Same
    res_theta_22 = np.zeros((len(alphas), len(betas)))                      # Same

    total_iterations = len(alphas) * len(betas)                             # To get live console feedback
    current_iteration = 0
    for (a_idx, a) in enumerate(alphas):                                    
        for (b_idx, b) in enumerate(betas):
            current_iteration += 1
            res_h_IMSE[a_idx, b_idx], res_sigma_2[a_idx, b_idx], res_theta_22[a_idx, b_idx] =\
                fct.simulate(alpha = a,                                     # Simulate - Store results
                        beta=b,
                        error_variance=error_variance,
                        number_of_samples=number_of_samples,
                        default_bandwith=default_bandwith,
                        number_of_blocks=number_of_blocks)
            print(f"Iteration {current_iteration} / {total_iterations} | Computed results for (alpha, beta)=({a}, {b})")
            print(f"(h_IMSE, sigma^2, theta_22)=({res_h_IMSE[a_idx, b_idx]}, {res_sigma_2[a_idx, b_idx]}, {res_theta_22[a_idx, b_idx]})")

    _, axes = plt.subplots(1, 3, figsize=(20, 6))                           # Plot environement

    
    sns.heatmap(res_h_IMSE,                                                 # Seaborn heatmap of h_IMSE    
                ax=axes[0], 
                cmap='viridis', 
                xticklabels=np.round(betas, 1), 
                yticklabels=np.round(alphas, 1),
                cbar_kws={'label': r"$h_{IMSE}$"})
    axes[0].set_xlabel(r"$\beta$")
    axes[0].set_ylabel(r"$\alpha$")
    axes[0].set_title(r'$h_{IMSE}$ Results')

  
    sns.heatmap(res_sigma_2,                                                # Seaborn heatmap of sigma^2
                ax=axes[1], 
                cmap='viridis',
                xticklabels=np.round(betas, 1), 
                yticklabels=np.round(alphas, 1),
                cbar_kws={'label': r"$\hat\sigma^2$"})
    axes[1].set_xlabel(r"$\beta$")
    axes[1].set_ylabel(r"$\alpha$")
    axes[1].set_title(r"$\hat\sigma^{2}$ Results")

    
    sns.heatmap(res_theta_22,                                               # Seaborn heatmap of theta_22
                ax=axes[2], 
                cmap='viridis',
                xticklabels=np.round(betas, 1), 
                yticklabels=np.round(alphas, 1),
                cbar_kws={'label': r"$\hat\theta_{22}$"})
    axes[2].set_xlabel(r"$\beta$")
    axes[2].set_ylabel(r"$\alpha$")
    axes[2].set_title(r"$\hat\theta_{22}$ Results")

    plt.suptitle(rf"Results in terms of $\alpha$ and $\beta$. $(\sigma^2, n, N, bw)=$({error_variance}, {number_of_samples}, {number_of_blocks}, {default_bandwith})")
    plt.tight_layout()

    plt.savefig('alpha_beta_impact_colormap.png')                           # Save name
    






# To plot the impact of the sample sizes and blocks   
def plot_sample_blocks_impact(number_of_samples_range, 
                              number_of_blocks, 
                              alpha, 
                              beta, 
                              error_variance, 
                              default_bandwith):
    
    res_h_IMSE = np.zeros((len(number_of_samples_range), len(number_of_blocks)))    # To store results
    res_sigma_2 = np.zeros((len(number_of_samples_range), len(number_of_blocks)))   # Same
    res_theta_22 = np.zeros((len(number_of_samples_range), len(number_of_blocks)))  # Same

    total_iterations = len(number_of_samples_range) * len(number_of_blocks)         # For console feedback
    current_iteration = 0
    for (smp_idx, n_smp) in enumerate(number_of_samples_range):
        for (blo_idx, n_blo) in enumerate(number_of_blocks):
            current_iteration += 1
            res_h_IMSE[smp_idx, blo_idx], res_sigma_2[smp_idx, blo_idx], res_theta_22[smp_idx, blo_idx] =\
                fct.simulate(alpha=alpha,                                           # Simulate + store results
                            beta=beta,
                            error_variance=error_variance,
                            number_of_samples=n_smp,
                            default_bandwith=default_bandwith,
                            number_of_blocks=n_blo)
            print(f"Iteration {current_iteration} / {total_iterations} | Computed results for (n_smp, n_blo)=({n_smp}, {n_blo})")


    _, axes = plt.subplots(1, 3, figsize=(20, 6))                                   # Plot env.

    sns.heatmap(res_h_IMSE,                                                         # Seaborn heatmap for h_IMSE
                ax=axes[0], 
                cmap='viridis',
                cbar_kws={'label': r"$h_{IMSE}$"})
    axes[0].set_xlabel('Number of Blocks')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_title(r'$h_{IMSE}$ Results')
    axes[0].set_xticks(np.arange(len(number_of_blocks)) + 0.5)                      # For appropriate indexing
    axes[0].set_xticklabels(number_of_blocks)
    axes[0].set_yticks(np.arange(0, len(number_of_samples_range), 10) + 0.5)
    axes[0].set_yticklabels(number_of_samples_range[::10])

    sns.heatmap(res_sigma_2,                                                        # Seaborn heatmap for sigma^2
                ax=axes[1], 
                cmap='viridis',
                cbar_kws={'label': r"$\hat\sigma^2$"})
    axes[1].set_xlabel('Number of Blocks')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title(r"$\hat\sigma^{2}$ Results")
    axes[1].set_xticks(np.arange(len(number_of_blocks)) + 0.5)
    axes[1].set_xticklabels(number_of_blocks)
    axes[1].set_yticks(np.arange(0, len(number_of_samples_range), 10) + 0.5)
    axes[1].set_yticklabels(number_of_samples_range[::10])

    sns.heatmap(res_theta_22,                                                       # Seabron heatmap for theta_22
                ax=axes[2], 
                cmap='viridis',
                cbar_kws={'label': r"$\hat\theta_{22}$"})
    axes[2].set_xlabel('Number of Blocks')
    axes[2].set_ylabel('Number of Samples')
    axes[2].set_title(r"$\hat\theta_{22}$ Results")
    axes[2].set_xticks(np.arange(len(number_of_blocks)) + 0.5)
    axes[2].set_xticklabels(number_of_blocks)
    axes[2].set_yticks(np.arange(0, len(number_of_samples_range), 10) + 0.5)
    axes[2].set_yticklabels(number_of_samples_range[::10])

    plt.suptitle(rf"Results in terms of number of samples and blocks. $(\alpha, \beta, \sigma^2, bw)=$({alpha}, {beta}, {error_variance}, {default_bandwith})")
    plt.tight_layout()
    
    plt.savefig('sample_size_block_number_impact_colormap.png')                     # Save name





# For plot with Mallow
def plot_sample_size_impact_mallow(alpha, 
                                   beta, 
                                   max_number_of_samples, 
                                   step, error_variance, 
                                   default_bandwith, 
                                   max_number_of_blocks=10):
    default_start = 10                                                              # Quick check
    if max_number_of_samples < default_start + step:
        raise ValueError("Not enough samples")
    
    n_range = np.arange(start=default_start, stop=max_number_of_samples, step=step)

    h_IMSE_list = []                                                                # To store
    sigma_2_list = []                                                               # Same
    theta_22_list = []                                                              # Same

    for (idx, n) in enumerate(n_range):                                             # Iterate on n
        covariate, response = fct.generate_sample(alpha=alpha,                      # Generate samples
                                        beta=beta,
                                        n_samples = n,
                                        sigma_2=error_variance)
            
        h_IMSE, sigma_2, theta_22 = fct.h_IMSE_Cp_optimized(covariate=covariate,    # Estimate
                                                    response=response,
                                                    number_of_samples=n,
                                                    max_number_of_blocks=max_number_of_blocks, 
                                                    default_bandwith=default_bandwith)
        
        h_IMSE_list.append(h_IMSE)                                                  # Store results
        sigma_2_list.append(sigma_2)                                                # Same
        theta_22_list.append(theta_22)                                              # Same

        print(f"Iteration {idx} / {len(n_range)}: (h_IMSE, sigma^2, theta_22) = ({h_IMSE},{sigma_2},{theta_22})")

    _, ax = plt.subplots(3, 1, figsize=(20, 12))                                    # Plot env.
 
    ax[0].plot(n_range, h_IMSE_list, marker='o', color="red", label=r"$h_{\mathrm{IMSE}}$")         # Plot for h_IMSE
    ax[0].set_title(r"$h_{\mathrm{IMSE}}$ vs. Sample Size")
    ax[0].set_ylabel(r"$h_{\mathrm{IMSE}}$")
    ax[0].legend()

    ax[1].plot(n_range, sigma_2_list, marker='o', color="blue", label=r"$\sigma^2$")                # Plot for sigma^2
    ax[1].set_title(rf"$\sigma^2$ vs. Sample Size, truth is $\sigma^2={error_variance}$")
    ax[1].set_ylabel(r"$\hat\sigma^2$")
    ax[1].legend()

    ax[2].plot(n_range, theta_22_list, marker='o', color="green", label=r"$\theta_{22}$")           # Plot for theta_22
    ax[2].set_title(r"$\hat\theta_{22}$ vs. Sample Size")
    ax[2].set_xlabel("Sample Size")
    ax[2].set_ylabel(r"$\hat\theta_{22}$")
    ax[2].legend()

    plt.suptitle(rf"Estimations against sample size (Mallow-selected block number), $(\alpha, \beta, \sigma^2, fixed bw)=$({alpha}, {beta}, {error_variance}, {default_bandwith})")
    plt.tight_layout()

    plt.savefig('plot_sample_size_mallow_10_10.png')                                      # Save name



#=====================================================================================================================
# Console handeling

if __name__=="__main__":
    ui = sys.argv   # User input
    menu = ui[1]    # First input corresponds to plot type

    print("Hi!")    # Just saying hi :)

    if menu == "simple_plot":                                                               # For simple plot
        print("TASK: produce simple fit plot on some data")
        # Below: get inputs
        alpha, beta, number_of_samples, fixed_bandwith, variance = float(ui[2]), float(ui[3]), int(ui[4]), float(ui[5]), float(ui[6])
        covariate, response = fct.generate_sample(alpha=alpha,                              # Generate sample
                                                  beta=beta,
                                                  n_samples=number_of_samples,
                                                  sigma_2=variance)
 
        plot_simple_fit(covariate=covariate,                                                # Call plot
                        response=response,
                        fixed_bandwith=fixed_bandwith, sigma_2=variance)
        print("TASK completed")
    elif menu == "alpha_beta_impact":                                                       # For alpha, beta impact
        print("TASK: produce impact of alpha/beta on h_IMSE, est.variance, est.theta22")
        # Below: get inputs
        number_of_samples, error_variance, default_bandwith, number_of_blocks = int(ui[2]), float(ui[3]), float(ui[4]), int(ui[5])
        # Below: ranges of alphas and betas
        alphas = np.arange(0.1, 10, step = 0.5) # alphas = np.arange(0.01, 5, step = 0.5), alphas = np.arange(0.01, 1, step = 0.01)
        betas = np.arange(0.1, 10, step = 0.5) # betas = np.arange(0.01, 5, step = 0.5),  betas = np.arange(0.01, 1, step = 0.01))

        plot_alpha_beta_impact(alphas=alphas,                                               # Call plot function
                               betas=betas, 
                               error_variance=error_variance, 
                               default_bandwith=default_bandwith, 
                               number_of_blocks=number_of_blocks)
        print("TASK completed")
    elif menu == "sample_blocks_impact":                                                    # For n/N impacts
        print("TASK: produce impact of sample size and block  on h_IMSE, est.variance, est.theta22")
        # Below: get ranges
        number_of_samples_range = np.arange(start=100, stop=5000 + 1, step = 50)
        number_of_blocks = np.arange(start = 2, stop = 10 + 1, step = 1)
        # Below: get inputs
        alpha, beta, error_variance, default_bandwith = float(ui[2]), float(ui[3]), float(ui[4]), float(ui[5])

        plot_sample_blocks_impact(number_of_samples_range,                                  # Call plot function
                                  number_of_blocks, 
                                  alpha, 
                                  beta, 
                                  error_variance, 
                                  default_bandwith)
        print("TASK completed")
    elif menu == "sample_size_mallow_evolution":                                            # For mallow plot
        # Below: get inputs
        alpha, beta, max_number_of_samples, step, error_variance, default_bandwith = float(ui[2]), float(ui[3]), int(ui[4]), int(ui[5]), float(ui[6]), float(ui[7])
        
        print("TASK: produce impact of sample size on h_IMSE, est.variance, est.theta22 with Mallow selection on blocks")
        
        plot_sample_size_impact_mallow(alpha,                                               # Call plotting function
                                       beta, 
                                       max_number_of_samples, 
                                       step, 
                                       error_variance, 
                                       default_bandwith)
        print("TASK completed")

