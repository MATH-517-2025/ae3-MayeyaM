import numpy as np

# Technical functions (not plot-related)
#=========================================================================================================

# Generate a sample
def generate_sample(alpha, 
                    beta, 
                    n_samples, 
                    sigma_2=1):
    
    m = lambda x: np.sin(1 / (x / 3 + 0.1))                                                 # Truth

    X = np.random.beta(a=alpha, b=beta, size=n_samples)                                     # Generate X / covariate
    epsilon = np.random.normal(loc=0, scale=np.sqrt(sigma_2), size=n_samples)               # Get additional errors
    Y = m(X) + epsilon                                                                      # Get Y

    response = Y                                                                            # For clarity 
    covariate = X                                                                           # Same
    return covariate, response



# To produce estimating function (maybe the name is not the best)
def estimate_parameters(covariate, 
                        response, 
                        bandwith, 
                        p=1, 
                        reg= 1e-5):
    X = covariate
    Y = response
    h = bandwith
    quartic_kernel = lambda x: (np.abs(x) <= 1) * (15 / 16) * (1 - x**2)**2                 # Define the Kernel
    
    def beta_est(x_array):                                                                  # The return function
        n = len(x_array)
        n_params = p + 1
        beta_x = np.empty((n, n_params))                                                    # Will store here
        
        for idx, x in enumerate(x_array):                                                   # Iterate through Xs
            weights = quartic_kernel((X - x) / h)                                           
            sqrt_W = np.sqrt(weights)                               

            X_powers = np.column_stack([(X - x)**p_ for p_ in range(p + 1)])                # 'X' matrix
            X_tmp = sqrt_W[:, None] * X_powers                                         
            Y_tmp = sqrt_W * Y
            
            XtWX = X_tmp.T @ X_tmp
            XtWX[np.diag_indices_from(XtWX)] += reg                                         # Add regularization
            
            beta_x[idx] = np.linalg.solve(XtWX, X_tmp.T @ Y_tmp)                            # Cholesky approach - solve
        
        return beta_x
   
    return beta_est



# To estimate sigma^2 and theta_22
def estimate_sigma_theta(covariate, 
                         response, 
                         bandwith=0.1, 
                         N_blocks=10):

    size = len(covariate)
    block_size = size // N_blocks
    sorted_indices = np.argsort(covariate)  
    X_sorted = covariate[sorted_indices]                                                    # For blocks
    Y_sorted = response[sorted_indices]                                                     # Same
    
    covariate_blocks = []                                                                   # To store
    response_blocks = []                                                                    # Same
    for i in range(N_blocks):
        start = i * block_size
        end = (i + 1) * block_size if i < N_blocks - 1 else size                            # Handles additional points
        covariate_blocks.append(X_sorted[start:end])                                        # Store
        response_blocks.append(Y_sorted[start:end])                                         # Same
    
    sigma_2_hat, theta_22 = 0, 0                                            

    for j in range(N_blocks):                                                               # For theta_22
        beta_est_j = estimate_parameters(covariate_blocks[j],                               # On jth block
                                         response_blocks[j], 
                                         bandwith, 
                                         p=4)
        X_j = covariate_blocks[j]                                                           # Evaluate
        beta_values = beta_est_j(X_j) 
        
        m_dd_vals = 2 * beta_values[:, 2]                                                   # Derivative
        theta_22 += np.sum(m_dd_vals**2)
    
    theta_22 /= size
    
    for j in range(N_blocks):                                                               # For sigma^2
        beta_est_j = estimate_parameters(covariate_blocks[j],                               # Similar to before
                                         response_blocks[j], 
                                         bandwith, 
                                         p=4)
        X_j = covariate_blocks[j]
        Y_j = response_blocks[j]
        
        m_hat = beta_est_j(X_j)[:, 0]
        residuals = Y_j - m_hat                                                             # Residuals
        sigma_2_hat += np.sum(residuals**2)                                                 # MSE
    
    sigma_2_hat /= (size - 5 * N_blocks)                                                    # following the README

    return sigma_2_hat, theta_22




# To compute h_IMSE with adaptive support consideration
def est_h_IMSE_support(sigma_2_hat, 
                       theta_22, 
                       size, 
                       covariate):
    support = max(covariate) - min(covariate)                                               # Compute sampled support
    h_IMSE = size**(-1/5) * ((35 * sigma_2_hat * support) / theta_22)**(1/5)                # Compute h_ISME
    
    return h_IMSE




# To compute Mallow Cp statistic
def compute_mallow_C_p(covariate, 
                       response, 
                       fixed_bandwith, 
                       N_blocks):
    size = len(covariate)
    block_size = size // N_blocks

    N_max = int(max(min(np.floor(size / 20), 5), 1))                                        # Following the README
    block_size_N_max = size // N_max
    
    sorted_indices = np.argsort(covariate)                                                  # As already seen
    X_sorted = covariate[sorted_indices]    
    Y_sorted = response[sorted_indices]

    covariate_blocks = []                                                                   # Storage 1
    response_blocks = []
    rss_N = 0

    covariate_blocks_N_max = []                                                             # Storage 2
    response_blocks_N_max = []
    rss_N_max = 0

    for i in range(N_blocks):                                                               # RSS 1
        start = i * block_size
        end = (i + 1) * block_size if i < N_blocks - 1 else size
        covariate_blocks.append(X_sorted[start:end])
        response_blocks.append(Y_sorted[start:end])

    for i in range(N_max):                                                                  # RSS 2
        start = i * block_size_N_max
        end = (i + 1) * block_size_N_max if i < N_max - 1 else size
        covariate_blocks_N_max.append(X_sorted[start:end])
        response_blocks_N_max.append(Y_sorted[start:end])

    for j in range(N_blocks):
        beta_est_j = estimate_parameters(covariate_blocks[j], 
                                         response_blocks[j], 
                                         fixed_bandwith, 
                                         p=4)

        X_j = covariate_blocks[j]
        Y_j = response_blocks[j]
        m_hat = beta_est_j(X_j)[:, 0]
        
        err = Y_j - m_hat
        rss_N += np.sum(err**2)                                                             # RSS 1
    
    for j in range(N_max):
        beta_est_j = estimate_parameters(covariate_blocks_N_max[j],                         # Similar
                                         response_blocks_N_max[j], 
                                         fixed_bandwith, 
                                         p=4)

        X_j = covariate_blocks_N_max[j]
        Y_j = response_blocks_N_max[j]
        m_hat = beta_est_j(X_j)[:, 0]
        
        err_N_max = Y_j - m_hat
        rss_N_max += np.sum(err_N_max**2)                                                    # RSS 2

    C_p = rss_N / (rss_N_max / (size - 5 * N_max)) - (size - 10*N_blocks)                    # Compute Mallow      
    
    return C_p 



# Get optimal number with Mallow criteria
def optimal_mallow(covariate, 
                   response, 
                   fixed_bandwith, 
                   max_N_blocks):
    C_p_list = np.zeros(max_N_blocks)
    for N in range(1, max_N_blocks + 1):
        C_p_list[N - 1] = compute_mallow_C_p(covariate=covariate, 
                                         response=response,
                                         fixed_bandwith=fixed_bandwith,
                                         N_blocks=N)
    
    minimising_Cp_N_blocks = np.argmin(C_p_list) + 1                                        # Get minimizing number

    return minimising_Cp_N_blocks                                                           # Return minimizing one



# Get values of estimates with Cp-optimized block size
def h_IMSE_Cp_optimized(covariate, 
                        response, 
                        number_of_samples, 
                        default_bandwith, 
                        max_number_of_blocks):
    
    minimising_Cp_N_blocks = optimal_mallow(covariate,                                      # Find minimizing
                                            response, 
                                            default_bandwith, 
                                            max_number_of_blocks)
    
    sigma_2_est, theta_22_est = estimate_sigma_theta(covariate=covariate,                   # Estimae with minimizing
                                                    response=response,
                                                    bandwith=default_bandwith,
                                                    N_blocks=minimising_Cp_N_blocks)
    
    h_IMSE_est = est_h_IMSE_support(sigma_2_est,                                            # Get h_IMSE
                                    theta_22_est, 
                                    size=number_of_samples, 
                                    covariate=covariate)

    return h_IMSE_est, sigma_2_est, theta_22_est                                            # Return optimized



# To simualte an estimation
def simulate(alpha, 
             beta, 
             number_of_samples, 
             error_variance, 
             default_bandwith, 
             number_of_blocks):
    
    covariate, response = generate_sample(alpha=alpha,                                      # Generate sample
                                          beta=beta,
                                          n_samples = number_of_samples,
                                          sigma_2=error_variance)
    
    sigma_2_est, theta_22_est = estimate_sigma_theta(covariate=covariate,                   # Perform estimations
                                                    response=response,
                                                    bandwith=default_bandwith,
                                                    N_blocks=number_of_blocks)

    h_IMSE_est = est_h_IMSE_support(sigma_2_est,                                            # Get h_IMSE
                                    theta_22_est, 
                                    size=number_of_samples, 
                                    covariate=covariate)

    return h_IMSE_est, sigma_2_est, theta_22_est