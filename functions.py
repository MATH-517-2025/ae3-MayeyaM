import numpy as np


#######################################
# NAME PROBLEM: N_blocks SHOULD BE CHANGED TO NUMBER OF BLOCKS
#######################################
def generate_sample(alpha, beta, n_samples, sigma_2=1):
    m = lambda x: np.sin(1 / (x / 3 + 0.1))

    X = np.random.beta(a=alpha, b=beta, size=n_samples)
    epsilon = np.random.normal(loc=0, scale=np.sqrt(sigma_2), size=n_samples)
    Y = m(X) + epsilon

    response = Y
    covariate = X
    return covariate, response

def estimate_parameters(covariate, response, bandwith, p=1):
    X = covariate
    Y = response
    h = bandwith
    quartic_kernel = lambda x: (np.abs(x) <= 1) * (15 / 16) * (1 - x**2)**2
    
    def beta_est(x_array):
        n = len(x_array)
        n_params = p + 1
        beta_x = np.empty((n, n_params))
        reg = 1e-5
        
        # Precompute powers of X if p > 0
        #X_powers = np.column_stack([X**p_ for p_ in range(p + 1)])
        
        for idx, x in enumerate(x_array):
            # Vectorized weight computation
            weights = quartic_kernel((X - x) / h)
            sqrt_W = np.sqrt(weights)
            
            # Weighted design matrix: sqrt(W) * X_matrix
            X_diff_powers = np.column_stack([(X - x)**p_ for p_ in range(p + 1)])
            X_weighted = sqrt_W[:, None] * X_diff_powers
            Y_weighted = sqrt_W * Y
            
            # Solve via Cholesky (faster than full inverse)
            XtWX = X_weighted.T @ X_weighted
            XtWX[np.diag_indices_from(XtWX)] += reg
            
            beta_x[idx] = np.linalg.solve(XtWX, X_weighted.T @ Y_weighted)
        
        return beta_x
   
    return beta_est

def estimate_sigma_theta(covariate, response, bandwith=0.1, N_blocks=10):
    """ # Build the blocks
    size = len(covariate)
    block_size = size // N_blocks
    sorted_indices = np.argsort(covariate)  
    X_sorted = covariate[sorted_indices]    
    Y_sorted = response[sorted_indices]
    
    covariate_blocks = []
    response_blocks = []
    for i in range(N_blocks):
        start = i * block_size
        end = (i + 1) * block_size if i < N_blocks - 1 else size
        covariate_blocks.append(X_sorted[start:end])
        response_blocks.append(Y_sorted[start:end])
    
    # Estimate for each block
    m_j_list = []
    for j in range(len(covariate_blocks)):
        beta_est = estimate_parameters(covariate_blocks[j], response_blocks[j], bandwith, p=4)
        m_j_evaluated_at_xi = beta_est(covariate)
        m_j_list.append(m_j_evaluated_at_xi)
    
    sigma_2_hat, theta_22 = 0, 0
    
    for j in range(len(covariate_blocks)):
        X_block = covariate_blocks[j]
        Y_block = response_blocks[j]
        
        for idx, xi in enumerate(X_block):
            yi = Y_block[idx]
            
            # Use the original index mapping instead of searching
            original_idx = j * block_size + idx
            if original_idx >= len(covariate):
                original_idx = len(covariate) - 1
            
            beta_row = m_j_list[j][original_idx]
            
            m_val = beta_row[0]
            m_dd_val = 2 * beta_row[2]  # second derivative
            
            # Standard formula: theta_22 is integrated (m'')^2
            theta_22 += m_dd_val**2
            sigma_2_hat += (yi - m_val)**2
    
    # Normalize (only once!)
    theta_22 /= size
    sigma_2_hat /= (size - 5 * N_blocks)

    return sigma_2_hat, theta_22
     """

    size = len(covariate)
    block_size = size // N_blocks
    sorted_indices = np.argsort(covariate)  
    X_sorted = covariate[sorted_indices]    
    Y_sorted = response[sorted_indices]
    
    covariate_blocks = []
    response_blocks = []
    for i in range(N_blocks):
        start = i * block_size
        end = (i + 1) * block_size if i < N_blocks - 1 else size
        covariate_blocks.append(X_sorted[start:end])
        response_blocks.append(Y_sorted[start:end])
    
    sigma_2_hat, theta_22 = 0, 0
    
    # Estimate theta_22
    for j in range(N_blocks):
        # Fit model on block j
        beta_est_j = estimate_parameters(covariate_blocks[j], response_blocks[j], bandwith, p=4)
        
        # Evaluate at points in block j
        X_j = covariate_blocks[j]
        beta_values = beta_est_j(X_j)  # Shape: (n_j, 5)
        
        # Second derivative at each point (evaluated at the point itself)
        # m''(x) = 2*β₂(x)  when evaluated at the center of local fit
        # But we fit at multiple points, so β varies
        m_dd_vals = 2 * beta_values[:, 2]
        theta_22 += np.sum(m_dd_vals**2)
    
    theta_22 /= size
    
    # Estimate sigma^2
    for j in range(N_blocks):
        beta_est_j = estimate_parameters(covariate_blocks[j], response_blocks[j], bandwith, p=4)
        X_j = covariate_blocks[j]
        Y_j = response_blocks[j]
        
        # Predict at points in block j
        m_hat = beta_est_j(X_j)[:, 0]
        residuals = Y_j - m_hat
        sigma_2_hat += np.sum(residuals**2)
    
    sigma_2_hat /= (size - 5 * N_blocks)

    return sigma_2_hat, theta_22

""" def est_h_IMSE(sigma_2_hat, theta_22, size):
    # Note that support is [0, 1] (beta distribution)
    #===================================================================== CHANGE SUPPORT BY GIVING COVARIATE AS INPUT CHAMP ====
    h_IMSE = size**(-1/5) * ((35 * sigma_2_hat * 1) / theta_22)**(1/5)
    
    return h_IMSE """

def est_h_IMSE_support(sigma_2_hat, theta_22, size, covariate):
    # Note that support is [0, 1] (beta distribution)
    #===================================================================== CHANGE SUPPORT BY GIVING COVARIATE AS INPUT CHAMP ====
    support = max(covariate) - min(covariate) 
    # print(support)
    h_IMSE = size**(-1/5) * ((35 * sigma_2_hat * support) / theta_22)**(1/5)
    
    return h_IMSE

def compute_mallow_C_p(covariate, response, fixed_bandwith, N_blocks):
    size = len(covariate)
    block_size = size // N_blocks

    N_max = int(max(min(np.floor(size / 20), 5), 1))
    block_size_N_max = size // N_max
    
    sorted_indices = np.argsort(covariate)  
    X_sorted = covariate[sorted_indices]    
    Y_sorted = response[sorted_indices]

    covariate_blocks = []
    response_blocks = []
    rss_N = 0

    covariate_blocks_N_max = []
    response_blocks_N_max = []
    rss_N_max = 0

    for i in range(N_blocks):
        start = i * block_size
        end = (i + 1) * block_size if i < N_blocks - 1 else size
        covariate_blocks.append(X_sorted[start:end])
        response_blocks.append(Y_sorted[start:end])

    for i in range(N_max):
        start = i * block_size_N_max
        end = (i + 1) * block_size_N_max if i < N_max - 1 else size
        covariate_blocks_N_max.append(X_sorted[start:end])
        response_blocks_N_max.append(Y_sorted[start:end])

    for j in range(N_blocks):
        # Fit on block j with degree p=4
        beta_est_j = estimate_parameters(covariate_blocks[j], response_blocks[j], fixed_bandwith, p=4)

        X_j = covariate_blocks[j]
        Y_j = response_blocks[j]
        m_hat = beta_est_j(X_j)[:, 0]
        
        # Accumulate squared residuals
        err = Y_j - m_hat
        rss_N += np.sum(err**2)
    
    for j in range(N_max):
        # Fit on block j with degree p=4
        beta_est_j = estimate_parameters(covariate_blocks_N_max[j], response_blocks_N_max[j], fixed_bandwith, p=4)

        X_j = covariate_blocks_N_max[j]
        Y_j = response_blocks_N_max[j]
        m_hat = beta_est_j(X_j)[:, 0]
        
        # Accumulate squared residuals
        err_N_max = Y_j - m_hat
        rss_N_max += np.sum(err_N_max**2)

    C_p = rss_N / (rss_N_max / (size - 5 * N_max)) - (size - 10*N_blocks)
    
    return C_p 

def optimal_mallow(covariate, response, fixed_bandwith, max_N_blocks):
    C_p_list = np.zeros(max_N_blocks)
    for N in range(1, max_N_blocks + 1):
        C_p_list[N - 1] = compute_mallow_C_p(covariate=covariate, 
                                         response=response,
                                         fixed_bandwith=fixed_bandwith,
                                         N_blocks=N)
    
    minimising_Cp_N_blocks = np.argmin(C_p_list) + 1

    return minimising_Cp_N_blocks

def h_IMSE_Cp_optimized(covariate, response, number_of_samples, error_variance, default_bandwith, max_number_of_blocks):
    
    minimising_Cp_N_blocks = optimal_mallow(covariate, response, default_bandwith, max_number_of_blocks)
    
    sigma_2_est, theta_22_est = estimate_sigma_theta(covariate=covariate,
                                                               response=response,
                                                               bandwith=default_bandwith,
                                                               N_blocks=minimising_Cp_N_blocks)
    
    #h_IMSE_est = est_h_IMSE(sigma_2_est, theta_22_est, size=number_of_samples)
    h_IMSE_est = est_h_IMSE_support(sigma_2_est, theta_22_est, size=number_of_samples, covariate=covariate)

    return h_IMSE_est, sigma_2_est, theta_22_est


def simulate(alpha, beta, number_of_samples, error_variance, default_bandwith, number_of_blocks):
    covariate, response = generate_sample(alpha=alpha,
                                          beta=beta,
                                          n_samples = number_of_samples,
                                          sigma_2=error_variance)
    
    sigma_2_est, theta_22_est = estimate_sigma_theta(covariate=covariate,
                                                               response=response,
                                                               bandwith=default_bandwith,
                                                               N_blocks=number_of_blocks)
    
    #h_IMSE_est = est_h_IMSE(sigma_2_est, theta_22_est, size=number_of_samples)
    h_IMSE_est = est_h_IMSE_support(sigma_2_est, theta_22_est, size=number_of_samples, covariate=covariate)

    return h_IMSE_est, sigma_2_est, theta_22_est