import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam, ClippedAdam
import gpytorch
import numpy as np

class CorrelationModel:
    def __init__(self, time_interval=7.5):
        self.time_interval = time_interval
        # Use GPyTorch kernel to ensure correct implementation
        self.kernel = gpytorch.kernels.RBFKernel()
    
    def get_theoretical_correlation(self, lengthscale):
        """Get theoretical correlation at the fixed time interval"""
        # Create dummy input points separated by time_interval
        x1 = torch.tensor([[0.0]])
        x2 = torch.tensor([[self.time_interval]])
        
        # Set kernel hyperparameters
        self.kernel.lengthscale = lengthscale
        
        # Compute kernel value (this is the correlation)
        with torch.no_grad():
            correlation = self.kernel(x1, x2).to_dense().squeeze((0,1))
        
        return correlation

def model(observed_correlations):
    """Pyro model for Type-II ML estimation"""
    
    # More informative priors based on the data and problem domain
    # For correlations ~0.6 at t=7.5, lengthscale should be around 7-10
    lengthscale = pyro.sample("lengthscale", 
                             dist.LogNormal(torch.tensor(2.0), torch.tensor(0.5)))  # mean ~7.4
    
    # Noise should be relatively small for correlation data
    noise_scale = pyro.sample("noise_scale", 
                             dist.LogNormal(torch.tensor(-1.5), torch.tensor(0.5)))  # mean ~0.22
    
    # Create correlation model
    corr_model = CorrelationModel()
    
    # Get theoretical correlation at time interval 7.5
    theoretical_corr = corr_model.get_theoretical_correlation(lengthscale)
    
    # Likelihood: observed correlations are noisy versions of theoretical correlation
    with pyro.plate("observations", len(observed_correlations)):
        pyro.sample("obs", 
                   dist.Normal(theoretical_corr, noise_scale), 
                   obs=observed_correlations)

def guide(observed_correlations):
    """Variational guide for SVI"""
    
    # Better initialization based on rough estimates from data
    # For mean correlation ~0.63 at t=7.5: lengthscale ≈ 7.8
    lengthscale_loc = pyro.param("lengthscale_loc", torch.tensor(2.0))  # log(~7.4)
    lengthscale_scale = pyro.param("lengthscale_scale", torch.tensor(0.1), 
                                  constraint=dist.constraints.positive)
    
    # Initialize noise scale reasonably
    noise_scale_loc = pyro.param("noise_scale_loc", torch.tensor(-1.5))  # log(~0.22)
    noise_scale_scale = pyro.param("noise_scale_scale", torch.tensor(0.1),
                                  constraint=dist.constraints.positive)
    
    # Sample from variational distributions
    pyro.sample("lengthscale", dist.LogNormal(lengthscale_loc, lengthscale_scale))
    pyro.sample("noise_scale", dist.LogNormal(noise_scale_loc, noise_scale_scale))

def fit_type_ii_ml(observed_correlations, num_iterations=3000):
    """Fit Type-II ML using SVI with improved stability"""
    
    # Convert to torch tensor
    if isinstance(observed_correlations, np.ndarray):
        observed_correlations = torch.tensor(observed_correlations, dtype=torch.float32)
    
    # Clear parameter store
    pyro.clear_param_store()

    # Use ClippedAdam with a fixed learning rate
    optimizer = ClippedAdam({"lr": 0.001, "clip_norm": 10.0})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    
    # Training loop with better monitoring
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    patience = 5000
    
    for i in range(num_iterations):
        
        loss = svi.step(observed_correlations)
        losses.append(loss)
        
        # Early stopping check
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if i % 200 == 0:
            current_lengthscale = pyro.param("lengthscale_loc").exp().item()
            current_noise = pyro.param("noise_scale_loc").exp().item()
            print(f'Iteration {i}, Loss: {loss:.4f}, LS: {current_lengthscale:.3f}, Noise: {current_noise:.3f}')
        
        # Early stopping
        if patience_counter > patience and i > 1000:
            print(f"Early stopping at iteration {i}")
            break
    
    # Extract fitted parameters with uncertainty
    lengthscale_posterior = dist.LogNormal(
        pyro.param("lengthscale_loc"), 
        pyro.param("lengthscale_scale")
    )
    noise_scale_posterior = dist.LogNormal(
        pyro.param("noise_scale_loc"), 
        pyro.param("noise_scale_scale")
    )
    
    # Get point estimates (posterior means)
    lengthscale_est = lengthscale_posterior.mean
    noise_scale_est = noise_scale_posterior.mean
    
    return {
        'lengthscale': lengthscale_est.item(),
        'noise_scale': noise_scale_est.item(),
        'lengthscale_posterior': lengthscale_posterior,
        'noise_scale_posterior': noise_scale_posterior,
        'losses': losses,
        'converged': patience_counter <= patience
    }

def analyze_results(observed_correlations, results):
    """Analyze and validate the fitted results"""
    
    corr_model = CorrelationModel()
    theoretical_corr = corr_model.get_theoretical_correlation(
        torch.tensor(results['lengthscale'])
    )
    
    observed_mean = observed_correlations.mean()
    observed_std = observed_correlations.std()
    residuals = observed_correlations - theoretical_corr.item()
    
    print(f"\n=== Results Analysis ===")
    print(f"Lengthscale: {results['lengthscale']:.4f}")
    print(f"Noise Scale: {results['noise_scale']:.4f}")
    print(f"Converged: {results['converged']}")
    
    print(f"\n=== Model Fit ===")
    print(f"Theoretical correlation at t=7.5: {theoretical_corr:.4f}")
    print(f"Observed mean correlation: {observed_mean:.4f}")
    print(f"Observed std correlation: {observed_std:.4f}")
    print(f"Mean absolute residual: {np.abs(residuals).mean():.4f}")
    print(f"RMSE: {np.sqrt((residuals**2).mean()):.4f}")
    
    # Check if noise estimate is reasonable
    expected_noise = observed_std
    print(f"\nNoise check - Estimated: {results['noise_scale']:.4f}, Expected ~{expected_noise:.4f}")

def main():
    time_diffs = np.array([7.5] * 8)
    observed_correlations = np.array([0.91, 0.61, 0.51, 0.19, 0.75, 0.59, 0.87, 0.58])

    print("Starting Type-II ML estimation...")
    print(f"Data: {observed_correlations}")
    print(f"Data mean: {observed_correlations.mean():.3f}, std: {observed_correlations.std():.3f}")

    # Fit the model
    results = fit_type_ii_ml(observed_correlations)
    
    # Analyze results
    analyze_results(observed_correlations, results)
    
    # Plot convergence if matplotlib available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(results['losses'])
        plt.title('Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('ELBO Loss')
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        # Plot smoothed loss
        window = 50
        if len(results['losses']) > window:
            smoothed = np.convolve(results['losses'], np.ones(window)/window, mode='valid')
            plt.plot(smoothed)
            plt.title('Smoothed Training Loss')
            plt.xlabel('Iteration')
            plt.ylabel('ELBO Loss')
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass

if __name__ == '__main__':
    main()