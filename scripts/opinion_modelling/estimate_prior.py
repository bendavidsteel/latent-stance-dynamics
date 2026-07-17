import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import gpytorch
import numpy as np

class CorrelationModel:
    def __init__(self, time_interval=7.5):
        self.time_interval = time_interval
        # Use GPyTorch kernel to ensure correct implementation
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def get_theoretical_correlation(self, lengthscale, output_scale):
        """Get theoretical correlation at the fixed time interval"""
        # Create dummy input points separated by time_interval
        x1 = torch.tensor([[0.0]])
        x2 = torch.tensor([[self.time_interval]])
        
        # Set kernel hyperparameters
        self.kernel.base_kernel.lengthscale = lengthscale
        self.kernel.outputscale = output_scale
        
        # Compute kernel value (this is the correlation)
        with torch.no_grad():
            correlation = self.kernel(x1, x2).squeeze()
        
        return correlation

def model(observed_correlations):
    """Pyro model for Type-II ML estimation"""
    
    # Priors on hyperparameters (broad, uninformative)
    lengthscale = pyro.sample("lengthscale", 
                             dist.LogNormal(torch.tensor(0.0), torch.tensor(2.0)))
    output_scale = pyro.sample("output_scale", 
                              dist.LogNormal(torch.tensor(0.0), torch.tensor(1.0)))
    noise_scale = pyro.sample("noise_scale", 
                             dist.LogNormal(torch.tensor(-2.0), torch.tensor(1.0)))
    
    # Create correlation model
    corr_model = CorrelationModel()
    
    # Get theoretical correlation at time interval 7.5
    theoretical_corr = corr_model.get_theoretical_correlation(lengthscale, output_scale)
    
    # Likelihood: observed correlations are noisy versions of theoretical correlation
    with pyro.plate("observations", len(observed_correlations)):
        pyro.sample("obs", 
                   dist.Normal(theoretical_corr, noise_scale), 
                   obs=observed_correlations)

def guide(observed_correlations):
    """Variational guide for SVI"""
    
    # Variational parameters for lengthscale
    lengthscale_loc = pyro.param("lengthscale_loc", torch.tensor(1.0))
    lengthscale_scale = pyro.param("lengthscale_scale", torch.tensor(0.5), 
                                  constraint=dist.constraints.positive)
    
    # Variational parameters for output_scale  
    output_scale_loc = pyro.param("output_scale_loc", torch.tensor(0.0))
    output_scale_scale = pyro.param("output_scale_scale", torch.tensor(0.5),
                                   constraint=dist.constraints.positive)
    
    # Variational parameters for noise_scale
    noise_scale_loc = pyro.param("noise_scale_loc", torch.tensor(-1.0))
    noise_scale_scale = pyro.param("noise_scale_scale", torch.tensor(0.5),
                                  constraint=dist.constraints.positive)
    
    # Sample from variational distributions
    pyro.sample("lengthscale", dist.LogNormal(lengthscale_loc, lengthscale_scale))
    pyro.sample("output_scale", dist.LogNormal(output_scale_loc, output_scale_scale))
    pyro.sample("noise_scale", dist.LogNormal(noise_scale_loc, noise_scale_scale))

def fit_type_ii_ml(observed_correlations, num_iterations=2000):
    """Fit Type-II ML using SVI"""
    
    # Convert to torch tensor
    if isinstance(observed_correlations, np.ndarray):
        observed_correlations = torch.tensor(observed_correlations, dtype=torch.float32)
    
    # Clear parameter store
    pyro.clear_param_store()
    
    # Setup SVI
    optimizer = Adam({"lr": 0.01})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    
    # Training loop
    losses = []
    for i in range(num_iterations):
        loss = svi.step(observed_correlations)
        losses.append(loss)
        
        if i % 200 == 0:
            print(f'Iteration {i}, Loss: {loss:.4f}')
    
    # Extract fitted parameters
    lengthscale_posterior = dist.LogNormal(
        pyro.param("lengthscale_loc"), 
        pyro.param("lengthscale_scale")
    )
    output_scale_posterior = dist.LogNormal(
        pyro.param("output_scale_loc"), 
        pyro.param("output_scale_scale")
    )
    noise_scale_posterior = dist.LogNormal(
        pyro.param("noise_scale_loc"), 
        pyro.param("noise_scale_scale")
    )
    
    # Get point estimates (posterior means)
    lengthscale_est = lengthscale_posterior.mean
    output_scale_est = output_scale_posterior.mean
    noise_scale_est = noise_scale_posterior.mean
    
    return {
        'lengthscale': lengthscale_est.item(),
        'output_scale': output_scale_est.item(), 
        'noise_scale': noise_scale_est.item(),
        'lengthscale_posterior': lengthscale_posterior,
        'output_scale_posterior': output_scale_posterior,
        'noise_scale_posterior': noise_scale_posterior,
        'losses': losses
    }

# Example usage with your data
# Replace this with your actual 12 correlation values
observed_correlations = torch.tensor([
    0.45, 0.52, 0.38, 0.48, 0.41, 0.55, 
    0.47, 0.43, 0.49, 0.46, 0.44, 0.50
], dtype=torch.float32)

# Fit the model
results = fit_type_ii_ml(observed_correlations)

print(f"\nType-II ML Results:")
print(f"Lengthscale: {results['lengthscale']:.4f}")
print(f"Output Scale: {results['output_scale']:.4f}")
print(f"Noise Scale: {results['noise_scale']:.4f}")

# Verify the fit
corr_model = CorrelationModel()
theoretical_corr = corr_model.get_theoretical_correlation(
    torch.tensor(results['lengthscale']), 
    torch.tensor(results['output_scale'])
)
print(f"\nTheoretical correlation at t=7.5: {theoretical_corr:.4f}")
print(f"Observed mean correlation: {observed_correlations.mean():.4f}")
print(f"Observed std correlation: {observed_correlations.std():.4f}")