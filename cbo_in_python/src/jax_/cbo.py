import torch
import numpy as np

from cbo_in_python.src.jax_.utils import inplace_randn, randn
from torch.utils.data import DataLoader

import jax
from jax import random as jrandom
import jax.numpy as jnp
import functools
import threading

def cbo_update(V, V_alpha, anisotropic, l, sigma, dt, device):
    # device = torch.device('cpu') if device is None else device
    # noise = inplace_randn(V.shape, device)
    key, key2 = jrandom.split(jrandom.PRNGKey(np.random.randint(0, 100)))

    # noise should be of the same structure as the particle parameters V
    # meaning if we have 200 particles with parameters for a NN (organised in layers with kernels and biases)
    # we want the noise in the same exact form
    randinit = lambda x: jrandom.normal(key, x.shape)
    noise = jax.tree_util.tree_map(randinit, V[0])



    noise = np.array([jax.tree_util.tree_map(lambda V_: [jrandom.normal(jrandom.split(jrandom.PRNGKey(np.random.randint(0, 100)))[0], v.shape) for v in V_] , V_) for V_ in V], dtype=object)
    # noise = jrandom.normal(key, V.shape)
    # with torch.no_grad():
    diff = V - V_alpha
    # noise_weight = jax.abs(diff) if anisotropic else jax.norm(diff, p=2, dim=1)
    # noise_weight = jnp.abs(diff) if anisotropic else jnp.linalg.norm(diff, ord=2, axis=1)
    get_norm_of_w_or_b = lambda diff: [jnp.linalg.norm(d, ord=2) for d in diff]
    get_norm_of_nn = lambda diff: jnp.linalg.norm((jnp.array(get_norm_of_w_or_b(diff))**2))
    noise_weight = np.array([get_norm_of_nn(d) for d in diff])

    noise_term = np.array([noise[i]*noise_weight[i] for i in range(noise_weight.shape[0])])
    noise_term = sigma * noise_term * (dt ** 0.5)

    V -= l * diff * dt
    V = np.array([V[i]*noise_term[i] for i in range(noise_term.shape[0])])
    return V


def compute_v_alpha(energy_values, particles, alpha, device=None, argmin=False):
    # device = torch.device('cpu') if device is None else device
    if argmin:
        consensus_index = energy_values.argmin()
        consensus = particles[consensus_index]
        return consensus
    else:
        weights = np.exp(-alpha * (energy_values - energy_values.min())).reshape(-1, 1)

        normalised_weights = weights/weights.sum()

        get_weighted_particle = lambda x, w: jax.tree_util.tree_map(functools.partial(lambda y, v: y*v, v=w), x)

        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(4)
        result = pool.starmap(get_weighted_particle, tuple([(particles[i], normalised_weights[i]) for i in range(len(particles))]))
        pool.close()
        pool.join()

        # Sum over the weighted particles to get the consensus point
        consensus = result[0]
        for i in range(1, len(result)):
            consensus = jax.tree_map(lambda x, y: x+y, consensus, result[i])

        # apply_weights = lambda p, w, s: w*p/s
        # tree_map = lambda p, w, f, s: jax.tree_util.tree_map(f, p, w, s)
        # partial_tree_map = functools.partial(tree_map, f=apply_weights, s=weights.sum())
        # consensus = partial_tree_map(weights, particles)
        # v_tree_map = jax.vmap(partial_tree_map, (0, 0))
        # v_tree_map(particles, weights)
        # # apply_weights = jax.vmap(apply_weights, (0, 0, 0))
        # # apply_weights(weights, particles, weights.sum())
        # apply_weights = functools.partial(apply_weights, s=weights.sum())
        # consensus = jax.tree_util.tree_map(apply_weights, particles[0], weights[:,0])
        # det_cons = lambda p: jax.tree_util.tree_map(apply_weights, p)
        # tree_map = functools.partial(jax.tree_util.tree_map, f = apply_weights)


        # consensus = (weights * particles) / weights.sum()
        return consensus


def compute_energy_values(function, V, device=None):
    # TODO(itukh) is it possible to apply vectorization here to improve the performance
    # device = torch.device('cpu') if device is None else device
    return jnp.stack([function(v) for v in V])


def minimize(
        # General CBO / optimization parameters
        function,
        dimensionality,
        n_particles,
        initial_distribution,
        dt,
        l,
        sigma,
        alpha,
        anisotropic,
        # Optimization parameters
        batch_size=None,
        n_particles_batches=None,
        epochs=None,
        time_horizon=None,
        # Optimization modifications parameters
        use_partial_update=False,
        use_additional_random_shift=False,
        use_additional_gradients_shift=False,
        random_shift_epsilon=None,
        gradients_shift_gamma=None,
        # Additional optional arguments
        best_particle_alpha=1e5,
        use_gpu_if_available=False,
        use_multiprocessing=False,
        return_trajectory=False,
        cooling=False):
    # Setting up computations on GPU / CPU
    # device = torch.device('cuda') if (use_gpu_if_available and torch.cuda.is_available()) else torch.device('cpu')
    # Standardize input arguments
    batch_size = int(n_particles // n_particles_batches) if batch_size is None else batch_size
    epochs = int(time_horizon // dt) if epochs is None else epochs
    # Initialize variables
    # V = initial_distribution.sample((n_particles, dimensionality)).to(device) # What is initial distribution?
    V = initial_distribution.sample((n_particles, dimensionality))
    # TODO: Needed for JAX implementation of Gradient shift?
    # if use_additional_gradients_shift:
    #     V.requires_grad = True

    # Keep Dataloader in JAX implementation for now
    V_batches = DataLoader(np.arange(n_particles), batch_size=batch_size, shuffle=True)
    V_alpha_old = None

    # Main optimization loop
    trajectory = []
    for epoch in range(epochs):
        for batch in V_batches:
            V_batch = V[batch]
            # batch_energy_values = compute_energy_values(function, V_batch, device=device)
            # V_alpha = compute_v_alpha(batch_energy_values, V_batch, alpha, device=device)
            batch_energy_values = compute_energy_values(function, V_batch)
            V_alpha = compute_v_alpha(batch_energy_values, V_batch, alpha)

            if use_partial_update:
                # V[batch] = cbo_update(V_batch, V_alpha, anisotropic, l, sigma, dt, device=device)

                V[batch] = cbo_update(V_batch, V_alpha, anisotropic, l, sigma, dt)
            else:
                # V = cbo_update(V, V_alpha, anisotropic, l, sigma, dt, device=device)
                V = cbo_update(V, V_alpha, anisotropic, l, sigma, dt)

            if use_additional_random_shift:
                if V_alpha_old is None:
                    continue
                # norm = torch.norm(V_alpha.view(-1) - V_alpha_old.view(-1), p=float('inf'), dim=0).detach().cpu().numpy()
                norm = jnp.linalg.norm(V_alpha.view(-1) - V_alpha_old.view(-1), ord=jnp.inf, axis=0)
                if np.less(norm, random_shift_epsilon):
                    # V += sigma * (dt ** 0.5) * inplace_randn(V.shape, device=device)
                    V += sigma * (dt ** 0.5) * randn(V.shape)
                V_alpha_old = V_alpha

        if use_additional_gradients_shift:

            # TODO: What's happening here? How to translate to JAX?
            if V.grad is not None:
                V.grad.zero_()
            # energy_values = compute_energy_values(function, V, device=device)
            energy_values = compute_energy_values(function, V)
            loss = energy_values.sum()
            loss.backward()
            # with torch.no_grad():
            # TODO: Gradient determiniation with JAX for V
            V -= gradients_shift_gamma * V.grad

        if return_trajectory:
            # energy_values = compute_energy_values(function, V, device=device)
            # V_alpha = compute_v_alpha(energy_values, V, alpha, device=device)
            # V_best = compute_v_alpha(energy_values, V, best_particle_alpha, device=device)
            # trajectory.append(
            #     {
            #         'V': V.clone().detach().cpu(),
            #         'V_alpha': V_alpha.clone().detach().cpu(),
            #         'V_best': V_best.clone().detach().cpu(),
            #     }
            # )

            energy_values = compute_energy_values(function, V)
            V_alpha = compute_v_alpha(energy_values, V, alpha)
            V_best = compute_v_alpha(energy_values, V, best_particle_alpha)
            trajectory.append(
                {
                    'V': V.clone(),
                    'V_alpha': V_alpha.clone(),
                    'V_best': V_best.clone()
                }
            )

        if cooling:
            alpha = alpha * 2
            sigma = sigma * np.log2(epoch + 1) / np.log2(epoch + 2)

    # energy_values = compute_energy_values(function, V, device=device)
    # V_alpha = compute_v_alpha(energy_values, V, alpha, device=device)
    # if return_trajectory:
    #     return V_alpha.detach().cpu(), trajectory

    # return V_alpha.detach().cpu()
    energy_values = compute_energy_values(function, V)
    V_alpha = compute_v_alpha(energy_values, V, alpha)
    if return_trajectory:
        return V_alpha, trajectory
    return V_alpha
