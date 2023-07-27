# TODO(itukh): fix energy values calculation
import os

# import torch.multiprocessing as mp

import multiprocessing as mp

import torch
import numpy as np

from cbo_in_python.src.jax_.particle import Particle
from cbo_in_python.src.jax_.cbo import cbo_update, compute_v_alpha
from cbo_in_python.src.jax_.utils import inplace_randn, randn

from torch.utils.data import DataLoader

import jax
import jax.numpy as jnp
import jax.random as jrandom

import flax
import time


class Optimizer:
    def __init__(self, model, n_particles=10, l=1, alpha=100, sigma=1, dt=0.01, anisotropic=True, eps=1e-2,
                 use_multiprocessing=False, n_processes=4, particles_batch_size=None, apply_random_drift=True,
                 gamma=None, device=None, partial_update=False, apply_common_drift=False,
                 evaluation_strategy='last', fmu=False, residual=False):
        """
        Consensus based optimizer.
        :param model: model to optimize.
        :param n_particles: number of particles to use in the optimization.
        :param l: alias for `lambda`, CBO hyperparameter.
        :param alpha: CBO hyperparameter.
        :param sigma: CBO hyperparameter.
        :param dt: CBO dynamics time step.
        :param anisotropic: boolean flag indicating whether to use the anisotropic noise or not.
        :param eps: argument indicating how small the consensus update has to be to apply the additional random shift
        (drift) to particles.
        :param use_multiprocessing: whether to use multiprocessing where possible.
        :param n_processes: number of processes to use for multiprocessing.
        :param particles_batch_size: batch size for particle-level batching. If not specified, no batching will be used.
        :param apply_random_drift: whether to apply additional random shift (drift) or not.
        :param gamma: coefficient of a gradient drift. If gamma is None, no gradient drift will be applied.
        :param partial_update: whether to apply CBO update to all particles, or just to the corresponding bunch of
        particles.
        """
        # CBO hyperparameters
        self.n_particles = n_particles
        self.l = l
        self.alpha = alpha
        self.sigma = sigma
        self.dt = dt
        self.anisotropic = anisotropic
        # CBO additional hyperparameters
        self.gamma = gamma
        self.eps = eps  # specifies how small the consensus updated has to be to apply the additional shift
        self.fmu = fmu
        self.residual = residual
        # CBO dynamics additional settings
        self.apply_random_drift = apply_random_drift
        self.apply_common_drift = apply_common_drift
        self.partial_update = partial_update
        if evaluation_strategy not in {'last', 'full', 'best'}:
            raise ValueError(f'Unknown model evaluation strategy: {evaluation_strategy}')
        self.evaluation_strategy = evaluation_strategy
        # Multiprocessing settings
        self.use_multiprocessing = use_multiprocessing
        self.n_processes = min(n_processes, mp.cpu_count())

        # TODO: replace with JAX Device settings
        # Device (CPU / GPU / TPU) settings
        # self.device = torch.device('cpu') if device is None else device
        # if self.use_multiprocessing:
        #     if device.type == 'cuda':
        #         raise RuntimeError('Unable to use multiprocessing along with cuda')
        #     torch.set_num_threads(n_processes)
        self.device = device

        # Initialize required internal fields
        self.time = 0
        self.particles = []
        self.loss = None
        self.X = None
        self.y = None
        self.V = None
        self.V_alpha = None
        self.V_alpha_old = None
        self.shift_norm = None
        self.consensus = None
        self.epoch = 1
        # Initialize particles
        self.model = model
        self._initialize_particles()
        self.particles_batch_size = particles_batch_size if particles_batch_size is not None else self.n_particles
        self.particles_dataloader = DataLoader(np.arange(self.n_particles), batch_size=self.particles_batch_size,
                                               shuffle=True)

        # TODO: Jax device settings
        # if device is not None:
        #     self.to(device)

        # Constants
        self.infinity = 1e5

    def cooling_step(self):
        self.alpha = self.alpha * np.log2(self.epoch + 2) / np.log2(self.epoch + 1)
        self.sigma = self.sigma * np.log2(self.epoch + 1) / np.log2(self.epoch + 2)
        self.dt = self.dt * (self.epoch + 1)/(self.epoch + 2)
        self.epoch += 1

    def set_loss(self, loss):
        """
        Updates the optimization loss (energy) function.
        :param loss: new loss function. Should take as arguments the model outputs and targets respectively.
        """
        self.loss = loss

    # TODO: Needed for JAX?
    def set_batch(self, X, y):
        """
        Updates the data batch to evaluate the energy function on.
        """
        # self.X = X.to(self.device)
        # self.y = y.to(self.device)
        self.X = X
        self.y = y

    def compute_consensus(self, batch=None, alpha=None, outputs=None):
        """
        Returns the consensus computed based on the current particles positions.
        """
        batch = np.arange(len(self.particles)) if batch is None else batch
        outputs = self._compute_particles_outputs(batch) if outputs is None else outputs
        # TODO(itukh): check this line
        # values = torch.FloatTensor([self.loss(output, self.y) for output in outputs])
        values = jnp.array([self.loss(output, self.y) for output in outputs])
        alpha = self.alpha if alpha is None else alpha
        return compute_v_alpha(values, self.V[batch], alpha, self.device)

    def step(self):
        """
        Execute one step of the CBO dynamics.
        """
        if self.X is None:
            raise RuntimeError('Unable to perform the step without the prior loss.backward() call')
        start = time.time()
        self.V = self._get_particles_params() # V is the nn_parameters of the particles
        # print(f'_get_particles_params: {time.time()-start}')
        # self.V = self.particles

        time__compute_particles_outputs = []
        time__compute_energy_values = []
        time_compute_v_alpha = []
        time_cbo_update = []

        # TODO: Multiprocessing for JAX
        if self.use_multiprocessing:
            self.V.share_memory_()
            outputs = self._compute_particles_outputs()
            # energy_values = torch.FloatTensor([self.loss(output, self.y) for output in outputs]).to(self.device)
            energy_values = jnp.array([self.loss(output, self.y) for output in outputs])
            self.V_alpha_old = self.V_alpha.clone() if self.V_alpha is not None else None
            self.V_alpha = self.compute_consensus(batch=self._generate_random_batch())  # TODO: check this line

            batches = [batch for batch in self.particles_dataloader]
            # q = mp.Queue()  Could we use it here?
            params = [(energy_values, self.V, batch, self.alpha, self.anisotropic,
                       self.l, self.sigma, self.dt) for batch in batches]
            with mp.Pool(processes=self.n_processes) as pool:
                new_V = pool.starmap(_batch_step, params)
            for new_batch_V, batch in zip(new_V, batches):
                self.V[batch] = new_batch_V
            self._maybe_apply_random_shift()
            self._maybe_apply_gradient_shift()
        else:
            for particles_batch in self.particles_dataloader:
                particles_batch = np.array(particles_batch)
                start = time.time()
                outputs = self._compute_particles_outputs(particles_batch)
                time__compute_particles_outputs.append(time.time()-start)
                # energy_values = torch.FloatTensor([self.loss(output, self.y) for output in outputs]).to(self.device)

                start = time.time()
                energy_values = jnp.array([self.loss(output, self.y) for output in outputs])
                time__compute_energy_values.append(time.time()-start)

                # self.V_alpha_old = self.V_alpha.clone() if self.V_alpha is not None else None
                self.V_alpha_old = self.V_alpha.copy() if self.V_alpha is not None else None

                start = time.time()
                # V_batch = [self.V[i].model.nn_parameters for i in particles_batch]
                self.V_alpha = compute_v_alpha(energy_values, self.V[particles_batch], self.alpha, self.device)
                time_compute_v_alpha.append(time.time()-start)

                start = time.time()
                if self.partial_update:
                    self.V[particles_batch] = cbo_update(self.V[particles_batch], self.V_alpha, self.anisotropic,
                                                         self.l, self.sigma, self.dt, self.device)
                else:
                    self.V = cbo_update(self.V, self.V_alpha, self.anisotropic,
                                        self.l, self.sigma, self.dt, self.device)
                time_cbo_update.append(time.time()-start)
                self._maybe_apply_random_shift()
                self._maybe_apply_gradient_shift(batch=particles_batch)

        start = time.time()
        self._set_particles_params(self.V)

        # print(f'time__compute_particles_outputs: {np.sum(np.array(time__compute_particles_outputs))}')
        # print(f'time__compute_energy_values: {np.sum(np.array(time__compute_energy_values))}')
        # print(f'time_compute_v_alpha: {np.sum(np.array(time_compute_v_alpha))}')
        # print(f'time_cbo_update: {np.sum(np.array(time_cbo_update))}')
        # print(f'_set_particles_params: {time.time()-start}')

        self._maybe_apply_common_drift()
        self._update_model_params()
        self.time += self.dt

    def backward(self, loss):
        """
        Applies backpropagation for each dynamics particle.
        """
        # TODO: backward function is used for gradient determination, replace with JAX gradient determination
        # TODO: use stack here instead
        outputs = self._compute_particles_outputs()
        for output in outputs:
            loss_value = loss(output, self.y)
            loss_value.backward()

    def zero_grad(self):
        """
        Zeroes the gradient values for all the particles and model. May be helpful when using the gradients.
        """
        print('zero_grad fucntion probably not necessary with JAX implementation. Dont call it')
        for particle in self.particles:
            particle.zero_grad()
        self.model.zero_grad()

    def get_current_time(self):
        """
        Returns the current timestamp. Timestamp is incremented bt the `dt` on every optimization step,
        """
        return self.time

    def _generate_random_batch(self):
        return np.random.choice(np.arange(self.n_particles), self.particles_batch_size, replace=False)

    def to(self, device):
        """
        Transfers optimization to a new device. Typical application is cuda usage.
        """
        self.device = device
        for i, particle in enumerate(self.particles):
            self.particles[i] = particle.to(device)
        self.model = self.model.to(device)
        if self.X is not None:
            self.X = self.X.to(device)
            self.y = self.y.to(device)

    def _maybe_apply_random_shift(self):
        if not self.apply_random_drift:
            return
        if self.V_alpha_old is not None:
            # norm = torch.norm(self.V_alpha.view(-1) - self.V_alpha_old.view(-1), p=float('inf'),
            #                   dim=0).detach().cpu().numpy()
            norm = []
            for i in range(self.V_alpha.shape[0]):
                norm.append(jnp.linalg.norm(self.V_alpha[i] - self.V_alpha_old[i], ord=np.inf))
            norm = jnp.linalg.norm(jnp.array(norm), ord=np.inf, axis=0)

            if jnp.less(norm, self.eps):
                self.V += self.sigma * (self.dt ** 0.5) * randn(self.V.shape, self.device)

            self.shift_norm = norm

    def _maybe_apply_gradient_shift(self, batch=None):
        # TODO: adapt to JAX
        if self.gamma is None:
            return
        batch = np.arange(self.V.shape[0]) if batch is None else batch
        self.V[batch] -= torch.cat([self.particles[i].get_gradient() for i in batch]).view(
            self.V[batch].shape) * self.gamma * self.dt

    def _maybe_apply_common_drift(self):
        if not self.apply_common_drift:
            return
        outputs = self._compute_particles_outputs()
        # energy_values = torch.FloatTensor([self.loss(output, self.y) for output in outputs]).to(self.device)
        energy_values = jnp.array([self.loss(output, self.y) for output in outputs])
        self.V_alpha = compute_v_alpha(energy_values, self.V, self.alpha, self.device)
        self.V = cbo_update(self.V, self.V_alpha, self.anisotropic, self.l, self.sigma, self.dt, self.device)
        self._set_particles_params(self.V)

    def _initialize_particles(self):
        self.particles = [Particle(self.model, self.fmu, self.residual) for _ in range(self.n_particles)]

    def _get_particles_params(self):
        return np.stack([particle.get_params() for particle in self.particles])

    def __get_particles_params(self):
        return [particle.model.nn_parameters for particle in self.particles]

    def _set_particles_params(self, new_particles_params):
        for particle, new_particle_params in zip(self.particles, new_particles_params):
            particle.set_params(new_particle_params)

    def _compute_particles_outputs(self, batch=None):
        # TODO: implement in a more efficient manner
        # TODO: adapt multiprocessing to JAX
        batch = np.arange(len(self.particles)) if batch is None else batch
        values = []
        if self.use_multiprocessing:
            q = mp.Queue()
            ps = []
            for i in range(self.n_processes):
                p = mp.Process(target=_forward, args=(q, self.particles[i], self.X,))
                ps.append(p)
                p.start()
            print(q.get())

            with mp.Pool(processes=4) as pool:

                # print "[0, 1, 4,..., 81]"
                print(pool.starmap(f, [(i, i-5) for i in range(10)]))

                values = pool.starmap(_forward, [(self.particles[i], self.X) for i in batch]) # PYTHON MP

            # p = mp.Process(target=_forward, args=(q, self.particles[0], self.X,))
            # with mp.Pool(processes=self.n_processes) as pool:
            #     # values = pool.starmap(_forward, [(self.particles[i], self.X) for i in batch]) # TORCH MP
            #
        else:
            for i in batch:
                values.append(self.particles[i](self.X))
        return values

    def _update_model_params(self):
        new_params = None
        if self.evaluation_strategy == 'last':
            new_params = self.V_alpha
        elif self.evaluation_strategy == 'full':
            self.V_alpha = self.compute_consensus()
            new_params = self.V_alpha
        elif self.evaluation_strategy == 'best':
            self.V_alpha = self.compute_consensus(alpha=self.infinity)
            new_params = self.V_alpha
        if new_params is None:
            return
        next_slice = 0
        # TODO give Flax model a function which returns the parameters as a iterable/iterator
        self.model.set_parameters_flat(new_params)
        # for p in self.model.parameters():
        #     slice_length = len(p.view(-1))
        #     # with torch.no_grad():
        #         # p.copy_(new_params[next_slice: next_slice + slice_length].view(p.shape))
        #     p.copy(new_params[next_slice: next_slice + slice_length].view(p.shape))
        #     next_slice += slice_length


# Multiprocessing helper functions

def _forward(q, model, X):
    print('process id:', os.getpid())
    # print(f'Model: {model}')
    value = model(X)
    print('Finished _forward')
    return q.put(value.detach())

def _batch_step(energy_values, V, alpha, anisotropic, l, sigma, dt):
    V_alpha = compute_v_alpha(energy_values, V, alpha)
    return cbo_update(V, V_alpha, anisotropic, l, sigma, dt)
