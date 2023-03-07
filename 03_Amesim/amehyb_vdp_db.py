# from pytorch_minimize import optim as ptminoptim
import pathlib
import re
import signal
import typing
from collections import OrderedDict
from enum import Enum
from types import SimpleNamespace

import fmpy
import numpy as np
import plac
import torch
import torch_optimizer as optim_contrib
import tqdm
from fmudiff.modelexchange import FmuMEEvaluator, FmuMEModule
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

plt.style.use("bmh")
plt.rcParams["axes.titlesize"] = "medium"
plt.rcParams["figure.titlesize"] = "large"
plt.rcParams["axes.labelsize"] = "medium"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["text.color"] = "black"
plt.rcParams["figure.max_open_warning"] = 100
plt.rcParams["font.size"] = 8
plt.rcParams["lines.linewidth"] = 1.0
defclrs = plt.rcParams["axes.prop_cycle"].by_key()["color"]


LOAD_STATEDICT = False
LOAD_STATEDICT_FILENAME = "runs/vdp_mec_constrained/c_fcn_d/sd-run059-00700.pt"
SAVE_STATEDICT = True


torch.set_default_dtype(torch.float64)


OdeSolverType = Enum("solver", "euler rk4")


class Model(nn.Module):
    def __init__(
        self,
        odemodel: FmuMEModule,
        fcnid: nn.Module,
        dt: float,
        solver: OdeSolverType = OdeSolverType.euler,
    ):
        super().__init__()
        self.odemodel = odemodel
        self.fcnid = fcnid
        self.dt = dt
        self.solver = solver
        self.physics_parameters = SimpleNamespace(switch_c=1, x1=2.0)
        self.physics_parameters.__dict__["mass_friction_endstops_1.xmin"] = 2.1
        self.physics_parameters.__dict__["mass_friction_endstops_1.xmax"] = 2.1

    def forward(self, U, globalparam: dict = {}):
        dt = self.dt
        f = lambda u, x, tnow: FmuMEEvaluator.evaluate(
            u,
            x,
            self.odemodel.fmu,
            tnow,
            self.odemodel.pointers,
            self.odemodel._ru,
            self.odemodel._ry,
        )

        X = torch.empty(U.shape[0], U.shape[1], len(self.odemodel._rx))
        Y = torch.empty(U.shape[0], U.shape[1], len(self.odemodel._ry))

        nrbatches = U.shape[0]
        gp = OrderedDict(
            [
                (k, [v for _ in range(nrbatches)])
                for k, v in self.physics_parameters.__dict__.items()
            ]
        )
        for k, v in globalparam.items():
            gp[k] = v
        assert all([len(x_) == nrbatches for x_ in gp.values()])
        gpvaluerefs = [self.odemodel._vrs[x_] for x_ in gp.keys()]

        for ibatch in range(U.shape[0]):
            gpvalues = [x_[ibatch] for x_ in gp.values()]

            self.odemodel.fmu_initialize(gpvaluerefs, gpvalues)
            x = self.odemodel.state
            y = self.odemodel.output
            for it, _ in enumerate(U[ibatch]):
                if self.solver == OdeSolverType.euler:
                    # Euler
                    u = self.fcnid(y[[0]])
                    Y[ibatch, it, :] = y
                    X[ibatch, it, :] = x
                    self.odemodel.tnow += dt
                    dx, y = self.odemodel(u, x)
                    x = x + dt * dx
                else:
                    # RK4. Call Function.forward only once, as backward will be called
                    # as often (from docs: "Functions are what autograd uses to encode
                    # the operation history and compute gradients.")
                    u = self.fcnid(y[[0]])
                    Y[ibatch, it, :] = y
                    X[ibatch, it, :] = x

                    k1, _ = f(u, x, self.odemodel.tnow)
                    k2, _ = f(u, x + dt * k1 / 2.0, self.odemodel.tnow + dt / 2.0)
                    k3, _ = f(u, x + dt * k2 / 2.0, self.odemodel.tnow + dt / 2.0)
                    self.odemodel.tnow += dt
                    k4, y = self.odemodel(u, x + dt * k3)
                    x = x + 1.0 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)

            self.odemodel.fmu_terminate()

        return Y, X


class VdP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.ones((1)))

    def forward(self, U):
        return -self.mu * (1.0 - U**2)


class RefDataset(Dataset):
    def __init__(self, t, U, Y, X, gp) -> None:
        super().__init__()
        self.t = t
        self.U = U
        self.Y = Y
        self.X = X
        self.gp = gp

    def __getitem__(self, index):
        gp = self.gp.copy()
        for k, v in self.gp.items():
            gp[k] = v[index]
        return (self.t, self.U[index], self.Y[index], self.X[index], gp)

    def __len__(self):
        return self.U.shape[0]


class GracefulExiter:
    def __init__(self) -> None:
        self.state = False
        signal.signal(signal.SIGINT, self.state_change)

    def state_change(self, signum, frame):
        print("gracefully exiting... (repeat to exit now)")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.state = True

    def exit(self):
        return self.state


def next_run_nr(path: typing.AnyStr) -> int:
    runnames = [str(runname) for runname in pathlib.Path(path).glob("run*")]
    if runnames:
        return 1 + max(
            [
                int(m.group())
                for m in [re.search(r"(?<=_)\d+", runname) for runname in runnames]
                if m
            ]
        )
    else:
        return 1


@plac.pos("modelname", "model name", choices=["vdp_mec", "vdp_mec_constrained"])
@plac.pos("case", "optimization case", choices=["c_fcn_d", "mu", "constant"])
@plac.opt("learningrate", "learning rate", type=float)
@plac.opt("nrepochs", "number of epochs", type=int)
@plac.opt("solver", "ODE solver", type=str)
@plac.opt("timestep", "ODE solver time step", type=float)
# @plac.pos("loss_includes_state", "state included in loss computation", type=bool)
def main(
    modelname="vdp_mec",
    case="constant",
    learningrate=0.1,
    nrepochs=100,
    solver="rk4",
    timestep=1e-2,
    state_in_loss=False,
):

    if solver == "euler":
        odesolver = OdeSolverType.euler
    elif solver == "rk4":
        odesolver = OdeSolverType.rk4
    else:
        raise RuntimeError("unknown ode solver specified")

    fname_fmu = f"{modelname}.fmu.me"

    unzipdir = fmpy.extract(fname_fmu)
    model_description = fmpy.read_model_description(fname_fmu)

    fmu = fmpy.fmi2.FMU2Model(
        guid=model_description.guid,
        unzipDirectory=unzipdir,
        modelIdentifier=model_description.modelName,
        instanceName=pathlib.Path(fname_fmu).with_suffix("").name,
    )

    odemodel = FmuMEModule(fmu, model_description, verbose=False, logging=False)
    odemodel.eval()

    tend = 15.0
    dt = timestep
    t = np.arange(tend / dt + 1, dtype=np.float32) * dt

    # Reference simulation: in co-sim result is slightly different
    # x1_values = ((torch.arange(4.)/3-.5)/.5*2).tolist()
    # x1_values = ((torch.arange(3.)/2)**2*2).tolist()
    # x1_values = ((torch.arange(4.)/3)**2*2)[1:].tolist()  # don't start at 0; will not move
    # x1_values = torch.tensor([1.5, 2.0, 2.2])
    x1_values = torch.tensor([2.0])
    U = torch.zeros((len(x1_values), t.size, len(odemodel._ru)))

    # in co-sim result is slightly different, so use that for now
    if False:
        # reference simulation: built-in VdP damping definition
        fcnid = lambda u: torch.tensor([0.0])
        model = Model(odemodel, fcnid, dt, odesolver)
        model.physics_parameters.switch_c = int(2)

        # execute a reference simulation
        with torch.no_grad():
            Y, X = model(U)
        Yref = Y.detach()
        Xref = X.detach()
    else:
        # execute a reference simulation with a hybrid model
        fcnid = VdP()
        fcnid.mu.data[0] = torch.tensor(5.0)
        model = Model(odemodel, fcnid, dt, odesolver)

        with torch.no_grad():
            #Y, X = model(U, globalparam=OrderedDict(zip(["x1"], [x1_values])))
            Y, X = model(U, globalparam=OrderedDict([("x1", x1_values)]))
        Yref = Y.detach()
        Xref = X.detach()

    # set up the hybrid model to be trained
    if case == "c_fcn_d":
        # define a NN that is going to identify the damper rate
        fcnid = nn.Sequential(
            nn.Linear(1, 1),
            # nn.Linear(1, 10),
            nn.Linear(1, 6),
            nn.Sigmoid(),
            # nn.Linear(10, 1),
            nn.Linear(6, 1),
            nn.Linear(1, 1),
        )
    elif case == "mu":
        fcnid = VdP()
        fcnid.mu.data[0] = torch.tensor(5.0)
    elif case == "constant":

        class Constant(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.p = nn.Parameter(torch.ones((1)))

            def forward(self, U):
                return self.p

        fcnid = Constant()

    model = Model(odemodel, fcnid, dt, odesolver)

    dataset = RefDataset(t, U, Yref, Xref, OrderedDict(x1=x1_values))
    dataloader = DataLoader(dataset, len(x1_values))

    # loss_fcn = nn.L1Loss()
    # loss_fcn = nn.SmoothL1Loss()
    loss_fcn = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=learningrate)
    optimizer = optim_contrib.AdaBelief(model.parameters(), lr=learningrate)
    # optimizer = optim.RAdam(model.parameters(), lr=learningrate)
    # optimizer = optim.LBFGS(model.parameters(), lr=learningrate)
    model.train()

    # initial guess when estimating mu
    if case == "mu":
        model.fcnid.mu.data[0] = torch.tensor(3.0)

    if LOAD_STATEDICT:
        sd = torch.load(LOAD_STATEDICT_FILENAME)
        # optimizer.load_state_dict(sd.optimizer)
        model.load_state_dict(sd.model)
        # for pg in optimizer.param_groups:
        #     pg["lr"] = learningrate

    (pathlib.Path() / "runs" / modelname / case).mkdir(parents=True, exist_ok=True)
    run_nr = next_run_nr(f"runs/{modelname}/{case}")

    writer = SummaryWriter(
        f"runs/{modelname}/{case}/run_{run_nr:03d}", flush_secs=120, max_queue=20
    )

    writer.add_text(
        "optimizer", " | ".join([x_.strip() for x_ in repr(optimizer).split("\n")])
    )

    def training_loop(dataloader, model, loss_fcn, optimizer):
        # not using the dataloader for now; all batches at once
        if not state_in_loss:
            Ypred, _ = model(dataloader.dataset.U, dataloader.dataset.gp)
            loss = loss_fcn(Ypred, dataloader.dataset.Y)

        else:
            Ypred, Xpred = model(dataloader.dataset.U, dataloader.dataset.gp)
            loss = loss_fcn(
                torch.cat((Ypred, Xpred), dim=2),
                torch.cat((dataloader.dataset.Y, dataloader.dataset.X), dim=2),
            )
        optimizer.zero_grad()
        loss.backward()
        if isinstance(optimizer, optim.LBFGS):
            # https://pytorch.org/docs/stable/optim.html#optimizer-step-closure
            # closure should clear the gradients, compute the loss, and return it
            def closure():
                optimizer.zero_grad()
                Ypred, _ = model(dataloader.dataset.U, dataloader.dataset.gp)
                loss = loss_fcn(Ypred, dataloader.dataset.Y)
                loss.backward()
                return loss

            optimizer.step(closure)
        else:
            optimizer.step()

        return float(loss)

    flag = GracefulExiter()
    # loss_history = []
    with tqdm.tqdm(range(nrepochs)) as t:
        t.set_description(f"{case}:{run_nr:03d}")
        for epoch in t:
            epoch_loss = training_loop(dataloader, model, loss_fcn, optimizer)
            t.set_postfix(epoch_loss=epoch_loss)
            # loss_history.append(epoch_loss)
            if epoch % 2 == 0:
                try:
                    # if called too often this may fail from time to time (in which
                    # case best to decrease the frequency at which this is called)
                    writer.add_scalar("Epoch loss", epoch_loss, epoch)
                    if case == "mu":
                        writer.add_scalar("mu", float(model.fcnid.mu.data[0]), epoch)
                    if case == "constant":
                        writer.add_scalar(
                            "constant", float(model.fcnid.p.data[0]), epoch
                        )
                except:
                    pass
            if SAVE_STATEDICT:
                if epoch % 100 == 0:
                    sd = SimpleNamespace(
                        model=model.state_dict(), optimizer=optimizer.state_dict()
                    )
                    torch.save(
                        sd, f"runs/{modelname}/{case}/sd-run{run_nr:03d}-{epoch:05d}.pt"
                    )
                    torch.save(
                        sd, f"runs/{modelname}/{case}/sd-run{run_nr:03d}-latest.pt"
                    )
            if isinstance(optimizer, optim.LBFGS):
                break
            if flag.exit():
                break

    writer.close()

    sd = SimpleNamespace(model=model.state_dict(), optimizer=optimizer.state_dict())
    torch.save(sd, f"runs/{modelname}/{case}/sd-run{run_nr:03d}-latest.pt")

    return


def junk():

    d = (torch.arange(201.0) / 100 - 1) * 2
    fcnid_ref = VdP()
    fcnid_ref.mu.data[0] = torch.tensor(5.0)
    with torch.no_grad():
        cref = fcnid_ref(d)

    fnames = [
        # "sd-run059-02000.pt",
        # "sd-run059-12000.pt",
        # "sd-run059-27200.pt",
        "sd-run061-01000.pt",
        "sd-run061-05000.pt",
        "sd-run061-40000.pt",
    ]
    pred = OrderedDict()
    for fname in fnames:
        pred[fname] = OrderedDict()
        sd = torch.load(f"runs/vdp_mec_constrained/c_fcn_d/{fname}")
        model.load_state_dict(sd.model)
        with torch.no_grad():
            # U = dataset.U[[0],:,:]
            # gp = OrderedDict([(k, [v[-1]]) for k, v in dataset.gp.items()])
            Ypred, Xpred = model(dataset.U, dataset.gp)
            pred[fname]["Y"] = Ypred
            pred[fname]["X"] = Xpred
            pred[fname]["c"] = model.fcnid(d[None, :, None])[0, :, 0]

    # time domain output
    fig, ax = plt.subplots(1, Yref.shape[0], sharex=True, sharey=True, squeeze=False)
    for c in range(Yref.shape[0]):
        ax[0, c].set_title("x1={:.2f}".format(dataset.gp["x1"][c]))
        ax[0, c].set_xlabel("time [s]")
    ax[0, 0].set_ylabel("displacement [m]")
    for c in range(Yref.shape[0]):
        ax[0, c].plot(t, Yref[c, :, 0], label="ref", lw=3, ls="dashed")
    for fname, p in pred.items():
        for c in range(Yref.shape[0]):
            ax[0, c].plot(
                t, p["Y"][c, :, 0], label=pathlib.Path(fname).stem.strip("sd-")
            )
    ax[0, -1].legend()
    fig.suptitle("model output (loss is computed on the difference to the reference)")

    # damper characteristic
    fig, ax = plt.subplots(1, 1, sharex=True, squeeze=False)
    ax[0, 0].plot(d, cref, label="ref", lw=3, ls="dashed")
    for fname, p in pred.items():
        ax[0, 0].plot(d, p["c"], label=pathlib.Path(fname).stem.strip("sd-"))
    ax[0, 0].legend()
    ax[0, 0].set_xlabel("damper compression [m]")
    ax[0, 0].set_ylabel("damping rate [N/m/s]")
    fig.suptitle("damper characteristic (neural network behaviour)")

    # phase portrait
    fig, ax = plt.subplots(1, Yref.shape[0], sharex=True, squeeze=False)
    for c in range(Yref.shape[0]):
        ax[0, c].set_title("x1={:.2f}".format(dataset.gp["x1"][c]))
        ax[0, c].set_xlabel("displacement [m]")
    ax[0, 0].set_ylabel("velocity [m/s]")
    for c in range(Yref.shape[0]):
        ax[0, c].plot(Xref[c, :, 1], Xref[c, :, 0], label="ref", lw=3, ls="dashed")
    for fname, p in pred.items():
        for c in range(Yref.shape[0]):
            ax[0, c].plot(
                p["X"][c, :, 1],
                p["X"][c, :, 0],
                label=pathlib.Path(fname).stem.strip("sd-"),
            )
    fig.suptitle("phase portrait: system behaviour")

    # time domain states
    fig, ax = plt.subplots(
        Xref.shape[2], Yref.shape[0], sharex=True, sharey="row", squeeze=False
    )
    for c in range(Xref.shape[0]):
        ax[0, c].set_title("x1={:.2f}".format(dataset.gp["x1"][c]))
        ax[-1, c].set_xlabel("time [s]")
    ax[0, 0].set_ylabel("displacement [m]")
    ax[1, 0].set_ylabel("velocity [m/s]")
    for r in range(Xref.shape[2]):
        for c in range(Xref.shape[0]):
            ax[r, c].plot(t, Xref[c, :, 1 - r], label="ref")
    for fname, p in pred.items():
        for r in range(Xref.shape[2]):
            for c in range(Xref.shape[0]):
                ax[r, c].plot(
                    t, p["X"][c, :, 1 - r], label=pathlib.Path(fname).stem.strip("sd-")
                )
    ax[0, -1].legend()
    fig.suptitle("states: system behaviour")

    # loss as function of epoch
    m = [re.search(r"(?<=run)\d+(?=-)", fname) for fname in fnames]
    runnrs = list(set([int(m_.group()) for m_ in m if m]))
    loss = OrderedDict()
    for runnr in runnrs:
        path_tb = pathlib.Path() / f"runs/vdp_mec_constrained/c_fcn_d/run_{runnr:03d}"
        fname = list(path_tb.glob("*tfevents*"))[0]
        loss[runnr] = read_tbsummary(fname)
    fig, ax = plt.subplots(1, 1, sharex=True, squeeze=False)
    ax[0, 0].set_yscale("log")
    for k, runnr in enumerate(runnrs):
        e, l = list(zip(*loss[runnr]["Epoch loss"]))
        ax[0, 0].plot(e, l, label=f"Run {runnr:d} loss", color="tab:gray")
    y0, y1 = ax[0, 0].get_ylim()
    x = [int(re.search(r"(?<=-)\d+(?=.pt)", fname).group()) for fname in fnames]
    for k, fname in enumerate(fnames):
        ax[0, 0].vlines(
            [x[k]],
            ymin=y0,
            ymax=y1,
            colors=defclrs[k + 1],
            label=pathlib.Path(fname).stem.strip("sd-"),
        )
    ax[0, 0].set_ylim([y0, y1])
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel("Loss")
    ax[0, 0].legend()
    fig.suptitle("Loss (with AdaBelief optimizer)")


def read_tbsummary(fname: str | pathlib.Path) -> dict:
    import struct
    from tensorboard.compat.proto import event_pb2

    def read_chunk(data):
        header = struct.unpack("Q", data[:8])
        event_str = data[12 : 12 + int(header[0])]
        data = data[12 + int(header[0]) + 4 :]
        return data, event_str

    with open(fname, "rb") as f:
        data = f.read()

    values = {}
    while data:
        data, evtstr = read_chunk(data)
        event = event_pb2.Event()
        event.ParseFromString(evtstr)
        if event.HasField("summary"):
            for value in event.summary.value:
                if value.HasField("simple_value"):
                    values.setdefault(value.tag, [])
                    values[value.tag].append((event.step, value.simple_value))
    return values


if __name__ == "__main__":
    plac.call(main)
