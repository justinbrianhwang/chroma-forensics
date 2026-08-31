"""Model zoo: roots trained from scratch, then post-training operations applied to them."""
import copy, json, time, hashlib, math, shutil
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA = str(ROOT_DIR / "data")
DEV = "cuda"

CIFAR_MEAN = torch.tensor([125.3, 123.0, 113.9]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([63.0, 62.1, 66.7]).view(1, 3, 1, 1)
SVHN_MEAN = torch.tensor([111.60894, 113.16128, 120.56513]).view(1, 3, 1, 1)
SVHN_STD = torch.tensor([50.49768, 51.25898, 50.24422]).view(1, 3, 1, 1)
NORMALIZATION = {"cifar10": (CIFAR_MEAN, CIFAR_STD), "svhn": (SVHN_MEAN, SVHN_STD)}


# ---------------------------------------------------------------- data

def _to_gpu(x, dataset="cifar10"):
    x = x.to(DEV).float()
    mean, std = NORMALIZATION[dataset]
    x = (x - mean.to(DEV)) / std.to(DEV)
    return x.contiguous(memory_format=torch.channels_last)


def load_splits(seed=0, dataset="cifar10"):
    """D_root / D_operation / D_probe / D_eval, mutually disjoint (plan 10.1).

    D_probe deliberately includes an out-of-domain source (SVHN) so a detector
    cannot succeed by recognising the intervention corpus.
    """
    if dataset not in NORMALIZATION:
        raise ValueError(f"unknown dataset: {dataset}")
    cifar_test = tv.datasets.CIFAR10(DATA, train=False, download=False)
    Xcifar = torch.tensor(cifar_test.data).permute(0, 3, 1, 2)
    ycifar = torch.tensor(cifar_test.targets)
    svhn_test = tv.datasets.SVHN(DATA, split="test", download=False)
    Xsvhn = torch.tensor(svhn_test.data)

    if dataset == "cifar10":
        train = tv.datasets.CIFAR10(DATA, train=True, download=False)
        Xtr = torch.tensor(train.data).permute(0, 3, 1, 2)
        ytr = torch.tensor(train.targets)
        eval_x, eval_y = Xcifar[2000:], ycifar[2000:]
    else:
        train = tv.datasets.SVHN(DATA, split="train", download=False)
        Xtr = torch.tensor(train.data)
        ytr = torch.tensor(train.labels).long()
        eval_x = Xsvhn[-8000:]
        eval_y = torch.tensor(svhn_test.labels[-8000:]).long()

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(Xtr), generator=g)
    root_idx, op_idx = perm[:40000], perm[40000:50000]

    return dict(
        root=(_to_gpu(Xtr[root_idx], dataset), ytr[root_idx].to(DEV)),
        op=(_to_gpu(Xtr[op_idx], dataset), ytr[op_idx].to(DEV)),
        # One fixed public probe tensor for every population.
        probe_id=_to_gpu(Xcifar[:2000]),
        probe_ood=_to_gpu(Xsvhn[:2000]),
        probe_id_y=ycifar[:2000].to(DEV),
        eval=(_to_gpu(eval_x, dataset), eval_y.to(DEV)),
    )


# ---------------------------------------------------------------- model

def _blk(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False),
                         nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class SmallCNN(nn.Module):
    def __init__(self, nc=10, w=64):
        super().__init__()
        self.s1 = nn.Sequential(_blk(3, w), _blk(w, 2 * w), nn.MaxPool2d(2))
        self.s2 = nn.Sequential(_blk(2 * w, 2 * w), _blk(2 * w, 2 * w), nn.MaxPool2d(2))
        self.s3 = nn.Sequential(_blk(2 * w, 4 * w), _blk(4 * w, 4 * w), nn.MaxPool2d(2))
        self.pool = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten())
        self.head = nn.Linear(4 * w, nc)

    def forward(self, x, acts=False):
        a1 = self.s1(x)
        a2 = self.s2(a1)
        a3 = self.s3(a2)
        z = self.pool(a3)
        out = self.head(z)
        if acts:
            # spatially pooled stage outputs + penultimate embedding
            return out, [a1.mean((2, 3)), a2.mean((2, 3)), a3.mean((2, 3)), z]
        return out


class BasicBlock(nn.Module):
    def __init__(self, ci, co, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(co)
        self.conv2 = nn.Conv2d(co, co, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(co)
        self.skip = (nn.Identity() if stride == 1 and ci == co else
                     nn.Sequential(nn.Conv2d(ci, co, 1, stride=stride, bias=False),
                                   nn.BatchNorm2d(co)))

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.bn2(self.conv2(x)) + residual, inplace=True)


class ResNetSmall(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.stem = _blk(3, 64)
        self.s1 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.s2 = nn.Sequential(BasicBlock(128, 128, 2), BasicBlock(128, 128))
        self.s3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.pool = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten())
        self.head = nn.Linear(256, nc)

    def forward(self, x, acts=False):
        x = self.stem(x)
        a1 = self.s1(x)
        a2 = self.s2(a1)
        a3 = self.s3(a2)
        z = self.pool(a3)
        out = self.head(z)
        if acts:
            return out, [a1.mean((2, 3)), a2.mean((2, 3)), a3.mean((2, 3)), z]
        return out


class VGGSmall(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.s1 = nn.Sequential(_blk(3, 64), _blk(64, 128), _blk(128, 128),
                                nn.MaxPool2d(2))
        self.s2 = nn.Sequential(_blk(128, 128), _blk(128, 128), _blk(128, 128),
                                nn.MaxPool2d(2))
        self.s3 = nn.Sequential(_blk(128, 256), _blk(256, 256), _blk(256, 256),
                                nn.MaxPool2d(2))
        self.pool = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten())
        self.head = nn.Linear(256, nc)

    def forward(self, x, acts=False):
        a1 = self.s1(x)
        a2 = self.s2(a1)
        a3 = self.s3(a2)
        z = self.pool(a3)
        out = self.head(z)
        if acts:
            return out, [a1.mean((2, 3)), a2.mean((2, 3)), a3.mean((2, 3)), z]
        return out


def _separable(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, ci, 3, padding=1, groups=ci, bias=False),
        nn.BatchNorm2d(ci), nn.ReLU(inplace=True),
        nn.Conv2d(ci, co, 1, bias=False),
        nn.BatchNorm2d(co), nn.ReLU(inplace=True),
    )


class DepthwiseCNN(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.s1 = nn.Sequential(_separable(3, 128), _separable(128, 128),
                                nn.MaxPool2d(2))
        self.s2 = nn.Sequential(_separable(128, 128), _separable(128, 128),
                                nn.MaxPool2d(2))
        self.s3 = nn.Sequential(_separable(128, 256), _separable(256, 256),
                                nn.MaxPool2d(2))
        self.pool = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten())
        self.head = nn.Linear(256, nc)

    def forward(self, x, acts=False):
        a1 = self.s1(x)
        a2 = self.s2(a1)
        a3 = self.s3(a2)
        z = self.pool(a3)
        out = self.head(z)
        if acts:
            return out, [a1.mean((2, 3)), a2.mean((2, 3)), a3.mean((2, 3)), z]
        return out


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = channels // reduction
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        scale = torch.sigmoid(self.fc2(F.relu(self.fc1(self.pool(x)), inplace=True)))
        return x * scale


class SEBasicBlock(BasicBlock):
    def __init__(self, ci, co, stride=1):
        super().__init__(ci, co, stride)
        self.se = SqueezeExcitation(co, reduction=8)

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.se(self.bn2(self.conv2(x)))
        return F.relu(x + residual, inplace=True)


class SEResNetSmall(ResNetSmall):
    def __init__(self, nc=10):
        nn.Module.__init__(self)
        self.stem = _blk(3, 64)
        self.s1 = nn.Sequential(SEBasicBlock(64, 128, 2), SEBasicBlock(128, 128))
        self.s2 = nn.Sequential(SEBasicBlock(128, 128, 2), SEBasicBlock(128, 128))
        self.s3 = nn.Sequential(SEBasicBlock(128, 256, 2), SEBasicBlock(256, 256))
        self.pool = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten())
        self.head = nn.Linear(256, nc)


MODELS = {
    "cnn": SmallCNN,
    "resnet": ResNetSmall,
    "vgg": VGGSmall,
    "depthwise": DepthwiseCNN,
    "seresnet": SEResNetSmall,
}


def new_model(seed, arch="cnn", w=64):
    if arch not in MODELS:
        raise ValueError(f"unknown architecture: {arch}")
    torch.manual_seed(seed)
    model = SmallCNN(w=w) if arch == "cnn" else MODELS[arch]()
    return model.to(DEV).to(memory_format=torch.channels_last)


# ---------------------------------------------------------------- train / eval

def _batches(n, bs, gen):
    perm = torch.randperm(n, device=DEV, generator=gen)
    for i in range(0, n - bs + 1, bs):
        yield perm[i:i + bs]


def fit(model, X, y, epochs, lr, seed, bs=512, flip=True):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    steps = max(1, epochs * (len(X) // bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr * 2, total_steps=steps)
    model.train()
    for _ in range(epochs):
        for idx in _batches(len(X), bs, gen):
            xb = X[idx]
            if flip:
                m = torch.rand(len(idx), 1, 1, 1, device=DEV, generator=gen) < 0.5
                xb = torch.where(m, xb.flip(-1), xb)
            with torch.autocast(DEV, torch.bfloat16):
                logit = model(xb)
            loss = F.cross_entropy(logit.float(), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return model


@torch.no_grad()
def accuracy(model, X, y, bs=1000):
    model.eval()
    c = 0
    for i in range(0, len(X), bs):
        with torch.autocast(DEV, torch.bfloat16):
            c += (model(X[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return c / len(X)


# ---------------------------------------------------------------- operations

# Calibration roots 900--902 mean eval_acc: null .85450, sft .85146,
# unlearn .85288, prune .85579, quant .85625.

def op_null(m, S, seed, epochs=2):
    """Continued benign training on the same distribution (control)."""
    X, y = S["op"]
    return fit(copy.deepcopy(m), X, y, epochs=epochs, lr=0.02, seed=seed)


def op_sft(m, S, seed, multiplier=0.75):
    """Fine-tune on a shifted version of the operation split."""
    X, y = S["op"]
    g = torch.Generator(device=DEV).manual_seed(seed)
    gray = X.mean(1, keepdim=True).expand_as(X)
    a = (torch.rand(len(X), 1, 1, 1, device=DEV, generator=g) * 0.6 + 0.2) * multiplier
    Xs = (1 - a) * X + a * gray
    Xs = Xs * (1 + 0.3 * multiplier * torch.randn(
        len(X), 3, 1, 1, device=DEV, generator=g))
    Xs = Xs.contiguous(memory_format=torch.channels_last)
    return fit(copy.deepcopy(m), Xs, y, epochs=2, lr=0.02, seed=seed)


def op_unlearn(m, S, seed, n_forget=500, coefficient=0.1):
    """Sample-level gradient-difference unlearning.

    Forgets a random subset rather than a whole class, so test behaviour barely
    moves: exactly the behaviourally-matched case of plan 11.3.
    """
    X, y = S["op"]
    g = torch.Generator(device=DEV).manual_seed(seed)
    idx = torch.randperm(len(X), device=DEV, generator=g)
    f_idx, r_idx = idx[:n_forget], idx[n_forget:]
    mu = copy.deepcopy(m)
    opt = torch.optim.SGD(mu.parameters(), lr=0.005, momentum=0.9)
    mu.train()
    for _ in range(60):
        rb = r_idx[torch.randint(len(r_idx), (256,), device=DEV, generator=g)]
        with torch.autocast(DEV, torch.bfloat16):
            l_ret = F.cross_entropy(mu(X[rb]).float(), y[rb])
            l_for = F.cross_entropy(mu(X[f_idx]).float(), y[f_idx])
        loss = l_ret - coefficient * torch.clamp(l_for, max=4.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return mu


def op_prune(m, S, seed, ratio=0.5):
    """Global magnitude pruning + short recovery fine-tune."""
    mp = copy.deepcopy(m)
    ws = [p for _, p in mp.named_parameters() if p.dim() > 1]
    allw = torch.cat([p.detach().abs().flatten() for p in ws]).float()
    thr = torch.quantile(allw[torch.randperm(len(allw), device=DEV)[:1_000_000]], ratio)
    with torch.no_grad():
        for p in ws:
            p.mul_((p.abs() > thr).float())
    X, y = S["op"]
    return fit(mp, X, y, epochs=1, lr=0.01, seed=seed)


def op_quant(m, S, seed, bits=8):
    """Symmetric per-channel fake quantisation, dequantised back to fp32.

    Storing dequantised weights removes the dtype/metadata cue, so a detector
    must read the quantisation residual rather than the file format.
    """
    mq = copy.deepcopy(m)
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for _, p in mq.named_parameters():
            if p.dim() > 1:
                s = (p.abs().flatten(1).max(1).values.clamp(min=1e-8) / qmax)
                s = s.view(-1, *([1] * (p.dim() - 1)))
                p.copy_(torch.round(p / s).clamp(-qmax, qmax) * s)
    return mq


def op_distill(m, S, seed):
    """Architecture-matched self-distillation with no hard-label loss."""
    X, _ = S["op"]
    teacher = m.eval()
    student = copy.deepcopy(m).train()
    gen = torch.Generator(device=DEV).manual_seed(seed)
    opt = torch.optim.SGD(student.parameters(), lr=0.02, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    steps = max(1, 2 * (len(X) // 512))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.04, total_steps=steps)
    temperature = 4
    for _ in range(2):
        for idx in _batches(len(X), 512, gen):
            xb = X[idx]
            flip = torch.rand(len(idx), 1, 1, 1, device=DEV, generator=gen) < 0.5
            xb = torch.where(flip, xb.flip(-1), xb)
            with torch.no_grad(), torch.autocast(DEV, torch.bfloat16):
                teacher_log_prob = F.log_softmax(teacher(xb).float() / temperature, dim=1)
            with torch.autocast(DEV, torch.bfloat16):
                student_log_prob = F.log_softmax(student(xb).float() / temperature, dim=1)
            loss = (student_log_prob.exp() *
                    (student_log_prob - teacher_log_prob)).sum(1).mean() * temperature ** 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return student


@torch.no_grad()
def op_merge(m, S, seed):
    """Average a root with a fine-tuned descendant, then rebuild BN statistics."""
    X, y = S["op"]
    with torch.enable_grad():
        descendant = fit(copy.deepcopy(m), X, y, epochs=2, lr=0.02, seed=seed + 1)
    merged = copy.deepcopy(m)
    averaged = {
        name: ((tensor.float() + descendant.state_dict()[name].float()) * 0.5).to(tensor.dtype)
        for name, tensor in m.state_dict().items()
    }
    merged.load_state_dict(averaged)
    bn_momenta = {}
    for module in merged.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            bn_momenta[module] = module.momentum
            module.momentum = None
    merged.train()
    for start in range(0, len(X), 512):
        with torch.autocast(DEV, torch.bfloat16):
            merged(X[start:start + 512])
    for module, momentum in bn_momenta.items():
        module.momentum = momentum
    return merged


def laundry_ft(m, S, seed):
    X, y = S["op"]
    return fit(copy.deepcopy(m), X, y, epochs=2, lr=0.02, seed=seed)


def laundry_quant(m, S, seed):
    return op_quant(m, S, seed)


def laundry_noise(m, S, seed):
    noisy = copy.deepcopy(m)
    generator = torch.Generator(device=DEV).manual_seed(seed)
    with torch.no_grad():
        for tensor in noisy.state_dict().values():
            if tensor.dim() > 1:
                sigma = 0.02 * tensor.float().std(unbiased=False)
                tensor.add_(torch.randn(tensor.shape, device=tensor.device,
                                        dtype=tensor.dtype, generator=generator) * sigma)
    return noisy


OPERATIONS = {"null": op_null, "sft": op_sft, "unlearn": op_unlearn,
              "prune": op_prune, "quant": op_quant}
INTENSITIES = {"null": (1, 2, 3), "quant": (4, 5, 6, 8)}

UNKNOWN_OPERATIONS = {"distill": op_distill, "merge": op_merge}
LAUNDRIES = {"ft": laundry_ft, "quant": laundry_quant, "noise": laundry_noise}

PAIRS = (("sft", "unlearn"), ("unlearn", "sft"),
         ("sft", "prune"), ("prune", "sft"),
         ("sft", "quant"), ("quant", "sft"))


# ---------------------------------------------------------------- generation

def build_zoo(out_dir, n_roots=40, root_epochs=8, root_lr=0.2, seed0=0):
    out_dir = Path(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    S = load_splits(seed=seed0)
    Xr, yr = S["root"]
    Xe, ye = S["eval"]
    manifest = []
    t_start = time.time()

    for r in range(n_roots):
        rseed = 1000 + r
        root = fit(new_model(rseed), Xr, yr, epochs=root_epochs, lr=root_lr, seed=rseed)
        entries = [("root", root, [])]
        for i, (op, fn) in enumerate(OPERATIONS.items()):
            entries.append((op, fn(root, S, rseed * 10 + i), [op]))

        accs = {}
        for name, model, history in entries:
            mid = f"root{r:03d}_{name}"
            sd = {k: v.half().cpu() for k, v in model.state_dict().items()}
            torch.save(sd, out_dir / "models" / f"{mid}.pt")
            accs[name] = round(accuracy(model, Xe, ye), 3)
            manifest.append(dict(
                model_id=mid, root_group=f"root{r:03d}", root_seed=rseed,
                history=history, label=name, eval_acc=accs[name],
                artifact_hash=hashlib.sha256(
                    b"".join(v.numpy().tobytes() for v in sd.values())).hexdigest()[:16],
            ))
        print(f"  root {r + 1}/{n_roots}  {time.time() - t_start:6.1f}s  {accs}", flush=True)

    with open(out_dir / "manifest.jsonl", "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    print(f"zoo: {len(manifest)} models in {time.time() - t_start:.1f}s -> {out_dir}")
    return manifest


def _draw_intensity(operation, seed):
    generator = torch.Generator().manual_seed(seed)
    if operation in INTENSITIES:
        choices = INTENSITIES[operation]
        return choices[torch.randint(len(choices), (), generator=generator).item()]
    unit = torch.rand((), generator=generator).item()
    if operation == "unlearn":
        return math.exp(math.log(0.04) + unit * (math.log(0.25) - math.log(0.04)))
    low, high = {"sft": (0.4, 1.1), "prune": (0.25, 0.75)}[operation]
    return low + unit * (high - low)


def _apply_intensity(operation, model, splits, seed, intensity):
    kwargs = {
        "null": {"epochs": intensity},
        "sft": {"multiplier": intensity},
        "unlearn": {"coefficient": intensity},
        "prune": {"ratio": intensity},
        "quant": {"bits": intensity},
    }
    return OPERATIONS[operation](model, splits, seed, **kwargs[operation])


def build_grid(out_dir, n_roots_per_cell=20, root_epochs=8, root_lr=0.2):
    """Build the dataset x architecture grid without touching earlier runs."""
    out_dir = Path(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    manifest = []
    started = time.time()
    architectures = tuple(MODELS)
    counts = {}
    for arch in architectures:
        model = new_model(0, arch=arch)
        counts[arch] = sum(parameter.numel() for parameter in model.parameters())
        del model
    print("parameter counts:", " ".join(f"{arch}={counts[arch]:,}" for arch in architectures))
    cells = [(dataset, arch) for dataset in ("cifar10", "svhn")
             for arch in architectures]

    for cell_index, (dataset, arch) in enumerate(cells):
        splits = load_splits(seed=0, dataset=dataset)
        Xr, yr = splits["root"]
        Xe, ye = splits["eval"]
        for root_index in range(n_roots_per_cell):
            root_seed = 2000 + 100 * cell_index + root_index
            root_group = f"{dataset}_{arch}_root{root_index:03d}"
            # VGG/depthwise diverged on SVHN at 0.2; one fixed lower LR is sufficient.
            arch_lr = 0.1 if arch in ("vgg", "depthwise") else root_lr
            root = fit(new_model(root_seed, arch=arch), Xr, yr, epochs=root_epochs,
                       lr=arch_lr, seed=root_seed)
            entries = [("root", root, None, 0)]
            for op_index, operation in enumerate(OPERATIONS):
                operation_seed = root_seed * 10 + op_index
                intensity = _draw_intensity(operation, operation_seed)
                n_replicates = 3 if dataset == "cifar10" and arch == "cnn" else 1
                for replicate in range(n_replicates):
                    replicate_seed = operation_seed + replicate * 100_000
                    torch.manual_seed(replicate_seed)
                    entries.append((operation, _apply_intensity(
                        operation, root, splits, replicate_seed, intensity),
                        intensity, replicate))

            accs = {}
            for label, model, intensity, replicate in entries:
                suffix = "" if replicate == 0 else f"_rep{replicate}"
                model_id = f"{root_group}_{label}{suffix}"
                artifact_hash = _save_model(model, out_dir / "models" / f"{model_id}.pt")
                accs[f"{label}{suffix}"] = round(accuracy(model, Xe, ye), 3)
                manifest.append(dict(
                    model_id=model_id, root_group=root_group, root_seed=root_seed,
                    history=[] if label == "root" else [label], label=label,
                    eval_acc=accs[f"{label}{suffix}"], artifact_hash=artifact_hash,
                    arch=arch, dataset=dataset, intensity=intensity, replicate=replicate,
                ))
            del entries, root
            print(f"  {dataset}/{arch} root {root_index + 1}/{n_roots_per_cell}  "
                  f"{time.time() - started:6.1f}s  {accs}", flush=True)
        elapsed = time.time() - started
        completed = cell_index + 1
        eta = elapsed / completed * (len(cells) - completed)
        print(f"  cell {completed}/{len(cells)} complete; ETA {eta / 60:.1f} min", flush=True)
        del splits, Xr, yr, Xe, ye
        torch.cuda.empty_cache()

    with open(out_dir / "manifest.jsonl", "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")
    print(f"grid: {len(manifest)} models in {time.time() - started:.1f}s -> {out_dir}")
    return manifest


def build_pairs(out_dir, roots_dir, n_roots=40):
    out_dir, roots_dir = Path(out_dir), Path(roots_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    with open(roots_dir / "manifest.jsonl") as f:
        roots = {entry["root_group"]: entry for line in f if line.strip()
                 for entry in (json.loads(line),) if entry["label"] == "root"}

    S = load_splits(seed=0)
    Xe, ye = S["eval"]
    manifest = []
    t_start = time.time()
    for r in range(n_roots):
        root_group = f"root{r:03d}"
        source = roots_dir / "models" / f"{root_group}_root.pt"
        if root_group not in roots or not source.exists():
            raise FileNotFoundError(f"missing root checkpoint or manifest row: {source}")
        root_entry = roots[root_group]
        rseed = root_entry["root_seed"]
        state = torch.load(source, map_location="cpu", weights_only=True)
        root = SmallCNN().to(DEV, memory_format=torch.channels_last)
        root.load_state_dict({name: tensor.float() for name, tensor in state.items()})
        shutil.copy2(source, out_dir / "models" / source.name)

        root_acc = round(accuracy(root, Xe, ye), 3)
        manifest.append(dict(
            model_id=f"{root_group}_root", root_group=root_group, root_seed=rseed,
            history=[], label="root", eval_acc=root_acc,
            artifact_hash=hashlib.sha256(
                b"".join(v.numpy().tobytes() for v in state.values())).hexdigest()[:16],
        ))
        accs = {"root": root_acc}
        for pair_index, (op1, op2) in enumerate(PAIRS):
            seed1 = rseed * 100 + pair_index * 10
            torch.manual_seed(seed1)
            model = OPERATIONS[op1](root, S, seed1)
            torch.manual_seed(seed1 + 1)
            model = OPERATIONS[op2](model, S, seed1 + 1)
            label = f"{op1}-{op2}"
            mid = f"{root_group}_{label}"
            sd = {name: tensor.half().cpu() for name, tensor in model.state_dict().items()}
            torch.save(sd, out_dir / "models" / f"{mid}.pt")
            accs[label] = round(accuracy(model, Xe, ye), 3)
            manifest.append(dict(
                model_id=mid, root_group=root_group, root_seed=rseed,
                history=[op1, op2], label=label, eval_acc=accs[label],
                artifact_hash=hashlib.sha256(
                    b"".join(v.numpy().tobytes() for v in sd.values())).hexdigest()[:16],
            ))
            del model
        print(f"  root {r + 1}/{n_roots}  {time.time() - t_start:6.1f}s  {accs}", flush=True)

    with open(out_dir / "manifest.jsonl", "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")
    print(f"pairs: {len(manifest)} models in {time.time() - t_start:.1f}s -> {out_dir}")
    return manifest


def _load_source(source_dir):
    source_dir = Path(source_dir)
    with open(source_dir / "manifest.jsonl") as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    return source_dir, manifest


def _load_model(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = SmallCNN().to(DEV, memory_format=torch.channels_last)
    model.load_state_dict({name: tensor.float() for name, tensor in state.items()})
    return model, state


def _save_model(model, path):
    state = {name: tensor.half().cpu() for name, tensor in model.state_dict().items()}
    torch.save(state, path)
    digest = hashlib.sha256(
        b"".join(tensor.numpy().tobytes() for tensor in state.values())).hexdigest()[:16]
    return digest


def build_unknowns(out_dir, roots_dir, n_roots=40):
    out_dir = Path(out_dir)
    roots_dir, source_manifest = _load_source(roots_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    roots = {entry["root_group"]: entry for entry in source_manifest
             if entry["label"] == "root"}
    S = load_splits(seed=0)
    Xe, ye = S["eval"]
    manifest = []
    started = time.time()

    for root_index in range(n_roots):
        root_group = f"root{root_index:03d}"
        source = roots_dir / "models" / f"{root_group}_root.pt"
        if root_group not in roots or not source.exists():
            raise FileNotFoundError(f"missing root checkpoint or manifest row: {source}")
        root_entry = roots[root_group]
        root, _ = _load_model(source)
        shutil.copy2(source, out_dir / "models" / source.name)
        manifest.append(dict(root_entry))
        accs = {"root": root_entry["eval_acc"]}

        for op_index, (label, operation) in enumerate(UNKNOWN_OPERATIONS.items()):
            seed = root_entry["root_seed"] * 10 + op_index
            model = operation(root, S, seed)
            model_id = f"{root_group}_{label}"
            artifact_hash = _save_model(model, out_dir / "models" / f"{model_id}.pt")
            accs[label] = round(accuracy(model, Xe, ye), 3)
            manifest.append(dict(
                model_id=model_id, root_group=root_group,
                root_seed=root_entry["root_seed"], history=[label], label=label,
                eval_acc=accs[label], artifact_hash=artifact_hash,
            ))
            del model
        print(f"  root {root_index + 1}/{n_roots}  {time.time() - started:6.1f}s  "
              f"{accs}", flush=True)

    with open(out_dir / "manifest.jsonl", "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")
    print(f"unknowns: {len(manifest)} models in {time.time() - started:.1f}s -> {out_dir}")
    return manifest


def build_laundered(out_dir, source_dir, n_roots=40):
    out_dir = Path(out_dir)
    source_dir, source_manifest = _load_source(source_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    selected_groups = {f"root{index:03d}" for index in range(n_roots)}
    selected = [entry for entry in source_manifest if entry["root_group"] in selected_groups]
    roots = {entry["root_group"]: entry for entry in selected if entry["label"] == "root"}
    if set(roots) != selected_groups:
        missing = sorted(selected_groups - set(roots))
        raise FileNotFoundError(f"missing root manifest rows: {missing}")

    S = load_splits(seed=0)
    Xe, ye = S["eval"]
    manifest = []
    started = time.time()
    for root_index in range(n_roots):
        root_group = f"root{root_index:03d}"
        root_entry = roots[root_group]
        root_source = source_dir / "models" / f"{root_entry['model_id']}.pt"
        if not root_source.exists():
            raise FileNotFoundError(root_source)
        shutil.copy2(root_source, out_dir / "models" / root_source.name)
        manifest.append(dict(root_entry, laundry=None))

        originals = [entry for entry in selected
                     if entry["root_group"] == root_group and entry["label"] != "root"]
        if len(originals) != len(OPERATIONS):
            raise ValueError(f"expected {len(OPERATIONS)} non-root models for {root_group}")
        for model_index, original in enumerate(originals):
            source = source_dir / "models" / f"{original['model_id']}.pt"
            if not source.exists():
                raise FileNotFoundError(source)
            original_model, _ = _load_model(source)
            for laundry_index, (laundry, transform) in enumerate(LAUNDRIES.items()):
                seed = original["root_seed"] * 100 + model_index * 10 + laundry_index
                model = transform(original_model, S, seed)
                model_id = f"{original['model_id']}__{laundry}"
                artifact_hash = _save_model(model, out_dir / "models" / f"{model_id}.pt")
                manifest.append(dict(
                    model_id=model_id, root_group=original["root_group"],
                    root_seed=original["root_seed"], history=original["history"],
                    label=original["label"], eval_acc=round(accuracy(model, Xe, ye), 3),
                    artifact_hash=artifact_hash, laundry=laundry,
                ))
                del model
            del original_model
        print(f"  root {root_index + 1}/{n_roots}  {time.time() - started:6.1f}s", flush=True)

    with open(out_dir / "manifest.jsonl", "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")
    print(f"laundered: {len(manifest)} models in {time.time() - started:.1f}s -> {out_dir}")
    return manifest
