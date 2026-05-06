# SPDX-FileCopyrightText: 2026 Sridhar M.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Code-DCR: Continuous Compiler Relaxation for FEniCS code correction.
# Supplementary reference implementation (arXiv / journal reproduction).
# Full license text: LICENSE (GNU AGPL-3.0 or later).
# Progress / metrics logs: set environment variable CODECINO_VERBOSE=1.

import os
import ast
import re
import random
import warnings
from collections import defaultdict

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report

CODECINO_VERBOSE = os.environ.get("CODECINO_VERBOSE", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _log(*args, **kwargs):
    if CODECINO_VERBOSE:
        print(*args, **kwargs)


SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = device.type == "cuda"
_log(f"Device: {device} | AMP: {use_amp}")
FENICS_ALIASES = {'create_unit_square': 'Mesh', 'create_unit_interval': 'Mesh', 'create_unit_cube': 'Mesh', 'create_box': 'Mesh', 'create_rectangle': 'Mesh', 'UnitSquareMesh': 'Mesh', 'UnitCubeMesh': 'Mesh', 'RectangleMesh': 'Mesh', 'dirichletbc': 'DirichletBC', 'functionspace': 'FunctionSpace', 'LinearProblem': 'LinearProblem', 'NonlinearProblem': 'NonlinearProblem', 'petsc': 'LinearProblem', 'solve': 'LinearProblem'}
VOCAB = {'Unknown': 0, 'Mesh': 1, 'FunctionSpace': 2, 'TrialFunction': 3, 'TestFunction': 4, 'Function': 5, 'DirichletBC': 6, 'LinearProblem': 7, 'NonlinearProblem': 8}
VOCAB_SIZE = len(VOCAB)
INV_VOCAB = {v: k for k, v in VOCAB.items()}
CORRECT_ORDER = ['Mesh', 'FunctionSpace', 'TrialFunction', 'TestFunction', 'Function', 'DirichletBC', 'LinearProblem', 'NonlinearProblem']

class DeepTreeParser(ast.NodeVisitor):

    def __init__(self):
        self.vocab = VOCAB
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.parent_stack = []

    def add_node(self, node_label):
        node_id = self.node_counter
        self.graph.add_node(node_id, x=self.vocab.get(node_label, 0))
        if self.parent_stack:
            self.graph.add_edge(self.parent_stack[-1], node_id)
        self.node_counter += 1
        return node_id

    def generic_visit(self, node):
        name = type(node).__name__
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                name = FENICS_ALIASES.get(node.func.attr, node.func.attr)
            elif isinstance(node.func, ast.Name):
                name = FENICS_ALIASES.get(node.func.id, node.func.id)
        nid = self.add_node(name)
        self.parent_stack.append(nid)
        super().generic_visit(node)
        self.parent_stack.pop()

def parse_to_pyg(code_string):
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        return None
    parser = DeepTreeParser()
    parser.visit(tree)
    if parser.graph.number_of_nodes() == 0:
        return None
    pyg_data = from_networkx(parser.graph)
    pyg_data.x = F.one_hot(torch.tensor([parser.graph.nodes[n]['x'] for n in sorted(parser.graph.nodes)]), num_classes=VOCAB_SIZE).float()
    return pyg_data
DEMO_FILES = ['demo_axis.py', 'demo_biharmonic.py', 'demo_cahn-hilliard.py', 'demo_comm-pattern.py', 'demo_elasticity.py', 'demo_gmsh.py', 'demo_half_loaded_waveguide.py', 'demo_hdg.py', 'demo_helmholtz.py', 'demo_interpolation-io.py', 'demo_lagrange_variants.py', 'demo_matrix_free_petsc.py', 'demo_mixed-poisson.py', 'demo_mixed-topology.py', 'demo_navier-stokes.py', 'demo_pml.py', 'demo_poisson.py', 'demo_poisson_matrix_free.py', 'demo_pyamg.py', 'demo_pyvista.py', 'demo_scattering_boundary_conditions.py', 'demo_static-condensation.py', 'demo_tnt-elements.py', 'demo_types.py']
BASE_URL = 'https://raw.githubusercontent.com/FEniCS/dolfinx/main/python/demo/'
CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CODE_ROOT, 'data', 'dolfinx_demos')
os.makedirs(DATA_DIR, exist_ok=True)
dataset = []
_log(f"Loading FEniCS demos (cache: {DATA_DIR})...")
for fname in DEMO_FILES:
    local_path = os.path.join(DATA_DIR, fname)
    try:
        code_text = None
        if os.path.exists(local_path):
            with open(local_path, 'r', encoding='utf-8') as f:
                code_text = f.read()
        else:
            r = requests.get(BASE_URL + fname, timeout=10)
            if r.status_code == 200:
                code_text = r.text
                with open(local_path, 'w', encoding='utf-8') as f:
                    f.write(code_text)
        if code_text is not None:
            g = parse_to_pyg(code_text)
            if g is not None:
                dataset.append(g)
                _log(f"  loaded {fname} ({g.x.size(0)} nodes)")
            else:
                warnings.warn(f"parse failed: {fname}", UserWarning, stacklevel=1)
        else:
            warnings.warn(f"download failed: {fname}", UserWarning, stacklevel=1)
    except Exception as exc:
        warnings.warn(f"skipped {fname}: {exc}", UserWarning, stacklevel=1)
_log(f"Loaded {len(dataset)} graphs.")
assert len(dataset) >= 22, f'Only {len(dataset)} demos loaded — need ≥22 for train/test split.'

class NeuroSymbolicGNN(nn.Module):

    def __init__(self, num_features, hidden_dim, num_classes, heads=4):
        super().__init__()
        self.gat1 = GATConv(num_features, hidden_dim, heads=heads, dropout=0.1)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.1)
        self.gat3 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.1)
        self.gat4 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=0.1)
        self.gat5 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=0.1)
        self.prediction_head = nn.Linear(hidden_dim + num_features, num_classes)
        self.correction_gate = GATConv(hidden_dim + 8, num_classes, heads=1, concat=False)
        self.error_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, data, tau=1.0, hard=False, error_signal=None, pass1_logits=None):
        x_orig = data.x.to(torch.float32)
        ei = data.edge_index
        h = F.elu(self.gat1(x_orig, ei))
        h = F.elu(self.gat2(h, ei))
        h = F.elu(self.gat3(h, ei))
        h = F.elu(self.gat4(h, ei))
        h_emb = F.elu(self.gat5(h, ei))
        if error_signal is None:
            logits = self.prediction_head(torch.cat([h_emb, x_orig], dim=-1))
        else:
            if error_signal.size(0) != h_emb.size(0):
                error_signal = error_signal.expand(h_emb.size(0), -1)
            logits = self.correction_gate(torch.cat([h_emb, error_signal.to(torch.float32)], dim=-1), ei)
        soft_tokens = F.gumbel_softmax(logits, tau=tau, hard=hard)
        error_type_log = self.error_head(h_emb.mean(dim=0))
        return (soft_tokens, error_type_log, logits)
NUM_STATES = 8
STATE_INIT, STATE_MESH, STATE_FS, STATE_TRIAL = (0, 1, 2, 3)
STATE_TEST, STATE_BC, STATE_SOLVED, STATE_ERROR = (4, 5, 6, 7)

def _execution_loss_single(state_trace, epoch, lambda_physics=10.0):
    sm, sfs = (state_trace[:, STATE_MESH], state_trace[:, STATE_FS])
    st, ste = (state_trace[:, STATE_TRIAL], state_trace[:, STATE_TEST])
    sbc = state_trace[:, STATE_BC]
    topo = torch.tensor(0.0, device=state_trace.device)
    if state_trace.size(0) > 1:

        def ncs(x):
            cs = torch.cumsum(x, dim=0)
            return cs / (cs.max() + 1e-08)
        cm, cfs = (ncs(sm), ncs(sfs))
        topo += torch.relu(sfs[1:] - cm[:-1]).mean()
        topo += torch.relu(st[1:] - cfs[:-1]).mean()
        topo += torch.relu(ste[1:] - cfs[:-1]).mean()
        topo += torch.relu(sbc[1:] - cfs[:-1]).mean()
    err = state_trace[:, STATE_ERROR].mean()
    reg = 0.001 * 0.99 ** epoch * torch.norm(state_trace)
    return (topo + err) * lambda_physics + reg

def execution_loss(state_trace, predictions, data, epoch=0, lambda_physics=10.0):
    batch_index = data.batch if hasattr(data, 'batch') else None
    if batch_index is None:
        return _execution_loss_single(state_trace, epoch, lambda_physics)
    n = batch_index.max().item() + 1
    total = torch.tensor(0.0, device=state_trace.device)
    for g in range(n):
        total += _execution_loss_single(state_trace[batch_index == g], epoch, lambda_physics)
    return total / n

class DifferentiableFEniCSStateMachine(nn.Module):

    def __init__(self, vocab_size=VOCAB_SIZE, num_states=NUM_STATES):
        super().__init__()
        T = torch.zeros(vocab_size, num_states, num_states)
        T[:, STATE_INIT, STATE_INIT] = 2.0
        T[:, STATE_ERROR, STATE_ERROR] = 2.0
        T[1, STATE_INIT, STATE_MESH] = 4.0
        T[2, STATE_MESH, STATE_FS] = 4.0
        T[2, STATE_INIT, STATE_ERROR] = 4.0
        T[3, STATE_FS, STATE_TRIAL] = 4.0
        T[4, STATE_FS, STATE_TEST] = 4.0
        T[5, STATE_FS, STATE_TEST] = 4.0
        T[5, STATE_TEST, STATE_TEST] = 4.0
        T[6, STATE_TEST, STATE_BC] = 4.0
        T[7, STATE_BC, STATE_SOLVED] = 5.0
        T[8, STATE_BC, STATE_SOLVED] = 5.0
        for s in range(STATE_INIT, STATE_BC):
            T[7, s, STATE_ERROR] = 5.0
            T[8, s, STATE_ERROR] = 5.0
        for s in range(STATE_INIT, STATE_TEST):
            T[6, s, STATE_ERROR] = 5.0
        for s in [STATE_INIT, STATE_MESH]:
            T[3, s, STATE_ERROR] = 5.0
            T[4, s, STATE_ERROR] = 5.0
            T[5, s, STATE_ERROR] = 5.0
        self.transition_logits = nn.Parameter(T)

    def forward(self, soft_tokens, batch_index=None):
        soft_tokens = soft_tokens.to(torch.float32)
        if batch_index is not None:
            dense_tokens, mask = to_dense_batch(soft_tokens, batch_index)
        else:
            dense_tokens = soft_tokens.unsqueeze(0)
            mask = torch.ones(1, soft_tokens.size(0), dtype=torch.bool, device=soft_tokens.device)
        B, M, V = dense_tokens.size()
        T_soft = F.softmax(self.transition_logits, dim=-1)
        T_tok = torch.einsum('bmv,vst->bmst', dense_tokens, T_soft)
        state = torch.zeros(B, 1, NUM_STATES, device=soft_tokens.device)
        state[:, 0, STATE_INIT] = 1.0
        all_states = []
        for i in range(M):
            state = F.softmax(torch.bmm(state, T_tok[:, i]), dim=-1)
            all_states.append(state)
        trajectory = torch.cat(all_states, dim=1)[mask]
        if batch_index is not None:
            dense_traj, tmask = to_dense_batch(trajectory, batch_index)
            last_idx = tmask.sum(dim=1) - 1
            final_states = dense_traj[torch.arange(dense_traj.size(0), device=trajectory.device), last_idx]
        else:
            final_states = trajectory[-1:, :]
        energy = (1.0 - final_states[:, STATE_SOLVED]).mean() + trajectory[:, STATE_ERROR].mean() * 5.0
        return (energy, trajectory)

def make_broken_variants(data):
    gt_labels = data.x.argmax(dim=-1).squeeze()
    variants = [(data, gt_labels)]
    for token_to_remove in [1, 2, 6, 7]:
        broken = data.clone()
        mask = broken.x.argmax(dim=-1) == token_to_remove
        if mask.any():
            broken.x[mask] = 0.0
            broken.x[mask, 0] = 1.0
            variants.append((broken, gt_labels))
    return variants
train_set = dataset[:20]
test_set = dataset[20:]
augmented_train_set = []
for raw_data in train_set:
    for data, gt_labels in make_broken_variants(raw_data):
        data.y = gt_labels.to(device)
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        augmented_train_set.append(data)
train_loader = DataLoader(augmented_train_set, batch_size=32, shuffle=True)
_log(
    f"Training: {len(augmented_train_set)} augmented graphs | Test: {len(test_set)} graphs"
)
model = NeuroSymbolicGNN(VOCAB_SIZE, 64, VOCAB_SIZE).to(device).to(torch.float32)
alm = DifferentiableFEniCSStateMachine().to(device).to(torch.float32)
opt_joint = torch.optim.Adam(list(model.parameters()) + list(alm.parameters()), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_joint, T_max=251, eta_min=1e-05)
scaler = amp.GradScaler() if use_amp else None
CLASS_WEIGHTS = torch.tensor([0.1, 15.0, 10.0, 8.0, 8.0, 5.0, 10.0, 8.0, 8.0], dtype=torch.float32).to(device)
NUM_EPOCHS = 301
_log(
    f"Starting training ({NUM_EPOCHS} epochs: 0-49 warmup, 50-300 joint)..."
)
for epoch in range(NUM_EPOCHS):
    if epoch == 50:
        _log("Warmup complete; switching to joint training.")
    tau = max(0.5, 2.0 * 0.95 ** epoch)
    total_loss = 0.0
    total_correct = 0
    total_nodes = 0
    model.train()
    alm.train()
    for batch in train_loader:
        batch.x = batch.x.to(torch.float32)
        if epoch < 50:
            opt_joint.zero_grad()
            sequence_loss = torch.tensor(0.0, device=device)
            num_graphs = batch.batch.max().item() + 1 if hasattr(batch, 'batch') else 1
            for g_idx in range(num_graphs):
                graph_y = batch.y[batch.batch == g_idx] if hasattr(batch, 'batch') else batch.y
                present = set((INV_VOCAB.get(l, 'Unknown') for l in graph_y.tolist() if INV_VOCAB.get(l, 'Unknown') != 'Unknown'))
                ordered_ref = [t for t in CORRECT_ORDER if t in present]
                if ordered_ref:
                    seq = torch.zeros(len(ordered_ref), VOCAB_SIZE, device=device)
                    for j, tok in enumerate(ordered_ref):
                        seq[j, VOCAB[tok]] = 1.0
                    _, seq_trace = alm(seq)
                    sequence_loss += 5.0 * (-seq_trace[-1, STATE_SOLVED] + seq_trace[:, STATE_ERROR].mean() * 3.0)
                    if len(ordered_ref) >= 3:
                        corrupted = ordered_ref.copy()
                        idx1, idx2 = random.sample(range(len(corrupted)), 2)
                        corrupted[idx1], corrupted[idx2] = (corrupted[idx2], corrupted[idx1])
                        seq_bad = torch.zeros(len(corrupted), VOCAB_SIZE, device=device)
                        for j, tok in enumerate(corrupted):
                            seq_bad[j, VOCAB[tok]] = 1.0
                        _, bad_trace = alm(seq_bad)
                        sequence_loss += 5.0 * (1.0 - bad_trace[:, STATE_ERROR].mean())
            sequence_loss = sequence_loss / num_graphs
            loss = sequence_loss
            loss.backward()
            opt_joint.step()
        else:
            opt_joint.zero_grad()
            sequence_loss = torch.tensor(0.0, device=device)
            with amp.autocast(device_type='cuda', enabled=use_amp):
                preds_p1, _, logits_p1 = model(batch, tau=tau)
                ce_loss_p1 = F.cross_entropy(logits_p1, batch.y, weight=CLASS_WEIGHTS)
                energy_p1, state_trace_p1 = alm(preds_p1, batch.batch)
                dense_s, s_mask = to_dense_batch(state_trace_p1, batch.batch)
                last_idx = s_mask.sum(dim=1) - 1
                terminal_error = dense_s[torch.arange(dense_s.size(0), device=device), last_idx]
                preds_p2, _, logits_p2 = model(batch, tau=tau, error_signal=terminal_error[batch.batch].detach(), pass1_logits=logits_p1.detach())
                ce_loss_p2 = F.cross_entropy(logits_p2, batch.y, weight=CLASS_WEIGHTS)
                energy_p2, state_trace_p2 = alm(preds_p2, batch.batch)
                correction_signal = F.relu(energy_p2 - energy_p1 + 0.1)
                phys = execution_loss(state_trace_p2, preds_p2, batch, epoch=epoch)
                num_graphs = batch.batch.max().item() + 1 if hasattr(batch, 'batch') else 1
                for g_idx in range(num_graphs):
                    graph_y = batch.y[batch.batch == g_idx] if hasattr(batch, 'batch') else batch.y
                    present = set((INV_VOCAB.get(l, 'Unknown') for l in graph_y.tolist() if INV_VOCAB.get(l, 'Unknown') != 'Unknown'))
                    ordered_ref = [t for t in CORRECT_ORDER if t in present]
                    if ordered_ref:
                        seq = torch.zeros(len(ordered_ref), VOCAB_SIZE, device=device)
                        for j, tok in enumerate(ordered_ref):
                            seq[j, VOCAB[tok]] = 1.0
                        _, seq_trace = alm(seq)
                        sequence_loss += 5.0 * (-seq_trace[-1, STATE_SOLVED] + seq_trace[:, STATE_ERROR].mean() * 3.0)
                        if len(ordered_ref) >= 3:
                            corrupted = ordered_ref.copy()
                            idx1, idx2 = random.sample(range(len(corrupted)), 2)
                            corrupted[idx1], corrupted[idx2] = (corrupted[idx2], corrupted[idx1])
                            seq_bad = torch.zeros(len(corrupted), VOCAB_SIZE, device=device)
                            for j, tok in enumerate(corrupted):
                                seq_bad[j, VOCAB[tok]] = 1.0
                            _, bad_trace = alm(seq_bad)
                            sequence_loss += 5.0 * (1.0 - bad_trace[:, STATE_ERROR].mean())
                sequence_loss = sequence_loss / num_graphs
                loss = ce_loss_p1 + ce_loss_p2 + 1.0 * energy_p2 + phys + correction_signal + sequence_loss.to(ce_loss_p1.dtype)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt_joint)
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(alm.parameters()), 1.0)
                scaler.step(opt_joint)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(alm.parameters()), 1.0)
                opt_joint.step()
            total_correct += (logits_p2.argmax(-1) == batch.y).sum().item()
            total_nodes += batch.y.size(0)
        total_loss += loss.item()
    if epoch >= 50:
        scheduler.step()
    if epoch % 100 == 0 or epoch == NUM_EPOCHS - 1:
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'alm': alm.state_dict()}, f'codecino_ep{epoch}.pt')
    if epoch % 10 == 0:
        acc = 100.0 * total_correct / total_nodes if total_nodes > 0 else 0.0
        mode = "Warmup" if epoch < 50 else "Joint"
        _log(
            f"Epoch {epoch:>3} | Loss: {total_loss / len(train_loader):.4f} "
            f"| Acc: {acc:.1f}% | tau={tau:.3f} | {mode}"
        )
_log("\n" + "=" * 60 + "\nEVALUATION\n" + "=" * 60)
model.eval()
alm.eval()
p1_preds, p2_preds, all_labels = ([], [], [])
with torch.no_grad():
    for td in test_set:
        td = td.to(device)
        td.x = td.x.to(torch.float32)
        preds_p1, _, logits_p1 = model(td, tau=0.5, hard=True)
        _, trace_p1 = alm(preds_p1)
        term_err = trace_p1[-1].unsqueeze(0).expand(preds_p1.size(0), -1)
        preds_p2, _, logits_p2 = model(td, tau=0.5, hard=True, error_signal=term_err, pass1_logits=logits_p1)
        gt = td.x.argmax(dim=-1).cpu().numpy()
        p1_preds.extend(logits_p1.argmax(-1).cpu().numpy())
        p2_preds.extend(logits_p2.argmax(-1).cpu().numpy())
        all_labels.extend(gt)
p1_acc = 100 * np.mean(np.array(p1_preds) == np.array(all_labels))
p2_acc = 100 * np.mean(np.array(p2_preds) == np.array(all_labels))
_log(f"\nPass 1 accuracy (blind):     {p1_acc:.2f}%")
_log(f"Pass 2 accuracy (corrected): {p2_acc:.2f}%")
_log(f"Correction gain:             {p2_acc - p1_acc:+.3f}%")
_log("\n=== Per-token recall ===")
token_correct = defaultdict(int)
token_total = defaultdict(int)
for pred, label in zip(p2_preds, all_labels):
    name = INV_VOCAB.get(label, 'Unknown')
    token_total[name] += 1
    if pred == label:
        token_correct[name] += 1
for tok in [
    "Mesh",
    "FunctionSpace",
    "TrialFunction",
    "TestFunction",
    "Function",
    "DirichletBC",
    "LinearProblem",
]:
    total = token_total[tok]
    correct = token_correct[tok]
    recall = 100 * correct / total if total > 0 else 0.0
    mark = "ok" if recall == 100 else ("partial" if recall > 0 else "miss")
    _log(f"  {mark} {tok:<20}: {correct}/{total}  recall={recall:.0f}%")
_log("\n=== Classification report ===")
labels_list = list(range(VOCAB_SIZE))
target_names = [INV_VOCAB[i] for i in range(VOCAB_SIZE)]
_log(
    classification_report(
        all_labels,
        p2_preds,
        labels=labels_list,
        target_names=target_names,
        zero_division=0,
    )
)
_log("\n" + "=" * 60 + "\nABLATION STUDY\n" + "=" * 60)
ablation_results = {}
configs = {'GNN only': {'physics': False, 'pass2': False}, 'GNN + Physics': {'physics': True, 'pass2': False}, 'Full Model (Ours)': {'physics': True, 'pass2': True}}
for name, cfg in configs.items():
    _log(f"\n  Config: {name}")
    m = NeuroSymbolicGNN(VOCAB_SIZE, 64, VOCAB_SIZE).to(device).to(torch.float32)
    a = DifferentiableFEniCSStateMachine().to(device).to(torch.float32)
    if cfg['pass2']:
        o_wu = torch.optim.Adam([a.transition_logits], lr=0.01)
        for ep in range(50):
            for batch in train_loader:
                o_wu.zero_grad()
                present = set((INV_VOCAB.get(l, 'Unknown') for l in batch.y.tolist() if INV_VOCAB.get(l, 'Unknown') != 'Unknown'))
                ordered = [t for t in CORRECT_ORDER if t in present]
                if not ordered:
                    continue
                seq = torch.zeros(len(ordered), VOCAB_SIZE, device=device)
                for j, tok in enumerate(ordered):
                    seq[j, VOCAB[tok]] = 1.0
                _, seq_trace = a(seq)
                loss = 5.0 * (-seq_trace[-1, STATE_SOLVED] + seq_trace[:, STATE_ERROR].mean() * 3.0)
                loss.backward()
                o_wu.step()
        _log("    Warmup complete.")
    o = torch.optim.Adam(list(m.parameters()) + list(a.parameters()), lr=0.001)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=150, eta_min=1e-05)
    for ep in range(150):
        m.train()
        a.train()
        for batch in train_loader:
            batch.x = batch.x.to(torch.float32)
            o.zero_grad()
            tau_abl = max(0.5, 2.0 * 0.95 ** ep)
            with amp.autocast(device_type='cuda', enabled=use_amp):
                preds_p1, _, logits_p1 = m(batch, tau=tau_abl)
                loss = F.cross_entropy(logits_p1, batch.y, weight=CLASS_WEIGHTS)
                if cfg['physics'] or cfg['pass2']:
                    energy_p1, state_trace_p1 = a(preds_p1, batch.batch)
                if cfg['physics']:
                    phys = execution_loss(state_trace_p1, preds_p1, batch, ep)
                    loss = loss + energy_p1 + phys
                if cfg['pass2']:
                    dense_s, s_mask = to_dense_batch(state_trace_p1, batch.batch)
                    last_idx = s_mask.sum(dim=1) - 1
                    term_err = dense_s[torch.arange(dense_s.size(0), device=device), last_idx][batch.batch].detach()
                    preds_p2, _, logits_p2 = m(batch, tau=tau_abl, error_signal=term_err, pass1_logits=logits_p1.detach())
                    energy_p2, _ = a(preds_p2, batch.batch)
                    correction_signal = F.relu(energy_p2 - energy_p1 + 0.1)
                    loss = loss + (F.cross_entropy(logits_p2, batch.y, weight=CLASS_WEIGHTS) + 0.1 * energy_p2 + correction_signal)
                    sequence_loss = torch.tensor(0.0, device=device)
                    num_graphs = batch.batch.max().item() + 1 if hasattr(batch, 'batch') else 1
                    for g_idx in range(num_graphs):
                        graph_y = batch.y[batch.batch == g_idx] if hasattr(batch, 'batch') else batch.y
                        present = set((INV_VOCAB.get(l, 'Unknown') for l in graph_y.tolist() if INV_VOCAB.get(l, 'Unknown') != 'Unknown'))
                        ordered_ref = [t for t in CORRECT_ORDER if t in present]
                        if ordered_ref:
                            seq = torch.zeros(len(ordered_ref), VOCAB_SIZE, device=device)
                            for j, tok in enumerate(ordered_ref):
                                seq[j, VOCAB[tok]] = 1.0
                            _, seq_trace = a(seq)
                            sequence_loss += 5.0 * (-seq_trace[-1, STATE_SOLVED] + seq_trace[:, STATE_ERROR].mean() * 3.0)
                    sequence_loss = sequence_loss / num_graphs
                    loss = loss + sequence_loss.to(loss.dtype)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(o)
                torch.nn.utils.clip_grad_norm_(list(m.parameters()) + list(a.parameters()), 1.0)
                scaler.step(o)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(m.parameters()) + list(a.parameters()), 1.0)
                o.step()
        sc.step()
    m.eval()
    a.eval()
    correct = total = 0
    with torch.no_grad():
        for td in test_set:
            td = td.to(device)
            td.x = td.x.to(torch.float32)
            p1_t, _, l1_t = m(td, tau=0.5, hard=True)
            if cfg['pass2']:
                _, st = a(p1_t)
                term = st[-1].unsqueeze(0).expand(p1_t.size(0), -1)
                _, _, l_t = m(td, tau=0.5, hard=True, error_signal=term, pass1_logits=l1_t)
            else:
                l_t = l1_t
            correct += (l_t.argmax(-1) == td.x.argmax(dim=-1)).sum().item()
            total += td.x.size(0)
    ablation_results[name] = 100.0 * correct / total
    _log(f"    {name}: {ablation_results[name]:.2f}%")
_log("\n=== Ablation table ===")
gnn_only = ablation_results.get('GNN only', 0)
for name, acc in ablation_results.items():
    delta = acc - gnn_only
    bar = "#" * int(acc / 5)
    _log(f"  {name:<25}: {acc:.2f}%  ({delta:+.2f}% vs GNN-only)  {bar}")
TEMPLATES = {'Mesh': 'mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)', 'FunctionSpace': "V = dolfinx.fem.FunctionSpace(mesh, ('Lagrange', 1))", 'TrialFunction': 'u = ufl.TrialFunction(V)', 'TestFunction': 'v = ufl.TestFunction(V)', 'Function': 'uh = dolfinx.fem.Function(V)', 'DirichletBC': 'bc = dolfinx.fem.dirichletbc(uh, facets, V)', 'LinearProblem': 'problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc])', 'NonlinearProblem': 'problem = NonlinearProblem(F, u, bcs=[bc])'}
ORDER_RULES = {'FunctionSpace': ['mesh'], 'TrialFunction': ['V'], 'TestFunction': ['V'], 'Function': ['V'], 'DirichletBC': ['V'], 'LinearProblem': ['bc'], 'NonlinearProblem': ['bc']}
TOKEN_DEPS = {'FunctionSpace': ['Mesh'], 'TrialFunction': ['FunctionSpace'], 'TestFunction': ['FunctionSpace'], 'Function': ['FunctionSpace'], 'DirichletBC': ['FunctionSpace'], 'LinearProblem': ['DirichletBC'], 'NonlinearProblem': ['DirichletBC']}

def validate_fenics_ordering(code_string):
    lines = code_string.split('\n')
    defined = set()
    for i, line in enumerate(lines):
        if '= dolfinx.mesh' in line or '= UnitSquareMesh' in line:
            defined.add('mesh')
        if re.match('^V\\s*=', line):
            defined.add('V')
        if re.match('^a\\s*=', line):
            defined.add('a')
        if re.match('^L\\s*=', line):
            defined.add('L')
        if re.match('^bc\\s*=', line):
            defined.add('bc')
        for obj, deps in ORDER_RULES.items():
            if obj in line and (not all((d in defined for d in deps))):
                return (False, f"Line {i + 1}: '{obj}' used before {[d for d in deps if d not in defined]}")
    return (True, 'Valid')

def unroll_state_machine(
    alm, gnn_preds=None, verbose=False, beam_width=3, heuristic_weight=0.2
):
    if isinstance(gnn_preds, tuple):
        gnn_preds = gnn_preds[0]
    available = set()
    if gnn_preds is not None:
        for idx in gnn_preds.argmax(dim=-1).cpu().tolist():
            name = INV_VOCAB.get(idx, 'Unknown')
            if name in TEMPLATES:
                available.add(name)
    if not available:
        available = set(TEMPLATES.keys())
    alm.eval()
    device = next(alm.parameters()).device
    available_list = sorted(list(available))
    target_len = len(available_list)
    progression_ladder = torch.tensor([0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, -10.0], device=device)
    beam = [([], 0.0)]
    with torch.no_grad():
        for step in range(target_len):
            new_candidates = []
            for seq, _ in beam:
                remaining_tokens = [t for t in available_list if t not in seq]
                for tok in remaining_tokens:
                    if not all((d in seq for d in TOKEN_DEPS.get(tok, []))):
                        continue
                    candidate_seq = seq + [tok]
                    seq_tensor = torch.zeros(len(candidate_seq), VOCAB_SIZE, device=device)
                    for j, t_name in enumerate(candidate_seq):
                        seq_tensor[j, VOCAB[t_name]] = 1.0
                    energy, trace = alm(seq_tensor)
                    final_state = trace[-1]
                    expected_progression = torch.dot(final_state, progression_ladder).item()
                    score = energy.item() - heuristic_weight * expected_progression
                    new_candidates.append((candidate_seq, score))
            new_candidates.sort(key=lambda x: x[1])
            beam = new_candidates[:beam_width]
    best_ordering, best_score = beam[0]
    if verbose:
        _log(f"  ALM ordering: {best_ordering} (A* score={best_score:.4f})")
    lines = ['import dolfinx', 'from mpi4py import MPI', 'import ufl']
    lines += [TEMPLATES[t] for t in best_ordering if t in TEMPLATES]
    return '\n'.join(lines)
_log("\n" + "=" * 60 + "\nTEST GRAPH GENERATION\n" + "=" * 60)
passed = 0
model.eval()
alm.eval()
for i, td in enumerate(test_set, 1):
    _log(f"\n  Test {i}/{len(test_set)}")
    td = td.to(device)
    td.x = td.x.to(torch.float32)
    with torch.no_grad():
        preds_p1, _, logits_p1 = model(td, tau=0.5, hard=True)
        _, trace_p1 = alm(preds_p1)
        term = trace_p1[-1].unsqueeze(0).expand(preds_p1.size(0), -1)
        preds_p2, _, _ = model(td, tau=0.5, hard=True, error_signal=term, pass1_logits=logits_p1)
    code = unroll_state_machine(
        alm, gnn_preds=preds_p2, verbose=CODECINO_VERBOSE
    )
    ok, reason = validate_fenics_ordering(code)
    _log(f"  {'VALID' if ok else 'FAILED: ' + reason}")
    if ok:
        passed += 1
_log(
    f"\nRESULT: {passed}/{len(test_set)} test graphs with valid FEniCS ordering"
)
_log("\n" + "=" * 60 + "\nOOD GENERALISATION\n" + "=" * 60)
ood_tests = [('demo_stokes.py', 'Linear Fluid Mechanics (Unseen OOD)', 'remote'), ('demo_test.py', 'Non-linear Synthetic Graph (Zero-shot)', 'local')]
for ood_file, ood_label, ood_source in ood_tests:
    _log(f"\n  OOD Case: {ood_label} [{ood_file}]")
    try:
        ood_text = None
        if ood_source == 'local':
            local_ood_path = os.path.join(DATA_DIR, ood_file)
            if os.path.exists(local_ood_path):
                with open(local_ood_path, 'r', encoding='utf-8') as f:
                    ood_text = f.read()
            else:
                warnings.warn(
                    f"Local OOD file not found: {local_ood_path}",
                    UserWarning,
                    stacklevel=1,
                )
                continue
        else:
            ood_r = requests.get(BASE_URL + ood_file, timeout=10)
            if ood_r.status_code == 200:
                ood_text = ood_r.text
            else:
                warnings.warn(
                    f"Could not fetch OOD demo (HTTP {ood_r.status_code})",
                    UserWarning,
                    stacklevel=1,
                )
                continue
        if ood_text is not None:
            ood_pyg = parse_to_pyg(ood_text)
            assert ood_pyg is not None, 'OOD parse failed'
            ood_pyg = ood_pyg.to(device)
            ood_pyg.x = ood_pyg.x.to(torch.float32)
            with torch.no_grad():
                ood_p1, _, ood_l1 = model(ood_pyg, tau=0.5, hard=True)
                _, ood_trace = alm(ood_p1)
                ood_term = ood_trace[-1].unsqueeze(0).expand(ood_p1.size(0), -1)
                ood_p2, _, _ = model(ood_pyg, tau=0.5, hard=True, error_signal=ood_term, pass1_logits=ood_l1)
            ood_code = unroll_state_machine(
                alm, gnn_preds=ood_p2, verbose=CODECINO_VERBOSE
            )
            ok, reason = validate_fenics_ordering(ood_code)
            _log(f"  {'OOD ok' if ok else 'OOD failed: ' + reason}")
            _log(f"\n--- Generated code ---\n{ood_code}\n{'-' * 40}")
    except Exception as e:
        warnings.warn(
            f"OOD test error for {ood_file}: {e}", UserWarning, stacklevel=1
        )
n_params = sum(p.numel() for p in model.parameters()) + sum(
    p.numel() for p in alm.parameters()
)
_log(f"\nTotal parameters: {n_params:,} (~{n_params / 1e6:.3f}M)")
