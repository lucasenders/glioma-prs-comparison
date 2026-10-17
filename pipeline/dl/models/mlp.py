"""Multilayer perceptron following the design of Badre et al. (2021), reduced to the size of AGOG."""

import copy
import time
import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Hidden layer widths of the main shape
HIDDEN = (256, 64, 16)

# Dropout after the first hidden layer, tuned, and after the others, fixed
DROPOUT = [0.2, 0.4, 0.6, 0.8]
DROPOUT_REST = 0.5

# Leaky ReLU slope, learning rate, batch size, maximum epochs and early-stopping patience
SLOPE = 0.2
LR = 1e-4
BATCH = 32
MAX_EPOCHS = 200
PATIENCE = 10

# Device to fit on
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    """Build a feed-forward network from the input through the hidden layers to one logit."""

    def __init__(self, input_size, hidden_sizes, dropout, pos_weight=1.0, seed=0,
                 act=nn.LeakyReLU(SLOPE)):
        # Set the seed, activation and case weight
        super().__init__()
        torch.manual_seed(seed)
        self.act = act
        self.seed = seed
        self.pos_weight = torch.tensor(float(pos_weight), device=DEVICE)
        sizes = [input_size] + list(hidden_sizes)

        # Build a linear, batch norm and dropout layer for each hidden layer
        self.linears = nn.ModuleList(nn.Linear(sizes[i], sizes[i + 1])
                                     for i in range(len(hidden_sizes)))
        self.norms = nn.ModuleList(nn.BatchNorm1d(n) for n in hidden_sizes)
        self.dropouts = nn.ModuleList(nn.Dropout(dropout if i == 0 else DROPOUT_REST)
                                      for i in range(len(hidden_sizes)))

        # Build the output layer, one logit per person
        self.output = nn.Linear(sizes[-1], 1)
        self.to(DEVICE)

    def forward(self, x):
        """Pass a batch through the network and return its logits."""

        for linear, norm, dropout in zip(self.linears, self.norms, self.dropouts):
            x = dropout(self.act(norm(linear(x))))
        return self.output(x).squeeze(1)

    def compute_loss(self, x, y):
        """Return the cross-entropy loss of a batch, with cases weighted to balance the classes."""

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        return loss_fn(self.forward(x), y)

    def fit(self, x, y, epochs=None, x_val=None, y_val=None):
        """Train with Adam, with early stopping on validation AUC or for a fixed number of epochs, and return the stopping epoch and best validation AUC."""

        # Set up the optimiser, the data and the batch order
        optimiser = torch.optim.Adam(self.parameters(), lr=LR)
        xt = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=DEVICE)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=DEVICE)
        order = torch.Generator(device="cpu").manual_seed(self.seed)

        best_auc, best_epoch, best_state, waited = -1.0, 0, None, 0
        limit = epochs or MAX_EPOCHS
        for epoch in range(1, limit + 1):

            # Train on the training set in shuffled batches
            self.train()
            for idx in torch.randperm(len(xt), generator=order).split(BATCH):
                if len(idx) < 2:
                    continue
                optimiser.zero_grad()
                self.compute_loss(xt[idx], yt[idx]).backward()
                optimiser.step()

            # Stop when the validation AUC has not improved for PATIENCE epochs
            if epochs is None:
                value = roc_auc_score(y_val, self.logits(x_val))
                if value > best_auc:
                    best_auc, best_epoch, waited = value, epoch, 0
                    best_state = copy.deepcopy(self.state_dict())
                else:
                    waited += 1
                    if waited >= PATIENCE:
                        break

        # Restore the best epoch's weights after early stopping
        if epochs is None:
            self.load_state_dict(best_state)
            return best_epoch, best_auc
        return limit, None

    def logits(self, x):
        """Score a matrix in evaluation mode and return the logits."""

        self.eval()
        with torch.no_grad():
            xt = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=DEVICE)
            return self.forward(xt).cpu().numpy()

    def importance(self):
        """Return the size of each SNP's first-layer weights, in column order."""

        return np.linalg.norm(self.linears[0].weight.detach().cpu().numpy(), axis=0)

class Estimator(BaseEstimator, ClassifierMixin):
    """Wrap the MLP in the sklearn interface, tuning dropout on its own inner folds."""

    def __init__(self, dropout=None, epochs=None, hidden=HIDDEN, inner=3, pos_weight=1.0,
                 seed=0):
        self.dropout, self.epochs, self.hidden = dropout, epochs, hidden
        self.inner, self.pos_weight, self.seed = inner, pos_weight, seed

    def fit(self, x, y):
        """Tune dropout if none is given, then train the final network on the training set."""

        # Tune dropout and the number of epochs if none are given
        x, y = np.asarray(x), np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.dropout_, self.epochs_, self.val_auc_ = self.dropout, self.epochs, None
        if self.dropout is None:
            self.dropout_, self.epochs_, self.val_auc_ = self.tune(x, y)

        # Train the final network at the chosen settings
        self.net_ = MLP(x.shape[1], self.hidden, self.dropout_, self.pos_weight, self.seed)
        self.net_.fit(x, y, epochs=self.epochs_)
        return self

    def tune(self, x, y):
        """Fit every dropout on every inner fold, and return the best with its scaled mean stopping epoch."""

        # Fit each dropout on each inner fold and record its validation AUC and stopping epoch
        split = StratifiedKFold(self.inner, shuffle=True, random_state=self.seed)
        search = []
        for dropout in DROPOUT:
            aucs, stops = [], []
            for fit_idx, val_idx in split.split(x, y):
                started = time.time()
                net = MLP(x.shape[1], self.hidden, dropout, self.pos_weight, self.seed)
                epoch, value = net.fit(x[fit_idx], y[fit_idx], x_val=x[val_idx], y_val=y[val_idx])
                aucs.append(value)
                stops.append(epoch)
                print(f"      dropout {dropout}  stopped {epoch:>3}  val AUC {value:.3f}"
                      f"  {time.time() - started:.0f}s", flush=True)
            search.append((dropout, float(np.mean(aucs)), int(round(np.mean(stops)))))

        # Choose the dropout with the best mean validation AUC and scale its stopping epoch by (inner - 1) / inner
        dropout, val_auc, epochs = max(search, key=lambda r: r[1])
        epochs = max(1, int(round(epochs * (self.inner - 1) / self.inner)))
        print(f"      chosen dropout {dropout}, {epochs} epochs on everyone", flush=True)
        return dropout, epochs, val_auc

    def decision_function(self, x):
        """Return each person's logit."""

        return self.net_.logits(x)

    def predict(self, x):
        """Return each person's predicted class."""

        return (self.decision_function(x) > 0).astype(int)

def model(seed, inner, pos_weight, hidden=HIDDEN):
    """Build the estimator, with dropout tuned on its inner folds."""

    return Estimator(hidden=hidden, inner=inner, pos_weight=pos_weight, seed=seed)

def chosen(fitted):
    """Rebuild the estimator at the chosen dropout and number of epochs."""

    return Estimator(dropout=fitted.dropout_, epochs=fitted.epochs_, hidden=fitted.hidden,
                     pos_weight=fitted.pos_weight, seed=fitted.seed)

def params(fitted):
    """Return the chosen settings as text."""

    shape = "-".join(str(n) for n in fitted.hidden)
    return f"hidden={shape} dropout={fitted.dropout_} epochs={fitted.epochs_} val_auc={fitted.val_auc_:.3f}"

def importance(fitted):
    """Return the size of each SNP's first-layer weights, in column order."""

    return fitted.net_.importance()
