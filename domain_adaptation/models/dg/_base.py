"""Base building blocks for the DG algorithms (DGModel + featurizer/classifier factories)."""

import os
import sys

import torch.nn as nn

# Make sibling top-level modules under ``basemodel-benchmarking/`` importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_BMB = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "basemodel-benchmarking"))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
for _p in (_BMB, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backbones import MLPFeaturizer, ResNetFeaturizer, TransformerFeaturizer


class FeatureClassifier(nn.Module):
    def __init__(self, featurizer, classifier):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier

    def forward(self, x):
        feats = self.featurizer(x)
        return self.classifier(feats)

    def forward_features(self, x):
        return self.featurizer(x)


def _build_featurizer(input_dim, hparams):
    backbone_name = hparams.get('backbone', 'MLP')
    dropout = hparams.get('dropout', 0.3)
    hidden_dim = hparams.get('hidden_dim', 256)

    if backbone_name == 'MLP':
        return MLPFeaturizer(
            input_dim,
            hidden_dim=hidden_dim,
            output_dim=128,
            num_layers=hparams.get('num_layers', 3),
            dropout=dropout
        )
    if backbone_name == 'ResNet':
        return ResNetFeaturizer(
            input_dim,
            hidden_dim=hidden_dim,
            output_dim=128,
            num_blocks=hparams.get('num_blocks', 2),
            dropout=dropout
        )
    if backbone_name == 'Transformer':
        return TransformerFeaturizer(
            input_dim,
            hidden_dim=128,
            output_dim=128,
            num_layers=hparams.get('num_layers', 2),
            nhead=hparams.get('nhead', 4),
            dropout=dropout
        )

    raise ValueError(f"Unknown backbone: {backbone_name}")


def _build_classifier(in_dim, num_classes, hparams):
    nonlinear = hparams.get('nonlinear_classifier', False)
    dropout = hparams.get('classifier_dropout', 0.0)
    hidden_dim = hparams.get('classifier_hidden_dim', in_dim)
    if not nonlinear:
        return nn.Linear(in_dim, num_classes)
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes)
    )


class DGModel(nn.Module):
    """
    Base class for Domain Generalization models.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hparams = hparams if hparams else {}
        self.num_domains = self.hparams.get('num_domains')

        self.featurizer = _build_featurizer(input_dim, self.hparams)
        self.classifier = _build_classifier(self.featurizer.output_dim, num_classes, self.hparams)
        self.network = FeatureClassifier(self.featurizer, self.classifier)

    def predict(self, x):
        return self.network(x)

    def forward(self, x):
        return self.predict(x)

    def update(self, minibatches, unlabeled=None, **kwargs):
        """
        Input: minibatches is a list of (x, y) pairs, one per domain.
        """
        raise NotImplementedError


