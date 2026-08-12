"""
Optimizer configuration and parameter group building utilities.

Provides:
- YAML-driven optimizer config (OptimizerConfig)
- Declarative parameter group specs (ParamGroupSpec, SourceSpec)
- Module-tree-based parameter collection (no fragile string matching)
- Safety validation that all trainable params are covered
- Structured serialization (``to_dict``) for console logging / wandb capture
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Pattern, Set, Tuple, Union

import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numeric coercion (PyYAML quirk defense)
# ---------------------------------------------------------------------------

def _coerce_float(value: object, key: str, default: Optional[float]) -> Optional[float]:
    """Coerce a config scalar to float, tolerating PyYAML's string quirk.

    PyYAML parses exponent notation *without* a decimal point (``1e-2``) as
    a string.  A ``'1e-2'`` weight_decay would otherwise flow into the
    optimizer and only explode at the first ``optimizer.step()``.  Numeric
    strings are therefore coerced to float with a warning; anything
    non-numeric (incl. bool) raises ``ValueError`` immediately.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"Config field '{key}' must be a number, got {value!r}.")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Config field '{key}' must be a number, got {value!r} "
            f"(type {type(value).__name__})."
        ) from None
    logger.warning(
        "Config field '%s' was given as numeric string %r; PyYAML parses "
        "exponent forms without a decimal point (e.g. '1e-2') as strings. "
        "Coerced to %s — prefer decimal-point form like 1.0e-2 or 0.01.",
        key, value, f,
    )
    return f


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SourceSpec:
    """Specifies how to collect parameters from a submodule.

    Attributes:
        attr: dot-separated attribute path relative to the model root, e.g.
              ``"model.layers"`` or ``"model.fuse"``. For ``ModuleList``
              children all submodules are walked automatically.
        dim: optional dimension constraint, e.g. ``">=2"``, ``"<2,==3"``.
             Comma-separated tokens are OR-ed.  ``None`` means no constraint.
        name: optional regex pattern matched (via ``re.search``) against the
              parameter name **relative to the submodule reached by attr**,
              e.g. ``"attn.q_proj.weight"`` or ``"lambda_params"``.  Only
              params whose name matches are collected.  Use ``^...$`` anchors
              to match the whole name.  Composed with ``dim`` by AND.
              ``None`` means no name constraint.
    """
    attr: str
    dim: Optional[str] = None
    name: Optional[str] = None


@dataclass
class ParamGroupSpec:
    """Declares one optimizer parameter group with matching rules and hyperparams.

    Attributes:
        name: human-readable group name (logged / used in error messages).
        source: built-in source name. ``"embedding"`` calls
                ``get_input_embeddings()``; ``"unembedding"`` calls
                ``get_output_embeddings()``.  Mutually exclusive with
                ``sources``.
        sources: list of ``SourceSpec`` entries to collect params from.
                 Mutually exclusive with ``source``.
        lr_mult: multiplier applied to ``peak_lr``.
        weight_decay: weight decay for this group. ``None`` to inherit
                      the global value.
        betas: Adam / Muon-aux-Adam betas.
        use_muon: if ``True``, 2D params in this group are marked with
                  ``p.use_muon = True``.
        muon_momentum: Muon momentum coefficient (used when ``use_muon=True``).

    ``sources`` entries may carry a ``name`` regex filter in addition to
    ``attr``/``dim`` (see :class:`SourceSpec`).
    """
    name: str
    source: Optional[str] = None
    sources: Optional[List[SourceSpec]] = None
    lr_mult: float = 1.0
    weight_decay: Optional[float] = None
    betas: Tuple[float, float] = (0.9, 0.95)
    use_muon: bool = False
    muon_momentum: float = 0.95

    def __post_init__(self):
        if self.source is not None and self.sources is not None:
            raise ValueError(
                f"ParamGroupSpec '{self.name}': 'source' and 'sources' are "
                f"mutually exclusive."
            )
        if self.source is None and self.sources is None:
            raise ValueError(
                f"ParamGroupSpec '{self.name}': either 'source' or 'sources' "
                f"must be provided."
            )
        if self.source is not None and self.source not in ("embedding", "unembedding"):
            raise ValueError(
                f"ParamGroupSpec '{self.name}': unknown source "
                f"'{self.source}'. Expected 'embedding' or 'unembedding'."
            )

    def to_dict(self) -> dict:
        """Serialize to a nested, JSON-serializable dict (logging / wandb).

        ``source`` / ``sources`` are included only when set; ``betas`` is
        emitted as a list since tuples are not JSON-serializable.
        """
        d = {
            "name": self.name,
            "lr_mult": self.lr_mult,
            "weight_decay": self.weight_decay,
            "betas": list(self.betas),
            "use_muon": self.use_muon,
            "muon_momentum": self.muon_momentum,
        }
        if self.source is not None:
            d["source"] = self.source
        if self.sources is not None:
            d["sources"] = [
                {"attr": s.attr, "dim": s.dim, "name": s.name}
                for s in self.sources
            ]
        return d


@dataclass
class OptimizerConfig:
    """Top-level optimizer configuration, typically loaded from a YAML file.

    Attributes:
        peak_lr: peak learning rate (before lr_mult).
        warmup_ratio: fraction of total steps used for linear warmup.
        min_lr_frac: minimum LR as a fraction of ``peak_lr`` (cosine floor).
        weight_decay: global weight decay (inherited by groups with
                      ``weight_decay=None``).
        grad_clip: global gradient clipping value. ``None`` disables clipping.
        param_groups: ordered list of ``ParamGroupSpec``. First-match-wins.
    """
    peak_lr: float = 0.02
    warmup_ratio: float = 0.005
    min_lr_frac: float = 0.01
    weight_decay: float = 0.1
    grad_clip: Optional[float] = None
    param_groups: List[ParamGroupSpec] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "OptimizerConfig":
        """Load an ``OptimizerConfig`` from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict) -> "OptimizerConfig":
        """Construct an ``OptimizerConfig`` from a plain dictionary."""
        groups = []
        for g in d.get("param_groups", []):
            gname = g.get("name", "<unnamed>")
            sources = None
            if g.get("sources"):
                sources = [
                    SourceSpec(attr=s["attr"], dim=s.get("dim"), name=s.get("name"))
                    for s in g["sources"]
                ]
            betas_raw = g.get("betas", [0.9, 0.95])
            groups.append(ParamGroupSpec(
                name=gname,
                source=g.get("source"),
                sources=sources,
                lr_mult=_coerce_float(g.get("lr_mult"), f"{gname}.lr_mult", 1.0),
                weight_decay=_coerce_float(
                    g.get("weight_decay"), f"{gname}.weight_decay", None
                ),
                betas=tuple(
                    _coerce_float(b, f"{gname}.betas[{i}]", 0.9 if i == 0 else 0.95)
                    for i, b in enumerate(betas_raw)
                ),
                use_muon=g.get("use_muon", False),
                muon_momentum=_coerce_float(
                    g.get("muon_momentum"), f"{gname}.muon_momentum", 0.95
                ),
            ))
        return cls(
            peak_lr=_coerce_float(d.get("peak_lr"), "peak_lr", 0.02),
            warmup_ratio=_coerce_float(d.get("warmup_ratio"), "warmup_ratio", 0.005),
            min_lr_frac=_coerce_float(d.get("min_lr_frac"), "min_lr_frac", 0.01),
            weight_decay=_coerce_float(d.get("weight_decay"), "weight_decay", 0.1),
            grad_clip=_coerce_float(d.get("grad_clip"), "grad_clip", None),
            param_groups=groups,
        )

    def to_dict(self) -> dict:
        """Serialize to a nested, JSON-serializable dict (logging / wandb).

        The structure mirrors the YAML: global hyperparameters plus an
        ordered ``param_groups`` list of per-group dicts.
        """
        return {
            "peak_lr": self.peak_lr,
            "warmup_ratio": self.warmup_ratio,
            "min_lr_frac": self.min_lr_frac,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "param_groups": [g.to_dict() for g in self.param_groups],
        }


# ---------------------------------------------------------------------------
# Name (regex) filter parsing
# ---------------------------------------------------------------------------

def _parse_name_regex(name_str: Optional[str]) -> Optional[Pattern]:
    """Compile a name filter regex, or return ``None`` when absent.

    Matching uses ``re.search`` semantics: the pattern may match any
    substring of the parameter name; use ``^``/``$`` anchors to require a
    full-name match.  Invalid patterns raise ``ValueError`` immediately so
    config typos fail at startup instead of silently matching nothing.
    """
    if name_str is None:
        return None
    try:
        return re.compile(name_str)
    except re.error as e:
        raise ValueError(f"Invalid 'name' regex {name_str!r}: {e}") from e


# ---------------------------------------------------------------------------
# Dim constraint parsing
# ---------------------------------------------------------------------------

def _parse_dim_constraint(dim_str: Optional[str]) -> Callable[[nn.Parameter], bool]:
    """Parse a dim constraint string into a predicate.

    Supported operators: ``>=``, ``>``, ``<=``, ``<``, ``==``, ``!=``.
    Comma-separated tokens are OR-ed.  ``None`` means match everything.

    Examples:
        ``">=2"``       → dim >= 2
        ``"<2,==3"``    → dim < 2  OR  dim == 3
    """
    if dim_str is None:
        return lambda p: True

    predicates: List[Callable[[int], bool]] = []
    for token in dim_str.split(","):
        token = token.strip()
        if token.startswith(">="):
            n = int(token[2:])
            predicates.append(lambda d, n=n: d >= n)
        elif token.startswith("<="):
            n = int(token[2:])
            predicates.append(lambda d, n=n: d <= n)
        elif token.startswith("!="):
            n = int(token[2:])
            predicates.append(lambda d, n=n: d != n)
        elif token.startswith("=="):
            n = int(token[2:])
            predicates.append(lambda d, n=n: d == n)
        elif token.startswith(">"):
            n = int(token[1:])
            predicates.append(lambda d, n=n: d > n)
        elif token.startswith("<"):
            n = int(token[1:])
            predicates.append(lambda d, n=n: d < n)
        else:
            raise ValueError(
                f"Invalid dim constraint token: '{token}'. "
                f"Expected one of >=N, >N, <=N, <N, ==N, !=N."
            )

    return lambda p: any(pred(p.dim()) for pred in predicates)


# ---------------------------------------------------------------------------
# Parameter collection from module tree
# ---------------------------------------------------------------------------

def _collect_from_module(
    root: nn.Module,
    attr_path: str,
    dim_pred: Callable[[nn.Parameter], bool],
    name_pred: Optional[Pattern] = None,
) -> List[Tuple[str, nn.Parameter]]:
    """Collect (name, param) pairs from a submodule reached via *attr_path*.

    *attr_path* is a dot-separated sequence of ``getattr`` steps relative to
    *root*.  ``ModuleList`` children are walked automatically.  Only
    parameters satisfying *dim_pred* **and** (when given) *name_pred* are
    kept; *name_pred* is matched via ``re.search`` against the name relative
    to the submodule (e.g. ``attn.q_proj.weight``).
    """
    parts = attr_path.split(".")
    current: nn.Module = root
    for part in parts:
        try:
            current = getattr(current, part)
        except AttributeError:
            raise AttributeError(
                f"Cannot resolve attr path '{attr_path}' on "
                f"{type(root).__name__}: missing attribute '{part}' "
                f"on {type(current).__name__}."
            )

    result: List[Tuple[str, nn.Parameter]] = []
    for name, param in current.named_parameters():
        if dim_pred(param) and (name_pred is None or name_pred.search(name)):
            result.append((name, param))
    return result


# ---------------------------------------------------------------------------
# Main entry point: build param groups from config
# ---------------------------------------------------------------------------

def build_param_groups(
    model: nn.Module,
    specs: List[ParamGroupSpec],
    peak_lr: float,
    global_weight_decay: float,
    optimizer_type: str = "muon",
) -> Tuple[List[dict], Set[str]]:
    """Build optimizer-ready ``param_groups`` from declarative specs.

    Parameters are matched **first-match-wins**: earlier groups consume
    parameters first, and those parameters are excluded from later groups.

    Args:
        model: the inner model (must have ``get_input_embeddings`` /
               ``get_output_embeddings`` if those sources are used).
        specs: ordered list of ``ParamGroupSpec``.
        peak_lr: base learning rate (multiplied by ``lr_mult``).
        global_weight_decay: fallback weight decay for groups that set
                             ``weight_decay=None``.
        optimizer_type: ``"muon"`` or ``"adam"``.

    Returns:
        ``(param_groups, uncovered)`` where *param_groups* is a list of
        dicts suitable for ``torch.optim.Optimizer`` / Muon constructor,
        and *uncovered* is the set of parameter names that were not
        assigned to any group (empty when coverage is complete).
    """
    # Index all trainable params: name -> param.  Tied params (same id)
    # naturally appear under multiple names; we track by id to deduplicate.
    all_params: Dict[str, nn.Parameter] = {}
    for name, param in model.named_parameters():
        all_params[name] = param

    assigned_ids: Set[int] = set()
    param_groups: List[dict] = []

    for spec in specs:
        collected: List[Tuple[str, nn.Parameter]] = []

        if spec.source == "embedding":
            emb = model.get_input_embeddings()
            for name, param in emb.named_parameters():
                collected.append((f"emb.{name}", param))
        elif spec.source == "unembedding":
            head = model.get_output_embeddings()
            for name, param in head.named_parameters():
                collected.append((f"head.{name}", param))
        elif spec.sources:
            for src in spec.sources:
                dim_pred = _parse_dim_constraint(src.dim)
                try:
                    name_pred = _parse_name_regex(src.name)
                except ValueError as e:
                    raise ValueError(
                        f"Error in group '{spec.name}', source attr='{src.attr}': {e}"
                    ) from e
                try:
                    src_params = _collect_from_module(
                        model, src.attr, dim_pred, name_pred
                    )
                except AttributeError as e:
                    raise AttributeError(
                        f"Error in group '{spec.name}', source attr='{src.attr}': {e}"
                    ) from e
                if not src_params and name_pred is not None:
                    # A name-filtered source that matches nothing is usually
                    # a typo'd pattern; warn instead of silently dropping.
                    logger.warning(
                        "Group '%s', source attr='%s': name filter %r matched "
                        "no parameters under the submodule.",
                        spec.name, src.attr, src.name,
                    )
                collected.extend(src_params)

        # First-match-wins: keep only params not yet assigned
        filtered: List[nn.Parameter] = []
        for name, param in collected:
            pid = id(param)
            if pid not in assigned_ids:
                filtered.append(param)
                assigned_ids.add(pid)

        if not filtered:
            continue

        lr = peak_lr * spec.lr_mult
        wd = spec.weight_decay if spec.weight_decay is not None else global_weight_decay

        group: dict = {
            "params": filtered,
            "lr": lr,
        }

        if optimizer_type == "muon":
            group["weight_decay"] = wd
            group["use_muon"] = spec.use_muon
            if spec.use_muon:
                # Muon group: momentum instead of betas
                group["momentum"] = spec.muon_momentum
                # Mark params for Muon update
                for p in filtered:
                    p.use_muon = True  # type: ignore[attr-defined]
            else:
                # Aux Adam group: betas + eps
                group["betas"] = spec.betas
                group["eps"] = 1e-10
        else:
            # Pure AdamW
            group["weight_decay"] = wd
            group["betas"] = spec.betas

        param_groups.append(group)

    # Compute uncovered
    all_ids = set(id(p) for p in model.parameters())
    uncovered_ids = all_ids - assigned_ids
    uncovered = {
        name for name, p in all_params.items() if id(p) in uncovered_ids
    }

    return param_groups, uncovered


# ---------------------------------------------------------------------------
# Safety validation
# ---------------------------------------------------------------------------

def validate_param_coverage(
    model: nn.Module,
    param_groups: List[dict],
    uncovered: Set[str],
) -> None:
    """Raise ``RuntimeError`` if any trainable parameter is not covered.

    Prints a detailed listing of uncovered parameters before raising.
    """
    if not uncovered:
        return

    lines = [
        f"ERROR: {len(uncovered)} parameter(s) are not covered by any "
        f"optimizer param group:",
    ]
    for name in sorted(uncovered):
        lines.append(f"  - {name}")
    lines.append(
        "Add a matching ParamGroupSpec to your optimizer config YAML."
    )
    raise RuntimeError("\n".join(lines))


# ---------------------------------------------------------------------------
# Convenience: load config from YAML
# ---------------------------------------------------------------------------

def load_optimizer_config(path: str) -> OptimizerConfig:
    """Shorthand to load an ``OptimizerConfig`` from a YAML file."""
    return OptimizerConfig.from_yaml(path)
