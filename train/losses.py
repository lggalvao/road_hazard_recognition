import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import WeightedRandomSampler
import numpy as np
#from utils.timing import timeit
import torch.nn.functional as F
import logging

logger = logging.getLogger("hazard_recognition")



#@timeit
def compute_loss(criterion, preds, targets, cfg):
    """Compute training loss."""
    if cfg.loss.loss_function == "focal_loss":
        return criterion(preds, targets, gamma=cfg.loss.focal_loss_gamma)

    return criterion(preds, targets)


def get_loss_function(cfg):
    """
    Returns the appropriate loss function based on args.
    
    Args:
        args (dict): Configuration dictionary with keys:
                     - loss_function
                     - class_weights
                     - device
                     - model
    """
    logger.info("----------- Getting Loss Function -----------")
    loss_name = cfg.loss.loss_function
    
    class_weights = cfg.loss.class_weights
    
    if not isinstance(class_weights, torch.Tensor):
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

    class_weights = class_weights.to(cfg.system.device)
    
    logger.info(f"Loss Function: {loss_name}")
    logger.info(f"Class weights: {class_weights}")
    assert class_weights.shape[0] == cfg.model.num_classes
    assert class_weights.device == cfg.system.device
    
    #logits = torch.tensor([
    #    [5.0, 0.1, 0.1],  # predicts class 0
    #    [0.1, 0.1, 5.0],  # predicts class 2
    #], requires_grad=True)
    #
    #targets = torch.tensor([0, 2])
    #
    #weights = torch.tensor([10.0, 1.0, 1.0])
    #
    #criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
    #loss = criterion(logits, targets)
    #print("Weighted CEL", loss)
    #
    #criterion = torch.nn.CrossEntropyLoss(reduction="none")
    #loss = criterion(logits, targets)
    #print("CEL", loss)
    #
    #criterion = FocalLossMultiClass(
    #        gamma=2,
    #        alpha=weights,
    #        use_class_weights=False,
    #        reduction=None,
    #    )
    #loss = criterion(logits, targets)
    #print("Focal Loss", loss)
    #
    #criterion = FocalLossMultiClass(
    #        gamma=2,
    #        alpha=weights,
    #        use_class_weights=True,
    #        reduction=None,
    #    )
    #loss = criterion(logits, targets)
    #print("Weighted Focal Loss", loss)
    #
    #exit()

    if cfg.model.model == "Trajectory_Embedding_LSTM":
        return nn.MSELoss()

    if loss_name == "weighted_CELoss":
        return nn.CrossEntropyLoss(weight=class_weights)

    elif loss_name == "FocalLoss" or loss_name == "weighted_FocalLoss":
        
        if loss_name == "weighted_FocalLoss":
            use_class_weights = True
        else:
            use_class_weights = False
        
        return FocalLossMultiClass(
            gamma=cfg.loss.focal_loss_gamma,
            alpha=class_weights,
            use_class_weights=use_class_weights,
            reduction="mean",
        )
    
    elif loss_name == "nll_loss":
        return nn.NLLLoss(weight=class_weights)

    elif loss_name == "bce_loss":
        return nn.BCELoss()

    else:  # default
        print("Normal Cross entropy Loss function loaded")
        return nn.CrossEntropyLoss()


class FocalLossMultiClass(nn.Module):
    """
    Multiclass focal loss with optional class weighting.
    Supports:
      - Standard focal loss
      - Weighted focal loss (class imbalance)
      - Reduction: mean / sum / none
    """
    def __init__(
        self,
        gamma=2.0,
        alpha=None,
        use_class_weights=False,
        reduction="mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.use_class_weights = use_class_weights
        self.reduction = reduction
        if alpha is not None:
            # Convert to tensor if needed
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                alpha = alpha.float()
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
    
    def forward(self, logits, targets):
        """
        logits:  [B, C]  (raw, unnormalized)
        targets: [B]     (class indices)
        """
        targets = targets.long()
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)  # Could use: probs = F.softmax(logits, dim=1)
        
        # p_t and log(p_t)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # focal modulation
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        
        # optional class weighting
        if self.use_class_weights:
            if self.alpha is None:
                raise ValueError("use_class_weights=True but alpha is None")
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def prepare_sampler_and_weights_from_sequences(
    cfg,
    sequences,
    device='cpu',
    label_key='hazard_name_hist',
    loss_function='focal_loss',
    classes_name = ""):
    """
    Prepares label encoder, class weights, and weighted sampler.
    """

    # ---- 1. Collect labels per sequence (LAST element safe for pandas Series/list) ----
    all_labels = []
    for seq in sequences:
        seq_label_series = seq[label_key]
        all_labels.append(seq_label_series.item())
    
    # ---- 2. Encode labels ----
    le = LabelEncoder()
    le.fit(classes_name)
    labels_int = le.transform(all_labels)

    logger.info(f"Class Name Order Data Loading (Class Weight): {le.classes_}")
    logger.info(f"Indice Example (Class Weight): {labels_int[0:10]}")
    
    assert list(le.classes_) == list(cfg.model.classes_name), \
        "LabelEncoder classes do not match cfg.model.classes_name"

    # ---- 3. Compute class weights ----
    class_weights_np = compute_class_weight(
        class_weight='balanced',
        classes=le.classes_,
        y=all_labels
    )

    # IMPORTANT: class_weights MUST be on CPU for sampler safety
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device='cpu')

    # Normalize if using focal loss
    if loss_function == 'focal_loss':
        class_weights = class_weights / class_weights.mean()

    # ---- 4. Compute sample weights ----
    labels_tensor = torch.tensor(labels_int, dtype=torch.long, device='cpu')
    labels_tensor = labels_int
    sample_weights = class_weights[labels_tensor]   # stays on CPU

    # ---- 4b. SANITIZE weights to avoid CUDA device-side asserts ----
    sample_weights = sample_weights.clone()

    # replace NaN/Inf with 0
    sample_weights[torch.isnan(sample_weights)] = 0.0
    sample_weights[torch.isinf(sample_weights)] = 0.0

    # clamp negatives
    sample_weights = torch.clamp(sample_weights, min=0.0)

    # ensure not all zero
    if float(sample_weights.sum()) == 0.0:
        sample_weights += 1e-6

    # ---- 5. Weighted sampler (weights MUST be CPU float64/32) ----
    sampler = WeightedRandomSampler(
        weights=sample_weights.double(),     # CPU tensor
        num_samples=len(sequences),
        replacement=True
    )

    return class_weights, sampler



def test_focal_loss():
    """Comprehensive test suite for FocalLossMultiClass"""
    
    print("="*70)
    print("FOCAL LOSS FUNCTION TESTS")
    print("="*70)
    
    # Test 1: Basic functionality with simple example
    print("\n" + "="*70)
    print("TEST 1: Basic Focal Loss (gamma=2.0, no weights)")
    print("="*70)
    
    # Create simple data: 3 samples, 4 classes
    logits = torch.tensor([
        [2.0, 1.0, 0.5, 0.1],  # Sample 0: confident about class 0
        [0.5, 3.0, 0.3, 0.2],  # Sample 1: confident about class 1
        [1.0, 1.0, 1.0, 1.0],  # Sample 2: uncertain (equal logits)
    ])
    targets = torch.tensor([0, 1, 2])  # Ground truth labels
    
    focal_loss = FocalLossMultiClass(gamma=2.0, reduction="none")
    loss = focal_loss(logits, targets)
    
    # Calculate probabilities for interpretation
    probs = F.softmax(logits, dim=1)
    
    print(f"\nInput logits:\n{logits}")
    print(f"\nTargets: {targets}")
    print(f"\nProbabilities (after softmax):\n{probs}")
    print(f"\nTarget probabilities:")
    for i in range(len(targets)):
        print(f"  Sample {i}: p(class {targets[i]}) = {probs[i, targets[i]]:.4f}")
    
    print(f"\nFocal Loss per sample: {loss}")
    print(f"Mean Focal Loss: {loss.mean():.4f}")
    
    # Test 2: Compare gamma values
    print("\n" + "="*70)
    print("TEST 2: Effect of Gamma Parameter")
    print("="*70)
    
    gammas = [0.0, 1.0, 2.0, 5.0]
    print("\nSame input, different gamma values:")
    for gamma in gammas:
        focal_loss = FocalLossMultiClass(gamma=gamma, reduction="mean")
        loss = focal_loss(logits, targets)
        print(f"  gamma={gamma}: loss={loss:.4f}")
    
    print("\nNote: Higher gamma = more focus on hard examples (low probability)")
    
    # Test 3: Class weights (handling imbalanced data)
    print("\n" + "="*70)
    print("TEST 3: Weighted Focal Loss (Class Imbalance)")
    print("="*70)
    
    # Simulate imbalanced dataset
    # Class 0: common (weight=0.5), Classes 1,2,3: rare (weight=1.5)
    alpha = torch.tensor([0.5, 1.5, 1.5, 1.5])
    
    focal_loss_weighted = FocalLossMultiClass(
        gamma=2.0, 
        alpha=alpha, 
        use_class_weights=True,
        reduction="none"
    )
    focal_loss_unweighted = FocalLossMultiClass(gamma=2.0, reduction="none")
    
    loss_weighted = focal_loss_weighted(logits, targets)
    loss_unweighted = focal_loss_unweighted(logits, targets)
    
    print(f"\nClass weights (alpha): {alpha}")
    print(f"\nUnweighted loss: {loss_unweighted}")
    print(f"Weighted loss:   {loss_weighted}")
    print(f"\nMean - Unweighted: {loss_unweighted.mean():.4f}")
    print(f"Mean - Weighted:   {loss_weighted.mean():.4f}")
    
    # Test 4: Edge cases
    print("\n" + "="*70)
    print("TEST 4: Edge Cases")
    print("="*70)
    
    # Very confident correct prediction
    confident_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    confident_target = torch.tensor([0])
    
    # Very wrong prediction
    wrong_logits = torch.tensor([[0.0, 0.0, 0.0, 10.0]])
    wrong_target = torch.tensor([0])
    
    focal_loss = FocalLossMultiClass(gamma=2.0, reduction="mean")
    
    loss_confident = focal_loss(confident_logits, confident_target)
    loss_wrong = focal_loss(wrong_logits, wrong_target)
    
    print(f"\nVery confident CORRECT prediction:")
    print(f"  Logits: {confident_logits[0].tolist()}")
    print(f"  Target: {confident_target.item()}")
    print(f"  Probability: {F.softmax(confident_logits, dim=1)[0, 0]:.6f}")
    print(f"  Focal Loss: {loss_confident:.6f}")
    
    print(f"\nVery confident WRONG prediction:")
    print(f"  Logits: {wrong_logits[0].tolist()}")
    print(f"  Target: {wrong_target.item()}")
    print(f"  Probability: {F.softmax(wrong_logits, dim=1)[0, 0]:.6f}")
    print(f"  Focal Loss: {loss_wrong:.4f}")
    
    # Test 5: Reduction modes
    print("\n" + "="*70)
    print("TEST 5: Different Reduction Modes")
    print("="*70)
    
    batch_logits = torch.randn(5, 4)  # 5 samples, 4 classes
    batch_targets = torch.tensor([0, 1, 2, 3, 0])
    
    for reduction in ["mean", "sum", "none"]:
        focal_loss = FocalLossMultiClass(gamma=2.0, reduction=reduction)
        loss = focal_loss(batch_logits, batch_targets)
        print(f"\nReduction='{reduction}':")
        print(f"  Loss: {loss}")
        if reduction == "none":
            print(f"  Shape: {loss.shape}")
    
    # Test 6: GPU compatibility test (if available)
    print("\n" + "="*70)
    print("TEST 6: Device Compatibility")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    focal_loss = FocalLossMultiClass(
        gamma=2.0, 
        alpha=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        use_class_weights=True
    ).to(device)
    
    logits_gpu = logits.to(device)
    targets_gpu = targets.to(device)
    
    loss = focal_loss(logits_gpu, targets_gpu)
    print(f"Loss computed on {device}: {loss.item():.4f}")
    print("✓ Device compatibility test passed!")
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    test_focal_loss()




