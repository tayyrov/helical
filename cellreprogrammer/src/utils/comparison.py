"""
Utility functions for comparing perturbation experiment results.

This module provides reusable functions for calculating fold improvements
and formatting perturbation results to avoid code duplication.
"""

from typing import List, Optional


def calculate_fold_improvement(target_mean: float, random_mean: float, threshold: float = 1e-8) -> Optional[float]:
    """
    Calculate fold improvement only when both values have the same sign.
    
    Parameters
    ----------
    target_mean : float
        Mean shift for target genes
    random_mean : float
        Mean shift for random controls
    threshold : float
        Minimum absolute value for random_mean to calculate fold improvement
        
    Returns
    -------
    Optional[float]
        Fold improvement if both have same sign and random_mean > threshold, else None
    """
    same_sign = (target_mean > 0 and random_mean > 0) or (target_mean < 0 and random_mean < 0)
    if same_sign and abs(random_mean) > threshold:
        return abs(target_mean) / abs(random_mean)
    return None


def format_perturbation_results(
    target_mean: float,
    target_std: float,
    random_mean: float,
    random_std: float,
    target_genes: List[str],
    random_genes: List[str],
    goal_state: str,
    fold_change: Optional[float] = None,
    perturbation_type: str = "overexpressed",
) -> None:
    """
    Format and print perturbation experiment results with appropriate interpretation.
    
    Parameters
    ----------
    target_mean : float
        Mean shift for target genes
    target_std : float
        Standard deviation for target genes
    random_mean : float
        Mean shift for random controls
    random_std : float
        Standard deviation for random controls
    target_genes : List[str]
        List of target gene names (for display)
    random_genes : List[str]
        List of random control gene names (for display)
    goal_state : str
        Goal cell state name
    fold_change : Optional[float]
        Fold change used for perturbation (None for Geneformer)
    perturbation_type : str
        Type of perturbation (e.g., "overexpressed", "perturbed")
    """
    improvement = target_mean - random_mean
    fold_improvement = calculate_fold_improvement(target_mean, random_mean)
    
    # Build perturbation description
    if fold_change:
        pert_desc = f"were {perturbation_type} {fold_change}x"
    else:
        # Default description (works for Geneformer and other models)
        pert_desc = f"were {perturbation_type}"
    
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(f"When {', '.join(target_genes)} {pert_desc}:")
    print(f"  • Mean shift toward {goal_state}: {target_mean:+.6f} ± {target_std:.6f}")
    if random_mean > 0:
        print(f"  • Random controls shift: {random_mean:+.6f} ± {random_std:.6f}")
    else:
        print(f"  • Random controls shift: {random_mean:.6f} ± {random_std:.6f}")
    print()
    
    # Interpretation based on sign relationship
    if target_mean > 0 and random_mean < 0:
        # Target positive (toward), random negative (away) - very strong result
        print(f"✓ Strong result: Target genes shift cells TOWARD {goal_state}, while random controls shift AWAY")
        print(f"  This indicates target genes have a clear positive effect on reprogramming")
        print(f"  Improvement: {improvement:+.6f} (cannot calculate fold change when directions differ)")
    elif target_mean < 0 and random_mean > 0:
        # Target negative, random positive - unexpected
        print(f"⚠ Unexpected: Target genes shift cells AWAY from {goal_state}, while random controls shift toward")
        print(f"  This may indicate an issue with the perturbation or analysis")
    elif target_mean < 0 and random_mean < 0:
        # Both negative - both moving away from goal
        if improvement > 0:
            print(f"⚠ Both target and random perturbations moved cells AWAY from {goal_state}")
            print(f"  However, target genes ({', '.join(target_genes)}) moved less away than random controls")
            print(f"  Improvement: {improvement:+.6f} (target is {abs(improvement):.6f} closer than random)")
            print(f"  Note: Negative shifts may indicate the perturbation approach needs adjustment")
        else:
            print(f"✗ Both target and random perturbations moved cells AWAY from {goal_state}")
            print(f"  Target genes did not show improvement over random controls")
    elif fold_improvement and fold_improvement > 1.0:
        # Same sign and meaningful fold improvement
        print(f"✓ Target genes showed {fold_improvement:.2f}x better shift toward {goal_state}")
        print(f"  compared to random controls ({', '.join(random_genes)})")
    elif improvement > 0:
        # Improvement but no meaningful fold change (e.g., one near zero)
        print(f"✓ Target genes shifted cells {improvement:+.6f} closer to {goal_state}")
        print(f"  compared to random controls ({', '.join(random_genes)})")
    else:
        print(f"✗ Target genes did not show improvement over random controls")
