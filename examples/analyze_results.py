#!/usr/bin/env python3
"""Script to analyze intervention results."""

import json
import sys
from pathlib import Path
from collections import defaultdict


def load_results(filepath):
    """Load intervention results from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['results'], data.get('config', {})


def analyze_token_counts(results):
    """Analyze token count changes across interventions."""
    
    # Group by configuration
    baseline = [r for r in results if r['layer'] == -1]
    interventions = [r for r in results if r['layer'] != -1]
    
    if not baseline:
        print("No baseline results found!")
        return
    
    baseline_tokens = sum(r['num_tokens'] for r in baseline) / len(baseline)
    print(f"Baseline average tokens: {baseline_tokens:.1f}")
    print()
    
    # Group by layer, strength, direction
    groups = defaultdict(list)
    for r in interventions:
        key = (r['layer'], r['strength'], r['direction_type'])
        groups[key].append(r['num_tokens'])
    
    # Find most impactful interventions
    impacts = []
    for key, tokens in groups.items():
        avg_tokens = sum(tokens) / len(tokens)
        delta = avg_tokens - baseline_tokens
        impacts.append((key, avg_tokens, delta))
    
    # Sort by absolute impact
    impacts.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("Top 10 Most Impactful Interventions:")
    print("-" * 80)
    for i, (key, avg_tokens, delta) in enumerate(impacts[:10], 1):
        layer, strength, direction = key
        sign = "+" if delta > 0 else ""
        print(f"{i:2}. Layer {layer:2}, {direction:10}, strength {strength:5.2f}: "
              f"{avg_tokens:6.1f} tokens ({sign}{delta:6.1f})")


def analyze_reasoning_suppression(results):
    """Identify cases where reasoning was suppressed."""
    
    interventions = [r for r in results if r['layer'] != -1]
    
    suppressed = []
    for r in interventions:
        output = r['output'].lower()
        # Check if <think> tags are absent (reasoning suppression)
        if '<think>' not in output and 'think' not in output:
            suppressed.append(r)
    
    print(f"\nFound {len(suppressed)} cases with suppressed reasoning:")
    print("-" * 80)
    
    # Group by configuration
    configs = defaultdict(int)
    for r in suppressed:
        key = (r['layer'], r['strength'], r['direction_type'])
        configs[key] += 1
    
    for key, count in sorted(configs.items(), key=lambda x: x[1], reverse=True)[:10]:
        layer, strength, direction = key
        print(f"Layer {layer:2}, {direction:10}, strength {strength:5.2f}: {count} times")


def main():
    """Main analysis function."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_file.json>")
        print("\nAvailable results files:")
        results_dir = Path("results")
        if results_dir.exists():
            for f in sorted(results_dir.glob("interventions_*.json")):
                print(f"  {f}")
        sys.exit(1)
    
    filepath = sys.argv[1]
    print(f"Analyzing: {filepath}")
    print("=" * 80)
    
    results, config = load_results(filepath)
    print(f"Total intervention runs: {len(results)}")
    print(f"Configuration: {config}")
    print()
    
    analyze_token_counts(results)
    analyze_reasoning_suppression(results)


if __name__ == "__main__":
    main()

