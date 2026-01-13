#!/usr/bin/env python3
"""
Run agent evaluations with Logfire tracing.

Similar to run_all_evaluations.py but focused on Logfire.
Can run alongside Langfuse when both are enabled.

Usage:
    python examples/run_logfire_evaluation.py --agent agent_1 --max-items 5
    python examples/run_logfire_evaluation.py --all --max-items 3
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings
from app.logfire_config import configure_logfire, instrument_pydantic_ai
from app.logfire_evaluation import LogfireEvaluator
from app.langfuse_datasets import (
    AGENT_1_DATASET,
    AGENT_2_DATASET,
    AGENT_3_DATASET,
    AGENT_4_DATASET,
    AGENT_5_DATASET,
)


DATASETS = {
    "agent_1": ("portfolio_extraction", AGENT_1_DATASET),
    "agent_2": ("financial_calculator", AGENT_2_DATASET),
    "agent_3": ("risk_tax_advisor", AGENT_3_DATASET),
    "agent_4": ("option_pricing", AGENT_4_DATASET),
    "agent_5": ("swift_iso20022", AGENT_5_DATASET),
}


def get_agent(agent_name: str):
    """Dynamically import and return the agent."""
    if agent_name == "agent_1":
        from examples.agent_1 import agent_1
        return agent_1
    elif agent_name == "agent_2":
        from examples.agent_2 import agent_2
        return agent_2
    elif agent_name == "agent_3":
        # Agent 3 is a multi-agent workflow, use the main portfolio_optimizer
        from examples.agent_3 import portfolio_optimizer
        return portfolio_optimizer
    elif agent_name == "agent_4":
        from examples.agent_4 import agent_4
        return agent_4
    elif agent_name == "agent_5":
        from examples.agent_5 import agent_5
        return agent_5
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


async def evaluate_agent(
    agent_name: str,
    max_items: int = 5,
    difficulty: str | None = None,
) -> dict:
    """Run evaluation for a single agent."""
    dataset_name, dataset = DATASETS.get(agent_name, (None, None))
    if not dataset:
        print(f"No dataset found for {agent_name}")
        return {}
    
    # Filter by difficulty if specified
    items = dataset
    if difficulty:
        items = [item for item in dataset if item.get("difficulty") == difficulty]
    
    # Limit items
    items = items[:max_items]
    
    print(f"\n{'='*60}")
    print(f"Evaluating {agent_name} with {len(items)} items")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Get agent
    try:
        agent = get_agent(agent_name)
    except Exception as e:
        print(f"Failed to load agent {agent_name}: {e}")
        return {}
    
    # Create evaluator
    evaluator = LogfireEvaluator(
        agent_name=agent_name,
        dataset_name=dataset_name,
    )
    
    # Run evaluations
    for i, item in enumerate(items):
        print(f"\n[{i+1}/{len(items)}] {item.get('difficulty', 'unknown')} - {item.get('category', 'unknown')}")
        print(f"Prompt: {item['prompt'][:100]}...")
        
        result = await evaluator.evaluate_item(
            agent=agent,
            prompt=item["prompt"],
            expected_output=item.get("expected_output"),
            item_id=f"{agent_name}_{i}",
            difficulty=item.get("difficulty", "unknown"),
            category=item.get("category", "unknown"),
            metadata=item.get("metadata"),
        )
        
        if result["success"]:
            print(f"  ✓ Success ({result['elapsed']:.2f}s)")
            print(f"  Scores: {result['scores']}")
        else:
            print(f"  ✗ Failed: {result['error']}")
    
    # Get summary
    summary = evaluator.get_summary()
    
    print(f"\n{'='*60}")
    print(f"Summary for {agent_name}")
    print(f"{'='*60}")
    print(f"Total: {summary['total_items']}, Success: {summary['successful']}")
    print(f"Success Rate: {summary['success_rate']*100:.1f}%")
    print(f"Average Scores:")
    for score_name, score_value in summary.get("average_scores", {}).items():
        print(f"  - {score_name}: {score_value:.3f}")
    
    return summary


async def main():
    parser = argparse.ArgumentParser(description="Run Logfire evaluations")
    parser.add_argument(
        "--agent",
        type=str,
        choices=["agent_1", "agent_2", "agent_3", "agent_4", "agent_5"],
        help="Specific agent to evaluate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all agents",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=5,
        help="Maximum items per agent (default: 5)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard", "expert"],
        help="Filter by difficulty",
    )
    
    args = parser.parse_args()
    
    if not args.agent and not args.all:
        parser.print_help()
        print("\nError: Specify --agent or --all")
        sys.exit(1)
    
    # Configure Logfire
    print("Configuring Logfire...")
    configure_logfire(send_to_logfire=True)
    instrument_pydantic_ai()
    print(f"Logfire enabled: {settings.enable_logfire}")
    print(f"Dashboard: https://logfire-eu.pydantic.dev/deal-ex-machina/open-finance")
    
    # Run evaluations
    agents_to_evaluate = (
        list(DATASETS.keys()) if args.all
        else [args.agent]
    )
    
    results = {}
    for agent_name in agents_to_evaluate:
        try:
            summary = await evaluate_agent(
                agent_name=agent_name,
                max_items=args.max_items,
                difficulty=args.difficulty,
            )
            results[agent_name] = summary
        except Exception as e:
            print(f"Error evaluating {agent_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    for agent_name, summary in results.items():
        if summary:
            print(f"{agent_name}: {summary.get('success_rate', 0)*100:.1f}% success")
    
    print(f"\nView traces at: https://logfire-eu.pydantic.dev/deal-ex-machina/open-finance")


if __name__ == "__main__":
    asyncio.run(main())
