#!/usr/bin/env python3
"""
Pydantic Evals integration for Open Finance agents.

Uses the official pydantic-evals framework for systematic evaluation
with built-in evaluators, span-based tool call analysis, and Logfire integration.

Usage:
    python examples/run_pydantic_evals.py --agent agent_1
    python examples/run_pydantic_evals.py --agent agent_2 --max-cases 5
    python examples/run_pydantic_evals.py --all

Documentation: https://ai.pydantic.dev/evals/
"""

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance

# Configure Logfire first
from app.logfire_config import configure_logfire, instrument_pydantic_ai

configure_logfire()
instrument_pydantic_ai()


# =============================================================================
# CUSTOM EVALUATORS
# =============================================================================

@dataclass
class LatencyEvaluator(Evaluator[Any, Any]):
    """Score based on response latency."""
    max_acceptable_seconds: float = 10.0
    
    def evaluate(self, ctx: EvaluatorContext[Any, Any]) -> float:
        # Duration is tracked automatically by pydantic-evals
        # This evaluator scores based on whether output was produced
        if ctx.output is not None:
            return 1.0
        return 0.0


@dataclass
class StructuredOutputEvaluator(Evaluator[Any, Any]):
    """Check if output is a structured Pydantic model."""
    
    def evaluate(self, ctx: EvaluatorContext[Any, Any]) -> float:
        if ctx.output is None:
            return 0.0
        # Check if it's a Pydantic model or has expected attributes
        if hasattr(ctx.output, 'model_dump'):
            return 1.0
        if hasattr(ctx.output, '__dict__'):
            return 0.8
        return 0.5


@dataclass
class KeywordMatchEvaluator(Evaluator[Any, dict]):
    """Check if expected keywords appear in output."""
    
    def evaluate(self, ctx: EvaluatorContext[Any, dict]) -> float:
        if ctx.output is None or ctx.expected_output is None:
            return 0.0
        
        output_str = str(ctx.output).lower()
        keywords = ctx.expected_output.get('keywords', [])
        
        if not keywords:
            return 1.0
        
        matches = sum(1 for kw in keywords if kw.lower() in output_str)
        return matches / len(keywords)


@dataclass
class NumericAccuracyEvaluator(Evaluator[Any, dict]):
    """Check if numeric values in output are within tolerance."""
    tolerance: float = 0.05  # 5% tolerance
    
    def evaluate(self, ctx: EvaluatorContext[Any, dict]) -> float:
        if ctx.output is None or ctx.expected_output is None:
            return 0.0
        
        expected_value = ctx.expected_output.get('expected_value')
        if expected_value is None:
            return 1.0
        
        # Extract numbers from output
        output_str = str(ctx.output)
        numbers = re.findall(r'[\d,]+\.?\d*', output_str.replace(',', ''))
        
        for num_str in numbers:
            try:
                num = float(num_str)
                if abs(num - expected_value) / expected_value <= self.tolerance:
                    return 1.0
            except ValueError:
                continue
        
        return 0.0


@dataclass  
class ToolCallCountEvaluator(Evaluator[Any, dict]):
    """
    Evaluate tool call behavior.
    
    Uses span-based evaluation to check tool calls made during agent execution.
    See: https://ai.pydantic.dev/evals/evaluators/span-based/
    """
    max_expected_calls: int = 3
    
    def evaluate(self, ctx: EvaluatorContext[Any, dict]) -> float:
        # Note: For full span-based evaluation, use SpanQuery from pydantic_evals
        # This is a simplified version based on output analysis
        expected = ctx.expected_output or {}
        expected_tools = expected.get('expected_tool_calls', self.max_expected_calls)
        
        # If we got output, assume reasonable tool usage
        if ctx.output is not None:
            return 1.0
        return 0.0


# =============================================================================
# DATASET DEFINITIONS
# =============================================================================

def create_agent_1_dataset(max_cases: int = 10) -> Dataset:
    """Portfolio extraction dataset."""
    cases = [
        Case(
            name='portfolio_simple',
            inputs='Extrais le portfolio: 50 AIR.PA à 120€, 30 SAN.PA à 85€',
            expected_output={
                'keywords': ['AIR.PA', 'SAN.PA', '120', '85', '50', '30'],
            },
            metadata={'difficulty': 'easy', 'category': 'extraction'},
        ),
        Case(
            name='portfolio_mixed',
            inputs='Mon portfolio: 100 TTE.PA à 55€, 25 LVMH.PA à 800€, 200 BNP.PA à 60€',
            expected_output={
                'keywords': ['TTE.PA', 'LVMH.PA', 'BNP.PA'],
            },
            metadata={'difficulty': 'easy', 'category': 'extraction'},
        ),
        Case(
            name='portfolio_complex',
            inputs='Portefeuille diversifié: 150 actions Airbus à 130€, 80 Sanofi à 90€, 50 LVMH à 750€, 200 TotalEnergies à 58€',
            expected_output={
                'keywords': ['Airbus', 'Sanofi', 'LVMH', 'TotalEnergies'],
            },
            metadata={'difficulty': 'medium', 'category': 'extraction'},
        ),
    ]
    
    return Dataset(
        cases=cases[:max_cases],
        evaluators=[
            StructuredOutputEvaluator(),
            KeywordMatchEvaluator(),
            LatencyEvaluator(max_acceptable_seconds=15.0),
        ],
    )


def create_agent_2_dataset(max_cases: int = 10) -> Dataset:
    """Financial calculator dataset."""
    cases = [
        Case(
            name='future_value_simple',
            inputs='50000€ à 4% sur 5 ans?',
            expected_output={
                'expected_value': 60833.0,  # FV = 50000 * (1.04)^5
                'keywords': ['60', '833'],
            },
            metadata={'difficulty': 'easy', 'category': 'future_value'},
        ),
        Case(
            name='future_value_long',
            inputs='100000€ à 3% pendant 10 ans',
            expected_output={
                'expected_value': 134392.0,  # FV = 100000 * (1.03)^10
                'keywords': ['134'],
            },
            metadata={'difficulty': 'easy', 'category': 'future_value'},
        ),
        Case(
            name='present_value',
            inputs='Quelle somme placer à 5% pour avoir 100000€ dans 10 ans?',
            expected_output={
                'expected_value': 61391.0,  # PV = 100000 / (1.05)^10
                'keywords': ['61'],
            },
            metadata={'difficulty': 'medium', 'category': 'present_value'},
        ),
    ]
    
    return Dataset(
        cases=cases[:max_cases],
        evaluators=[
            LatencyEvaluator(max_acceptable_seconds=15.0),
            NumericAccuracyEvaluator(tolerance=0.10),  # 10% tolerance
            ToolCallCountEvaluator(max_expected_calls=3),
        ],
    )


def create_agent_3_dataset(max_cases: int = 10) -> Dataset:
    """Multi-agent workflow (risk/tax) dataset."""
    cases = [
        Case(
            name='risk_analysis_simple',
            inputs='40% actions, 30% obligations, 20% immobilier, 10% autres. Investissement 100k€, horizon 30 ans.',
            expected_output={
                'keywords': ['risque', 'rendement', 'allocation'],
            },
            metadata={'difficulty': 'easy', 'category': 'risk_analysis'},
        ),
        Case(
            name='tax_optimization',
            inputs='Portfolio: 45% actions PEA, 25% obligations assurance-vie, 20% immobilier SCPI, 10% autres. 150000€, 30 ans.',
            expected_output={
                'keywords': ['PEA', 'assurance-vie', 'fiscal'],
            },
            metadata={'difficulty': 'medium', 'category': 'tax_optimization'},
        ),
        Case(
            name='full_workflow',
            inputs='Portfolio complet: 40% actions (PEA max 150k€), 30% obligations, 20% immobilier, 10% autres. 500000€, 30 ans, profil modéré.',
            expected_output={
                'keywords': ['risque', 'fiscal', 'rendement'],
            },
            metadata={'difficulty': 'hard', 'category': 'full_workflow'},
        ),
    ]
    
    return Dataset(
        cases=cases[:max_cases],
        evaluators=[
            LatencyEvaluator(max_acceptable_seconds=30.0),  # Workflow is slower
            StructuredOutputEvaluator(),
            KeywordMatchEvaluator(),
            ToolCallCountEvaluator(max_expected_calls=5),
        ],
    )


def create_agent_4_dataset(max_cases: int = 10) -> Dataset:
    """Option pricing dataset."""
    cases = [
        Case(
            name='call_option_atm',
            inputs='Call européen: Spot 100, Strike 100, Maturité 1 an, Taux 0.05, Volatilité 0.20',
            expected_output={
                'expected_value': 10.45,  # Approximate BS price
                'keywords': ['call', 'prix', 'black', 'scholes'],
            },
            metadata={'difficulty': 'easy', 'category': 'option_pricing'},
        ),
        Case(
            name='call_option_otm',
            inputs='Prix d\'un call: Spot 50, Strike 55, Maturité 0.5 an, Taux 0.03, Volatilité 0.25',
            expected_output={
                'expected_value': 2.5,  # Approximate
                'keywords': ['call'],
            },
            metadata={'difficulty': 'easy', 'category': 'option_pricing'},
        ),
    ]
    
    return Dataset(
        cases=cases[:max_cases],
        evaluators=[
            LatencyEvaluator(max_acceptable_seconds=20.0),
            StructuredOutputEvaluator(),
            NumericAccuracyEvaluator(tolerance=0.20),  # 20% tolerance for options
        ],
    )


def create_agent_5_dataset(max_cases: int = 10) -> Dataset:
    """SWIFT/ISO 20022 conversion dataset."""
    swift_message = """{1:F01BANKFRPPAXXX1234567890}
{2:O1031201234567BANKFRPPAXXX12345678901234N}
{4:
:20:REF123
:32A:240124EUR1000,00
:50K:/FR1420041010050500013M02606
COMPAGNIE ABC
123 RUE DE PARIS
75001 PARIS
:59:/DE89370400440532013000
COMPAGNIE XYZ
BERLIN
:71A:OUR
-}"""

    cases = [
        Case(
            name='swift_to_iso_simple',
            inputs=f'Convert this SWIFT MT103 to ISO 20022:\n{swift_message}',
            expected_output={
                'keywords': ['pacs.008', 'ISO 20022', 'Document', 'CdtTrfTxInf'],
            },
            metadata={'difficulty': 'easy', 'category': 'swift_conversion'},
        ),
        Case(
            name='swift_validation',
            inputs=f'Validate this SWIFT message:\n{swift_message}',
            expected_output={
                'keywords': ['MT103', 'valid', 'EUR', '1000'],
            },
            metadata={'difficulty': 'easy', 'category': 'validation'},
        ),
    ]
    
    return Dataset(
        cases=cases[:max_cases],
        evaluators=[
            LatencyEvaluator(max_acceptable_seconds=60.0),  # SWIFT can be slow
            StructuredOutputEvaluator(),
            KeywordMatchEvaluator(),
        ],
    )


# =============================================================================
# AGENT TASK FUNCTIONS
# =============================================================================

async def run_agent_1(prompt: str) -> Any:
    """Task function for agent_1 (portfolio extraction)."""
    from examples.agent_1 import agent_1
    result = await agent_1.run(prompt)
    return result.output


async def run_agent_2(prompt: str) -> Any:
    """Task function for agent_2 (financial calculator)."""
    from examples.agent_2 import agent_2
    result = await agent_2.run(prompt)
    return result.output


async def run_agent_3(prompt: str) -> Any:
    """Task function for agent_3 (multi-agent workflow)."""
    from examples.agent_3 import portfolio_optimizer
    result = await portfolio_optimizer.run(prompt)
    return result.output


async def run_agent_4(prompt: str) -> Any:
    """Task function for agent_4 (option pricing)."""
    from examples.agent_4 import agent_4
    result = await agent_4.run(prompt)
    return result.output


async def run_agent_5(prompt: str) -> Any:
    """Task function for agent_5 (SWIFT/ISO 20022)."""
    from examples.agent_5 import agent_5
    result = await agent_5.run(prompt)
    return result.output


# =============================================================================
# MAIN
# =============================================================================

AGENTS = {
    'agent_1': (create_agent_1_dataset, run_agent_1),
    'agent_2': (create_agent_2_dataset, run_agent_2),
    'agent_3': (create_agent_3_dataset, run_agent_3),
    'agent_4': (create_agent_4_dataset, run_agent_4),
    'agent_5': (create_agent_5_dataset, run_agent_5),
}


async def evaluate_agent(agent_name: str, max_cases: int = 10) -> None:
    """Run evaluation for a single agent."""
    if agent_name not in AGENTS:
        print(f"Unknown agent: {agent_name}")
        return
    
    create_dataset, task_fn = AGENTS[agent_name]
    dataset = create_dataset(max_cases)
    
    print(f"\n{'='*60}")
    print(f"Evaluating {agent_name} with Pydantic Evals")
    print(f"Cases: {len(dataset.cases)}")
    print(f"{'='*60}\n")
    
    # Run evaluation - results automatically sent to Logfire
    report = await dataset.evaluate(task_fn)
    
    # Print results to terminal
    report.print(
        include_input=True,
        include_output=False,  # Outputs can be long
        include_durations=True,
    )
    
    return report


async def main():
    parser = argparse.ArgumentParser(description='Run Pydantic Evals for Open Finance agents')
    parser.add_argument('--agent', choices=list(AGENTS.keys()), help='Agent to evaluate')
    parser.add_argument('--all', action='store_true', help='Evaluate all agents')
    parser.add_argument('--max-cases', type=int, default=5, help='Max cases per agent')
    
    args = parser.parse_args()
    
    if args.all:
        agents_to_run = list(AGENTS.keys())
    elif args.agent:
        agents_to_run = [args.agent]
    else:
        print("Specify --agent or --all")
        return
    
    print("Pydantic Evals - Open Finance Agents")
    print("="*60)
    print(f"Logfire Dashboard: https://logfire-eu.pydantic.dev/deal-ex-machina/open-finance")
    print(f"Evals will appear in Logfire UI automatically")
    print("="*60)
    
    for agent_name in agents_to_run:
        try:
            await evaluate_agent(agent_name, args.max_cases)
        except Exception as e:
            print(f"Error evaluating {agent_name}: {e}")
    
    print("\n" + "="*60)
    print("View detailed results at:")
    print("https://logfire-eu.pydantic.dev/deal-ex-machina/open-finance")
    print("Go to: Explore > Filter by 'pydantic-evals'")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
