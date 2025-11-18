"""Wrapper qui exécute agent_2 (calc financier) et passe un contrôle compliance."""

import asyncio
from typing import List

from pydantic_ai import Agent, ModelSettings

from .agent_2_tools import finance_calculator_agent  # type: ignore  # noqa: E402
from app.models import finance_model


async def run_finance_agent(question: str):
    """Exécute l'agent financier et retourne (result, tool_calls_log)."""
    result = await finance_calculator_agent.run(question)

    tool_calls: List[str] = []
    for msg in result.all_messages():
        msg_calls = getattr(msg, "tool_calls", None) or []
        for call in msg_calls:
            name = None
            args = None
            if hasattr(call, "function"):
                func = call.function
                name = getattr(func, "name", None)
                args = getattr(func, "arguments", None)
            elif hasattr(call, "tool_name"):
                name = call.tool_name
                args = getattr(call, "args", None)
            if name is None and hasattr(call, "name"):
                name = call.name
            if name is None:
                continue

            normalized_args = args
            if normalized_args is not None and not isinstance(normalized_args, str):
                normalized_args = str(normalized_args)

            tool_calls.append(f"{name}: {normalized_args}")
    return result, tool_calls


compliance_agent = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=600),
    system_prompt=(
        "Tu es un contrôleur compliance.\n"
        "On te fournit: la question client, la réponse envoyée, et la liste des appels d'outils.\n"
        "Règles:\n"
        "1. Si la liste d'outils est vide → Non conforme.\n"
        "2. Sinon → Conforme, mentionner quels outils ont été déclenchés.\n"
        "3. Si la réponse mentionne un calcul non couvert par les outils, flag potential issue.\n"
        "Réponds en français, format court: 'Conforme' ou 'Non conforme' + justification." 
    ),
)


async def run_with_compliance(question: str):
    result, tool_calls = await run_finance_agent(question)

    compliance_input = (
        "QUESTION CLIENT:\n" + question + "\n\n"
        "RÉPONSE FOURNIE:\n" + result.output + "\n\n"
        "APPELS D'OUTILS:\n" + ("\n".join(tool_calls) if tool_calls else "Aucun")
    )

    compliance = await compliance_agent.run(compliance_input)

    print("QUESTION:\n", question)
    print("\nRÉPONSE AGENT:\n", result.output)
    print("\nAPPELS D'OUTILS DÉTECTÉS:")
    if tool_calls:
        for line in tool_calls:
            print(" -", line)
    else:
        print("⚠ Aucun (non conforme)")
    print("\nAVIS COMPLIANCE:\n", compliance.output)
    print("\n" + "-" * 70 + "\n")


async def main():
    questions = [
        "J'ai 25 000€ à 4% pendant 8 ans. Combien aurai-je?",
        "J'emprunte 150 000€ sur 15 ans à 2.8%. Quel est le versement mensuel?",
    ]
    for q in questions:
        await run_with_compliance(q)


if __name__ == "__main__":
    # À lancer avec: python -m examples.agent_2_compliance
    asyncio.run(main())
