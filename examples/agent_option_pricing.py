"""Agent d'option pricing utilisant QuantLib via PydanticAI tools."""

import asyncio
from contextlib import contextmanager
from typing import List

from pydantic_ai import Agent, ModelSettings, Tool

try:
    import QuantLib as ql
except ImportError as exc:  # pragma: no cover - QuantLib disponible dans l'env
    raise RuntimeError("QuantLib-Python est requis pour cet exemple") from exc

from app.models import finance_model


@contextmanager
def _ql_evaluation_date(evaluation_date: "ql.Date"):
    """Scope QuantLib evaluationDate mutations to avoid leaking global state."""
    settings = ql.Settings.instance()
    previous_date = settings.evaluationDate
    try:
        settings.evaluationDate = evaluation_date
        yield
    finally:
        settings.evaluationDate = previous_date


def calculer_prix_call_black_scholes(
    spot: float,
    strike: float,
    maturite_annees: float,
    taux_sans_risque: float,
    volatilite: float,
    dividende: float = 0.0,
) -> str:
    """OBLIGATOIRE: Prix d'un call européen via Black-Scholes (QuantLib).

    Args:
        spot: Prix spot du sous-jacent (en devise du contrat)
        strike: Prix d'exercice
        maturite_annees: Temps jusqu'à maturité en années (ex: 0.5)
        taux_sans_risque: Taux sans risque annualisé (ex: 0.02 pour 2%)
        volatilite: Volatilité annualisée (ex: 0.25)
        dividende: Rendement de dividende annualisé (défaut: 0)

    Returns:
        Prix du call + grecques principales.
    """
    evaluation_date = ql.Date.todaysDate()
    with _ql_evaluation_date(evaluation_date):
        settlement_date = evaluation_date

        calendar = ql.NullCalendar()
        day_count = ql.Actual365Fixed()

        payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
        maturity_date = evaluation_date + int(maturite_annees * 365)
        exercise = ql.EuropeanExercise(maturity_date)

        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(settlement_date, taux_sans_risque, day_count))
        dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(settlement_date, dividende, day_count))
        vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(settlement_date, calendar, volatilite, day_count))

        process = ql.BlackScholesMertonProcess(spot_handle, dividend_ts, flat_ts, vol_ts)
        option = ql.VanillaOption(payoff, exercise)
        engine = ql.AnalyticEuropeanEngine(process)
        option.setPricingEngine(engine)

        price = option.NPV()
        delta = option.delta()
        gamma = option.gamma()
        vega = option.vega()
        theta = option.theta()

    return (
        f"Prix call (Black-Scholes): {price:,.4f}\n"
        f"Delta: {delta:.4f}\n"
        f"Gamma: {gamma:.6f}\n"
        f"Vega: {vega:.4f}\n"
        f"Theta: {theta:.4f}"
    )


option_agent = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=1200),
    system_prompt=(
        "Vous êtes un ingénieur financier spécialisé en pricing d'options.\n"
        "Utilisez EXCLUSIVEMENT les outils fournis pour tous les calculs.\n"
        "Pour un call européen → APPELEZ calculer_prix_call_black_scholes avec les paramètres exacts.\n"
        "N'expliquez pas comment calculer, présentez seulement les résultats fournis par l'outil."
    ),
    tools=[
        Tool(
            calculer_prix_call_black_scholes,
            name="calculer_prix_call_black_scholes",
            description=(
                "OBLIGATOIRE pour le pricing d'un call européen via Black-Scholes."
                " Fournissez spot, strike, maturité (en années), taux sans risque, volatilité, dividende."
            ),
        )
    ],
)


async def exemple_pricing_call():
    question = (
        "Calcule le prix d'un call européen sur l'indice X:\n"
        "- Spot: 100\n"
        "- Strike: 105\n"
        "- Maturité: 0.5 an\n"
        "- Taux sans risque: 0.02\n"
        "- Volatilité: 0.25\n"
        "- Dividende: 0.01"
    )
    result = await option_agent.run(question)
    print("Question:\n", question)
    print("\nRéponse:\n", result.output)
    print("\nTool calls détectés:")
    tool_used = False
    for msg in result.all_messages():
        if getattr(msg, "tool_calls", None):
            tool_used = True
            print(msg.tool_calls)
    if not tool_used:
        print("⚠ Aucun")


if __name__ == "__main__":
    asyncio.run(exemple_pricing_call())
