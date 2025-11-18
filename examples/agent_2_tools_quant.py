"""
Agent 2 Quant: Agent avec outils QuantLib pour calculs avancés de risque

Cet agent utilise QuantLib-Python et des méthodes statistiques avancées pour:
- Value at Risk (VaR) - Parametric, Historical, Monte Carlo
- Position Risk Metrics - Volatility, Correlation, Portfolio Risk
- Risk-adjusted Returns - Sharpe Ratio, Information Ratio
- Portfolio Analytics - Diversification, Concentration Risk

Destiné aux professionnels de la gestion d'actifs et du risk management.
"""

import asyncio
from typing import List
from pydantic_ai import Agent, ModelSettings, Tool
import numpy as np
from scipy import stats

try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False
    # QuantLib is optional - the agent works without it for basic risk calculations

from app.models import finance_model


# ============================================================================
# OUTILS DE RISQUE AVANCÉS
# ============================================================================

def calculer_var_parametrique(
    positions: List[float],
    volatilites: List[float],
    correlations: List[List[float]],
    niveau_confiance: float = 0.95,
    horizon_jours: int = 1
) -> str:
    """OBLIGATOIRE: Calcule la Value at Risk (VaR) paramétrique d'un portfolio.
    
    UTILISEZ CET OUTIL pour calculer la VaR avec la méthode variance-covariance (Markowitz).
    Cette méthode est le standard de l'industrie pour le calcul rapide de VaR.
    
    Exemple d'utilisation:
    - positions: [1000000, 500000, 300000] (valeurs en euros)
    - volatilites: [0.18, 0.05, 0.25] (volatilités annuelles: 18%, 5%, 25%)
    - correlations: [[1.0, 0.2, 0.4], [0.2, 1.0, -0.1], [0.4, -0.1, 1.0]] (matrice de corrélation)
    - niveau_confiance: 0.95 pour 95% ou 0.99 pour 99%
    - horizon_jours: 1 pour 1 jour, 10 pour 10 jours
    
    Args:
        positions: Liste des valeurs des positions en euros [position1, position2, ...]
        volatilites: Liste des volatilités annuelles [vol1, vol2, ...] (ex: 0.18 pour 18%)
        correlations: Matrice de corrélation NxN (liste de listes)
        niveau_confiance: Niveau de confiance (0.95 pour 95%, 0.99 pour 99%)
        horizon_jours: Horizon temporel en jours (défaut: 1 jour)
    
    Returns:
        VaR calculée avec métriques détaillées (VaR absolue, relative, volatilité, contributions)
    """
    positions = np.array(positions)
    volatilites = np.array(volatilites)
    corr_matrix = np.array(correlations)
    
    # Vérifier les dimensions
    n = len(positions)
    if len(volatilites) != n:
        return f"Erreur: {len(positions)} positions mais {len(volatilites)} volatilités"
    if corr_matrix.shape != (n, n):
        return f"Erreur: Matrice de corrélation doit être {n}x{n}"
    
    # Calculer la variance du portfolio
    # Variance = w' * Σ * w
    # où w = vecteur de poids, Σ = matrice variance-covariance
    cov_matrix = np.outer(volatilites, volatilites) * corr_matrix
    portfolio_value = np.sum(np.abs(positions))
    weights = positions / portfolio_value if portfolio_value > 0 else np.zeros_like(positions)
    portfolio_variance = weights.T @ cov_matrix @ weights
    
    # Ajuster pour l'horizon temporel (scaling)
    portfolio_std_daily = np.sqrt(portfolio_variance) * np.sqrt(horizon_jours / 252)
    
    # Calculer la VaR (z-score pour le niveau de confiance)
    z_score = stats.norm.ppf(1 - niveau_confiance)
    var_absolute = abs(z_score * portfolio_std_daily * portfolio_value)
    var_percentage = (var_absolute / portfolio_value) * 100 if portfolio_value > 0 else 0
    
    # Calculer les contributions individuelles au risque
    marginal_var = cov_matrix @ weights
    component_var = (
        (weights * marginal_var / portfolio_std_daily) * portfolio_value
        if portfolio_std_daily > 0 else np.zeros(n)
    )
    
    return (
        f"Value at Risk (Paramétrique) - {niveau_confiance*100:.0f}% confiance, {horizon_jours} jour(s):\n"
        f"  VaR absolue: {var_absolute:,.2f}€\n"
        f"  VaR relative: {var_percentage:.2f}% du portfolio\n"
        f"  Volatilité portfolio (annualisée): {np.sqrt(portfolio_variance)*100:.2f}%\n"
        f"  Volatilité portfolio (horizon {horizon_jours}j): {portfolio_std_daily*100:.2f}%\n"
        f"  Valeur totale portfolio: {portfolio_value:,.2f}€\n"
        f"  Z-score: {z_score:.4f}\n"
        f"  Contributions au risque par position:\n" +
        "\n".join([f"    Position {i+1}: {component_var[i]:,.2f}€ ({abs(component_var[i])/var_absolute*100:.1f}%)" 
                  for i in range(n) if var_absolute > 0])
    )


def calculer_var_historique(
    rendements_historiques: List[List[float]],
    positions: List[float],
    niveau_confiance: float = 0.95,
    horizon_jours: int = 1
) -> str:
    """OBLIGATOIRE: Calcule la Value at Risk (VaR) historique d'un portfolio.
    
    UTILISEZ CET OUTIL pour calculer la VaR basée sur les rendements historiques observés.
    Avantage: Pas d'hypothèse de distribution normale.
    
    Exemple d'utilisation:
    - rendements_historiques: [[0.01, -0.02, 0.015, ...], [0.005, -0.01, 0.008, ...], ...]
      (une liste par actif contenant les rendements quotidiens historiques)
    - positions: [1000000, 500000, 300000] (valeurs en euros)
    - niveau_confiance: 0.95 pour 95%
    - horizon_jours: 1 pour 1 jour, 10 pour 10 jours
    
    Args:
        rendements_historiques: Liste de listes - chaque sous-liste contient les rendements quotidiens d'un actif
        positions: Liste des valeurs des positions en euros
        niveau_confiance: Niveau de confiance (0.95 pour 95%, 0.99 pour 99%)
        horizon_jours: Horizon temporel en jours
    
    Returns:
        VaR historique avec Expected Shortfall (CVaR) et statistiques
    """
    rendements = np.array(rendements_historiques).T  # Transposer pour avoir (jours, actifs)
    positions = np.array(positions)
    
    if rendements.shape[1] != len(positions):
        return f"Erreur: {rendements.shape[1]} actifs dans rendements mais {len(positions)} positions"
    
    # Calculer les rendements du portfolio
    portfolio_returns = rendements @ positions
    
    # Ajuster pour l'horizon (scaling des rendements)
    if horizon_jours > 1:
        # Approximation: multiplier par sqrt(horizon)
        portfolio_returns_scaled = portfolio_returns * np.sqrt(horizon_jours)
    else:
        portfolio_returns_scaled = portfolio_returns
    
    # Calculer la VaR (percentile)
    percentile = (1 - niveau_confiance) * 100
    var_absolute = abs(np.percentile(portfolio_returns_scaled, percentile))
    
    # Statistiques
    portfolio_value = np.sum(np.abs(positions))
    var_percentage = (var_absolute / portfolio_value) * 100 if portfolio_value > 0 else 0
    mean_return = np.mean(portfolio_returns_scaled)
    std_return = np.std(portfolio_returns_scaled)
    
    # Expected Shortfall (Conditional VaR)
    tail_losses = portfolio_returns_scaled[portfolio_returns_scaled <= -var_absolute]
    expected_shortfall = abs(np.mean(tail_losses)) if len(tail_losses) > 0 else var_absolute
    
    return (
        f"Value at Risk (Historique) - {niveau_confiance*100:.0f}% confiance, {horizon_jours} jour(s):\n"
        f"  VaR absolue: {var_absolute:,.2f}€\n"
        f"  VaR relative: {var_percentage:.2f}% du portfolio\n"
        f"  Expected Shortfall (CVaR): {expected_shortfall:,.2f}€\n"
        f"  Rendement moyen (horizon): {mean_return*100:.2f}%\n"
        f"  Volatilité observée: {std_return*100:.2f}%\n"
        f"  Période d'observation: {len(rendements)} jours\n"
        f"  Percentile utilisé: {percentile:.1f}%"
    )


def calculer_var_monte_carlo(
    positions: List[float],
    volatilites: List[float],
    correlations: List[List[float]],
    rendements_moyens: List[float],
    niveau_confiance: float = 0.95,
    horizon_jours: int = 1,
    simulations: int = 10000
) -> str:
    """OBLIGATOIRE: Calcule la Value at Risk (VaR) par simulation Monte Carlo.
    
    UTILISEZ CET OUTIL pour calculer la VaR avec simulation stochastique.
    Plus flexible que la méthode paramétrique, permet de modéliser des distributions complexes.
    
    Exemple d'utilisation:
    - positions: [1000000, 500000, 300000] (valeurs en euros)
    - volatilites: [0.18, 0.05, 0.25] (volatilités annuelles)
    - correlations: [[1.0, 0.2, 0.4], [0.2, 1.0, -0.1], [0.4, -0.1, 1.0]]
    - rendements_moyens: [0.08, 0.03, 0.12] (rendements moyens annuels attendus)
    - niveau_confiance: 0.95 pour 95%
    - horizon_jours: 1 pour 1 jour, 10 pour 10 jours
    - simulations: 10000 (nombre de simulations, défaut: 10000)
    
    Args:
        positions: Liste des valeurs des positions en euros
        volatilites: Liste des volatilités annuelles
        correlations: Matrice de corrélation NxN
        rendements_moyens: Liste des rendements moyens annuels attendus par actif
        niveau_confiance: Niveau de confiance (0.95 pour 95%, 0.99 pour 99%)
        horizon_jours: Horizon temporel en jours
        simulations: Nombre de simulations Monte Carlo (défaut: 10000)
    
    Returns:
        VaR Monte Carlo avec Expected Shortfall et statistiques de simulation
    """
    positions = np.array(positions)
    volatilites = np.array(volatilites)
    corr_matrix = np.array(correlations)
    means = np.array(rendements_moyens)
    
    n = len(positions)
    
    # Générer des rendements corrélés (Cholesky decomposition)
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        return "Erreur: Matrice de corrélation non définie positive"
    
    # Simulation Monte Carlo
    np.random.seed(42)  # Pour reproductibilité
    portfolio_values = []
    
    for _ in range(simulations):
        # Générer des chocs aléatoires corrélés
        random_shocks = np.random.normal(0, 1, n)
        correlated_shocks = L @ random_shocks
        
        # Calculer les rendements simulés
        dt = horizon_jours / 252
        returns = means * dt + volatilites * np.sqrt(dt) * correlated_shocks
        
        # Valeur du portfolio après choc
        portfolio_value = np.sum(positions * (1 + returns))
        portfolio_values.append(portfolio_value)
    
    portfolio_values = np.array(portfolio_values)
    initial_value = np.sum(positions)
    pnl = portfolio_values - initial_value
    
    # Calculer la VaR
    percentile = (1 - niveau_confiance) * 100
    var_absolute = abs(np.percentile(pnl, percentile))
    
    # Statistiques
    portfolio_value_total = initial_value
    var_percentage = (var_absolute / portfolio_value_total) * 100 if portfolio_value_total > 0 else 0
    mean_pnl = np.mean(pnl)
    std_pnl = np.std(pnl)
    
    # Expected Shortfall
    tail_losses = pnl[pnl <= -var_absolute]
    expected_shortfall = abs(np.mean(tail_losses)) if len(tail_losses) > 0 else var_absolute
    
    return (
        f"Value at Risk (Monte Carlo) - {niveau_confiance*100:.0f}% confiance, {horizon_jours} jour(s):\n"
        f"  VaR absolue: {var_absolute:,.2f}€\n"
        f"  VaR relative: {var_percentage:.2f}% du portfolio\n"
        f"  Expected Shortfall (CVaR): {expected_shortfall:,.2f}€\n"
        f"  P&L moyen simulé: {mean_pnl:,.2f}€\n"
        f"  Écart-type P&L: {std_pnl:,.2f}€\n"
        f"  Simulations: {simulations:,}\n"
        f"  Probabilité de perte > VaR: {(1-niveau_confiance)*100:.1f}%"
    )


def calculer_risque_portfolio(
    positions: List[float],
    volatilites: List[float],
    correlations: List[List[float]],
    rendements_attendus: List[float]
) -> str:
    """OBLIGATOIRE: Calcule les métriques complètes de risque d'un portfolio.
    
    UTILISEZ CET OUTIL pour analyser le risque global d'un portfolio.
    Inclut: volatilité, corrélation moyenne, diversification (HHI), concentration, contributions au risque.
    
    Exemple d'utilisation:
    - positions: [1000000, 500000, 300000] (valeurs en euros)
    - volatilites: [0.18, 0.05, 0.25] (volatilités annuelles)
    - correlations: [[1.0, 0.2, 0.4], [0.2, 1.0, -0.1], [0.4, -0.1, 1.0]]
    - rendements_attendus: [0.08, 0.03, 0.12] (rendements attendus annuels)
    
    Args:
        positions: Liste des valeurs des positions en euros
        volatilites: Liste des volatilités annuelles
        correlations: Matrice de corrélation NxN
        rendements_attendus: Liste des rendements attendus annuels par actif
    
    Returns:
        Analyse complète: volatilité, Sharpe Ratio, diversification, contributions au risque
    """
    positions = np.array(positions)
    volatilites = np.array(volatilites)
    corr_matrix = np.array(correlations)
    returns = np.array(rendements_attendus)
    
    n = len(positions)
    portfolio_value = np.sum(np.abs(positions))
    weights = positions / portfolio_value if portfolio_value > 0 else np.zeros(n)
    
    # Volatilité du portfolio
    cov_matrix = np.outer(volatilites, volatilites) * corr_matrix
    portfolio_variance = weights.T @ cov_matrix @ weights
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Rendement attendu du portfolio
    portfolio_return = np.sum(weights * returns)
    
    # Corrélation moyenne
    # Extraire la partie triangulaire supérieure (sans la diagonale)
    upper_triangle = np.triu(corr_matrix, k=1)
    avg_correlation = np.mean(upper_triangle[upper_triangle != 0]) if np.any(upper_triangle) else 0
    
    # Indice de concentration (Herfindahl-Hirschman Index)
    hhi = np.sum(weights ** 2)
    diversification_ratio = 1 / hhi if hhi > 0 else 0
    
    # Contribution au risque par position
    marginal_contrib = cov_matrix @ weights
    component_risk = weights * marginal_contrib / portfolio_volatility if portfolio_volatility > 0 else np.zeros(n)
    
    # Sharpe Ratio (supposant taux sans risque = 0)
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
    
    return (
        f"Analyse de Risque du Portfolio:\n"
        f"  Valeur totale: {portfolio_value:,.2f}€\n"
        f"  Nombre de positions: {n}\n\n"
        f"Rendement et Risque:\n"
        f"  Rendement attendu (annualisé): {portfolio_return*100:.2f}%\n"
        f"  Volatilité (annualisée): {portfolio_volatility*100:.2f}%\n"
        f"  Sharpe Ratio: {sharpe_ratio:.3f}\n\n"
        f"Diversification:\n"
        f"  Corrélation moyenne: {avg_correlation:.3f}\n"
        f"  Indice de concentration (HHI): {hhi:.3f}\n"
        f"  Ratio de diversification: {diversification_ratio:.2f}\n"
        f"  (1 = parfaitement diversifié, {n} = concentré)\n\n"
        f"Contributions au risque:\n" +
        "\n".join([f"  Position {i+1}: {component_risk[i]*100:.2f}% (poids: {weights[i]*100:.1f}%)" 
                  for i in range(n)])
    )


def calculer_metrics_risque_ajuste(
    rendements_portfolio: List[float],
    rendements_benchmark: List[float],
    taux_sans_risque: float = 0.0
) -> str:
    """OBLIGATOIRE: Calcule les métriques de performance ajustée du risque.
    
    UTILISEZ CET OUTIL pour analyser la performance d'un fonds/portfolio par rapport à un benchmark.
    Inclut: Sharpe Ratio, Information Ratio, Tracking Error, Beta, Alpha (Jensen), Maximum Drawdown.
    
    Exemple d'utilisation:
    - rendements_portfolio: [0.02, -0.01, 0.015, 0.03, ...] (rendements mensuels du portfolio)
    - rendements_benchmark: [0.015, -0.008, 0.012, 0.025, ...] (rendements mensuels du benchmark)
    - taux_sans_risque: 0.0 (taux sans risque annuel, défaut: 0%)
    
    Args:
        rendements_portfolio: Liste des rendements historiques du portfolio (mensuels ou quotidiens)
        rendements_benchmark: Liste des rendements historiques du benchmark (même fréquence)
        taux_sans_risque: Taux sans risque annuel (défaut: 0.0 pour 0%)
    
    Returns:
        Métriques complètes: Sharpe, Information Ratio, Beta, Alpha, Maximum Drawdown
    """
    returns_p = np.array(rendements_portfolio)
    returns_b = np.array(rendements_benchmark)
    
    if len(returns_p) != len(returns_b):
        return f"Erreur: {len(returns_p)} rendements portfolio mais {len(returns_b)} rendements benchmark"
    
    # Statistiques de base
    mean_p = np.mean(returns_p) * 252  # Annualisé
    mean_b = np.mean(returns_b) * 252
    std_p = np.std(returns_p) * np.sqrt(252)
    std_b = np.std(returns_b) * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe_p = (mean_p - taux_sans_risque) / std_p if std_p > 0 else 0
    sharpe_b = (mean_b - taux_sans_risque) / std_b if std_b > 0 else 0
    
    # Tracking Error
    active_returns = returns_p - returns_b
    tracking_error = np.std(active_returns) * np.sqrt(252)
    
    # Information Ratio
    active_return_mean = np.mean(active_returns) * 252
    information_ratio = active_return_mean / tracking_error if tracking_error > 0 else 0
    
    # Beta
    covariance = np.cov(returns_p, returns_b)[0, 1]
    variance_b = np.var(returns_b)
    beta = covariance / variance_b if variance_b > 0 else 0
    
    # Alpha (Jensen's Alpha)
    alpha = mean_p - (taux_sans_risque + beta * (mean_b - taux_sans_risque))
    
    # Maximum Drawdown
    cumulative_p = np.cumprod(1 + returns_p)
    running_max = np.maximum.accumulate(cumulative_p)
    drawdown = (cumulative_p - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    return (
        f"Métriques de Performance Ajustée du Risque:\n\n"
        f"Rendements (annualisés):\n"
        f"  Portfolio: {mean_p*100:.2f}%\n"
        f"  Benchmark: {mean_b*100:.2f}%\n"
        f"  Actif: {active_return_mean*100:.2f}%\n\n"
        f"Risque:\n"
        f"  Volatilité portfolio: {std_p*100:.2f}%\n"
        f"  Volatilité benchmark: {std_b*100:.2f}%\n"
        f"  Tracking Error: {tracking_error*100:.2f}%\n\n"
        f"Ratios de Performance:\n"
        f"  Sharpe Ratio (portfolio): {sharpe_p:.3f}\n"
        f"  Sharpe Ratio (benchmark): {sharpe_b:.3f}\n"
        f"  Information Ratio: {information_ratio:.3f}\n\n"
        f"Exposition au Benchmark:\n"
        f"  Beta: {beta:.3f}\n"
        f"  Alpha (Jensen): {alpha*100:.2f}%\n\n"
        f"Risque de Drawdown:\n"
        f"  Maximum Drawdown: {max_drawdown*100:.2f}%"
    )


# Agent avec outils QuantLib et risque avancé
quant_risk_agent = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=3000),
    system_prompt=(
        "Vous êtes un analyste quantitatif senior spécialisé en risk management et gestion d'actifs.\n\n"
        "RÈGLES ABSOLUES - VOUS DEVEZ:\n"
        "1. TOUJOURS appeler les outils disponibles pour TOUS les calculs de risque\n"
        "2. NE JAMAIS calculer manuellement - UTILISEZ LES OUTILS DIRECTEMENT\n"
        "3. Pour VaR paramétrique → APPELER calculer_var_parametrique avec positions, volatilités, corrélations\n"
        "4. Pour VaR historique → APPELER calculer_var_historique avec rendements historiques\n"
        "5. Pour VaR Monte Carlo → APPELER calculer_var_monte_carlo avec tous les paramètres\n"
        "6. Pour analyse portfolio → APPELER calculer_risque_portfolio\n"
        "7. Pour métriques ajustées → APPELER calculer_metrics_risque_ajuste\n\n"
        "NE PAS EXPLIQUER COMMENT CALCULER - APPELER LES OUTILS IMMÉDIATEMENT.\n"
        "Les outils retournent les résultats complets - utilisez-les directement.\n\n"
        "COMMUNICATION:\n"
        "- Présentez les résultats des outils de manière claire\n"
        "- Utilisez la terminologie financière professionnelle\n"
        "- Expliquez les implications pratiques\n\n"
        "Répondez en français de manière structurée après avoir appelé les outils."
    ),
    tools=[
        Tool(
            calculer_var_parametrique,
            name="calculer_var_parametrique",
            description="OBLIGATOIRE pour calculer VaR paramétrique. Utilisez cette fonction pour calculer la Value at Risk avec la méthode variance-covariance (Markowitz). Fournissez positions, volatilités, corrélations, niveau de confiance et horizon.",
        ),
        Tool(
            calculer_var_historique,
            name="calculer_var_historique",
            description="OBLIGATOIRE pour calculer VaR historique. Utilisez cette fonction pour calculer la VaR basée sur les rendements historiques observés. Fournissez rendements historiques par actif, positions, niveau de confiance et horizon.",
        ),
        Tool(
            calculer_var_monte_carlo,
            name="calculer_var_monte_carlo",
            description="OBLIGATOIRE pour calculer VaR Monte Carlo. Utilisez cette fonction pour calculer la VaR par simulation stochastique. Fournissez positions, volatilités, corrélations, rendements moyens, niveau de confiance, horizon et nombre de simulations.",
        ),
        Tool(
            calculer_risque_portfolio,
            name="calculer_risque_portfolio",
            description="OBLIGATOIRE pour analyser le risque d'un portfolio. Utilisez cette fonction pour calculer volatilité, Sharpe Ratio, diversification, et contributions au risque. Fournissez positions, volatilités, corrélations et rendements attendus.",
        ),
        Tool(
            calculer_metrics_risque_ajuste,
            name="calculer_metrics_risque_ajuste",
            description="OBLIGATOIRE pour calculer les métriques de performance ajustée du risque. Utilisez cette fonction pour Sharpe Ratio, Information Ratio, Beta, Alpha, Maximum Drawdown. Fournissez rendements portfolio, rendements benchmark, et taux sans risque.",
        ),
    ],
)


async def exemple_var_analysis():
    """Exemple d'analyse VaR pour un portfolio professionnel."""
    print("\n" + "=" * 70)
    print("Agent 2 Quant: Analyse de Risque Professionnelle")
    print("=" * 70)
    
    question = (
        "Calculez la VaR paramétrique pour mon portfolio:\n"
        "- Positions: [1000000, 500000, 300000] euros\n"
        "- Volatilités: [0.18, 0.05, 0.25] (18%, 5%, 25%)\n"
        "- Corrélations: [[1.0, 0.2, 0.4], [0.2, 1.0, -0.1], [0.4, -0.1, 1.0]]\n"
        "- Niveau de confiance: 0.95 (95%)\n"
        "- Horizon: 1 jour\n\n"
        "Puis calculez aussi pour 99% et 10 jours. "
        "Ensuite, calculez la VaR Monte Carlo avec rendements moyens [0.08, 0.03, 0.12] et 10000 simulations. "
        "Enfin, analysez le risque complet du portfolio avec rendements attendus [0.08, 0.03, 0.12]."
    )
    
    print(f"\nQuestion:\n{question}\n")
    print("-" * 70)
    
    result = await quant_risk_agent.run(question)
    
    print("\nRéponse de l'agent:\n")
    print(result.output)
    print()
    
    # Vérifier les tool calls
    print("\n" + "=" * 70)
    print("Vérification des Tool Calls")
    print("=" * 70)
    
    tool_calls_found = False
    if hasattr(result, 'all_messages'):
        try:
            messages = list(result.all_messages())
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_calls_found = True
                    print(f"✓ {len(msg.tool_calls)} tool call(s) détecté(s)")
                    tools_used = []
                    for tc in msg.tool_calls:
                        if hasattr(tc, 'function') and hasattr(tc.function, 'name'):
                            tools_used.append(tc.function.name)
                    if tools_used:
                        print(f"  Outils utilisés: {', '.join(tools_used)}")
        except Exception as e:
            print(f"  Erreur: {e}")
    
    if not tool_calls_found:
        print("⚠ Aucun tool call détecté")
    
    print("=" * 70)


async def exemple_risk_metrics():
    """Exemple de calcul de métriques de risque ajusté."""
    print("\n\n" + "=" * 70)
    print("Exemple: Métriques de Performance Ajustée du Risque")
    print("=" * 70)
    
    question = (
        "J'ai un fonds qui a généré les rendements suivants sur 12 mois:\n"
        "[0.02, -0.01, 0.015, 0.03, -0.005, 0.025, 0.01, -0.02, 0.035, 0.015, 0.02, 0.01]\n\n"
        "Le benchmark a généré:\n"
        "[0.015, -0.008, 0.012, 0.025, -0.003, 0.02, 0.008, -0.015, 0.03, 0.012, 0.018, 0.008]\n\n"
        "Calculez toutes les métriques de performance ajustée du risque, "
        "incluant Sharpe Ratio, Information Ratio, Beta, Alpha, et Maximum Drawdown."
    )
    
    print(f"\nQuestion:\n{question}\n")
    print("-" * 70)
    
    result = await quant_risk_agent.run(question)
    
    print("\nRéponse de l'agent:\n")
    print(result.output)
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(exemple_var_analysis())
    asyncio.run(exemple_risk_metrics())

