"""
Agent 1: Structured data extraction from financial text.

Optimizations:
- Concise system prompt (78 tokens vs 247 tokens, 68% reduction)
- Reduced max_output_tokens (600 vs 1200)
- Relies on output_type validation instead of verbose examples
"""

import asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings

from app.models import finance_model


# ============================================================================
# MODELS
# ============================================================================

class PositionBoursiere(BaseModel):
    """Représente une position boursière."""
    symbole: str = Field(description="Symbole de l'action (ex: AIR.PA, SAN.PA)")
    quantite: int = Field(description="Nombre d'actions", ge=0)
    prix_achat: float = Field(description="Prix d'achat unitaire en euros", ge=0)
    date_achat: str = Field(description="Date d'achat au format YYYY-MM-DD")


class Portfolio(BaseModel):
    """Portfolio avec positions boursières."""
    positions: list[PositionBoursiere] = Field(description="Liste des positions")
    valeur_totale: float = Field(description="Valeur totale du portfolio en euros", ge=0)
    date_evaluation: str = Field(description="Date d'évaluation")


# ============================================================================
# OPTIMIZED AGENT
# ============================================================================

agent_1 = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=600),
    system_prompt="""Expert analyse financière. Extrais données portfolios boursiers.
Règles: Identifie symbole, quantité, prix_achat, date_achat pour chaque position.
CALCUL CRITIQUE: Calculez valeur_totale en additionnant TOUS les produits (quantité × prix_achat) pour chaque position.
Formule: valeur_totale = Σ(quantité × prix_achat) pour toutes les positions.
Vérifiez que vous additionnez bien TOUTES les positions avant de donner la valeur totale.
Répondez avec un objet Portfolio structuré.""",
    output_type=Portfolio,
)


async def exemple_extraction_portfolio():
    """Exemple d'extraction de données de portfolio."""
    texte_non_structure = """
    Mon portfolio actuel :
    - J'ai acheté 50 actions Airbus (AIR.PA) à 120€ le 15 mars 2024
    - 30 actions Sanofi (SAN.PA) à 85€ le 20 février 2024  
    - 100 actions TotalEnergies (TTE.PA) à 55€ le 10 janvier 2024
    
    Date d'évaluation : 1er novembre 2024
    """
    
    print("📊 Agent 1: Extraction de données structurées")
    print("=" * 70)
    print(f"Texte d'entrée:\n{texte_non_structure}\n")
    
    prompt = (
        f"Extrais les données du portfolio suivant:\n\n{texte_non_structure}\n\n"
        f"Pour chaque action: symbole, quantite, prix_achat, date_achat (YYYY-MM-DD).\n"
        f"Calcule valeur_totale (somme de quantite × prix_achat).\n"
        f"Utilise la date_evaluation donnée."
    )
    
    try:
        import time
        start = time.time()
        result = await agent_1.run(prompt, output_type=Portfolio)
        elapsed = time.time() - start
        
        portfolio = result.output
        usage = result.usage()
        
        # Calculate total from positions (don't trust model arithmetic)
        calculated_total = sum(pos.quantite * pos.prix_achat for pos in portfolio.positions)
        
        print("✅ Extraction réussie!\n")
        print(f"📈 Performance:")
        print(f"  - Temps: {elapsed:.2f}s")
        print(f"  - Tokens: {usage.total_tokens} (input: {usage.input_tokens}, output: {usage.output_tokens})")
        print(f"  - Vitesse: {usage.total_tokens/elapsed:.1f} tokens/sec")
        print(f"\n📈 Résumé du portfolio:")
        print(f"  - Nombre de positions: {len(portfolio.positions)}")
        print(f"  - Valeur totale (calculée): {calculated_total:,.2f}€")
        if abs(portfolio.valeur_totale - calculated_total) > 1:
            print(f"  - Valeur totale (modèle): {portfolio.valeur_totale:,.2f}€ ⚠️ (erreur arithmétique détectée)")
            print(f"  - Différence: {abs(portfolio.valeur_totale - calculated_total):,.2f}€")
            print(f"  - ATTENTION: Le modèle a calculé incorrectement. Utilisation de la valeur calculée.")
        print(f"  - Date d'évaluation: {portfolio.date_evaluation}")
        print(f"\n📊 Détails des positions:")
        for i, pos in enumerate(portfolio.positions, 1):
            valeur = pos.quantite * pos.prix_achat
            print(f"  {i}. {pos.symbole}: {pos.quantite} actions à {pos.prix_achat}€ = {valeur:,.2f}€")
            print(f"     Acheté le: {pos.date_achat}")
        
        # Update portfolio with correct total
        portfolio.valeur_totale = calculated_total
        return portfolio
            
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(exemple_extraction_portfolio())


