"""
Tests pour Agent 3: Workflow multi-étapes

Teste chaque étape individuellement puis le workflow complet.
Version optimisée avec tests unitaires des outils et validation structurelle.
"""

import asyncio
from examples.agent_3_multi_step import (
    risk_analyst,
    tax_advisor,
    portfolio_optimizer,
    check_compliance,
    extract_tool_calls,
    calculer_rendement_portfolio,
    calculer_valeur_future_investissement,
    AnalyseRisque,
    AnalyseFiscale,
)


# ============================================================================
# TESTS UNITAIRES DES OUTILS (pas d'API)
# ============================================================================

def test_tools():
    """Test des outils financiers (sans API)."""
    print("\n" + "=" * 70)
    print("🧪 TEST 1: Outils financiers (unitaire)")
    print("=" * 70)
    
    # Test calculer_rendement_portfolio
    print("\n1. Test calculer_rendement_portfolio:")
    result1 = calculer_rendement_portfolio(
        allocation_actions=0.40,
        allocation_obligations=0.30,
        allocation_immobilier=0.20,
        allocation_autres=0.10
    )
    assert "Rendement attendu" in result1, "Devrait calculer le rendement"
    assert "40.0%" in result1 or "40%" in result1, "Devrait afficher l'allocation actions"
    print(f"   ✅ {result1.split(chr(10))[0]}")
    
    # Test calculer_valeur_future_investissement
    print("\n2. Test calculer_valeur_future_investissement:")
    result2 = calculer_valeur_future_investissement(
        capital_initial=100000,
        taux_annuel=0.05,
        duree_annees=30
    )
    assert "Valeur future" in result2, "Devrait calculer la valeur future"
    assert "100,000" in result2 or "100000" in result2, "Devrait mentionner le capital initial"
    print(f"   ✅ {result2.split(chr(10))[0]}")
    
    # Test validation allocation (doit échouer)
    print("\n3. Test validation allocation (doit détecter erreur):")
    result3 = calculer_rendement_portfolio(
        allocation_actions=0.50,
        allocation_obligations=0.30,
        allocation_immobilier=0.20,
        allocation_autres=0.10  # Total = 110%
    )
    assert "Erreur" in result3, "Devrait détecter l'erreur d'allocation"
    print(f"   ✅ Erreur détectée: {result3[:60]}...")
    
    print("\n✅ Tous les tests d'outils passés")


# ============================================================================
# TESTS DE STRUCTURE (validation des modèles)
# ============================================================================

def test_structured_models():
    """Test des modèles structurés."""
    print("\n" + "=" * 70)
    print("🧪 TEST 2: Modèles structurés")
    print("=" * 70)
    
    # Test AnalyseRisque
    print("\n1. Test modèle AnalyseRisque:")
    risk = AnalyseRisque(
        niveau_risque=3,
        facteurs_risque=["Volatilité élevée", "Concentration"],
        recommandation="Diversifier",
        justification="Portfolio concentré"
    )
    assert 1 <= risk.niveau_risque <= 5, "Niveau doit être entre 1 et 5"
    assert len(risk.facteurs_risque) > 0, "Doit avoir des facteurs"
    print(f"   ✅ Modèle valide: niveau={risk.niveau_risque}, facteurs={len(risk.facteurs_risque)}")
    
    # Test AnalyseFiscale
    print("\n2. Test modèle AnalyseFiscale:")
    tax = AnalyseFiscale(
        regime_fiscal="PEA",
        implications=["Exonération après 5 ans"],
        avantages=["Pas d'impôt sur les plus-values"],
        inconvenients=["Plafond de versement"],
        recommandation="Utiliser le PEA"
    )
    assert tax.regime_fiscal, "Doit avoir un régime fiscal"
    print(f"   ✅ Modèle valide: régime={tax.regime_fiscal}")
    
    print("\n✅ Tous les tests de modèles passés")


# ============================================================================
# TESTS DES AGENTS (avec API - optionnel)
# ============================================================================

async def test_risk_analyst_quick():
    """Test rapide de l'agent d'analyse de risque."""
    print("\n" + "=" * 70)
    print("🧪 TEST 3: Agent Risk Analyst (quick)")
    print("=" * 70)
    
    scenario = "Portfolio: 40% actions, 30% obligations, 20% immobilier, 10% crypto"
    
    print(f"\nScénario: {scenario}")
    print("Exécution...")
    
    try:
        result = await risk_analyst.run(
            f"Analyse le niveau de risque (1-5) de: {scenario}. "
            "Réponds brièvement avec niveau, facteurs principaux, recommandation. "
            "Utilise les outils si nécessaire."
        )
        
        # Vérifier structured output
        has_structured = hasattr(result, 'data') and result.data
        print(f"\n   Structured output: {'✅ Oui' if has_structured else '⚠️  Non'}")
        
        if has_structured:
            risk_analysis = result.data
            print(f"   Niveau: {risk_analysis.niveau_risque}/5")
            print(f"   Facteurs: {len(risk_analysis.facteurs_risque)}")
            print(f"   ✅ Validation Pydantic réussie!")
        else:
            # Afficher la sortie brute pour debug
            print(f"   Sortie brute: {result.output[:200]}...")
        
        # Vérifier tool calls
        tool_calls = extract_tool_calls(result)
        print(f"   Tool calls: {len(tool_calls)}")
        if tool_calls:
            print(f"   ✅ Outils utilisés: {tool_calls[0][:50]}...")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n   ❌ Erreur: {type(e).__name__}")
        print(f"   Message: {error_msg[:200]}...")
        
        # Essayer de capturer plus de détails sur l'erreur de validation
        if "422" in error_msg or "validation" in error_msg.lower():
            print(f"   ⚠️  Erreur de validation - le modèle n'a pas produit de JSON valide")
            print(f"   💡 Vérifiez que le prompt force bien le format JSON strict")
        
        return False


async def test_tax_advisor_quick():
    """Test rapide de l'agent conseiller fiscal."""
    print("\n" + "=" * 70)
    print("🧪 TEST 4: Agent Tax Advisor (quick)")
    print("=" * 70)
    
    scenario = "Investissement: 40% actions PEA, 30% obligations, 20% SCPI"
    
    print(f"\nScénario: {scenario}")
    print("Exécution...")
    
    try:
        result = await tax_advisor.run(
            f"Quelles sont les implications fiscales de: {scenario}? Réponds brièvement."
        )
        
        has_structured = hasattr(result, 'data') and result.data
        print(f"\n   Structured output: {'✅ Oui' if has_structured else '⚠️  Non'}")
        
        if has_structured:
            tax_analysis = result.data
            print(f"   Régime: {tax_analysis.regime_fiscal}")
            print(f"   Avantages: {len(tax_analysis.avantages)}")
        
        return True
        
    except Exception as e:
        print(f"\n   ❌ Erreur: {type(e).__name__}: {str(e)[:100]}")
        return False


async def test_portfolio_optimizer_quick():
    """Test rapide de l'agent optimiseur."""
    print("\n" + "=" * 70)
    print("🧪 TEST 5: Agent Portfolio Optimizer (quick)")
    print("=" * 70)
    
    scenario = "100k€, 30 ans, profil modéré"
    
    print(f"\nScénario: {scenario}")
    print("Exécution...")
    
    try:
        result = await portfolio_optimizer.run(
            f"Propose une allocation pour: {scenario}. Utilise les outils pour calculer."
        )
        
        tool_calls = extract_tool_calls(result)
        print(f"\n   Tool calls: {len(tool_calls)}")
        if tool_calls:
            print(f"   ✅ Outils utilisés: {tool_calls[0][:50]}...")
        else:
            print(f"   ⚠️  Aucun outil (réponse: {result.output[:80]}...)")
        
        return True
        
    except Exception as e:
        print(f"\n   ❌ Erreur: {type(e).__name__}: {str(e)[:100]}")
        return False


# ============================================================================
# TEST DE COMPLIANCE (sans API pour la partie extraction)
# ============================================================================

def test_tool_extraction():
    """Test de l'extraction de tool calls (sans API)."""
    print("\n" + "=" * 70)
    print("🧪 TEST 6: Extraction Tool Calls")
    print("=" * 70)
    
    # Simuler un résultat avec tool calls
    class MockMessage:
        def __init__(self):
            class MockCall:
                def __init__(self):
                    class MockFunc:
                        name = "calculer_rendement_portfolio"
                        arguments = '{"allocation_actions": 0.4}'
                    self.function = MockFunc()
            self.tool_calls = [MockCall()]
    
    class MockResult:
        def all_messages(self):
            return [MockMessage()]
    
    mock_result = MockResult()
    tool_calls = extract_tool_calls(mock_result)
    
    assert len(tool_calls) > 0, "Devrait extraire les tool calls"
    assert "calculer_rendement_portfolio" in tool_calls[0], "Devrait trouver le nom de l'outil"
    print(f"   ✅ Extraction réussie: {tool_calls[0][:60]}...")
    
    return True


# ============================================================================
# SUITE DE TESTS
# ============================================================================

async def run_all_tests(skip_api_tests=False):
    """Exécute tous les tests."""
    print("\n" + "=" * 70)
    print("🚀 SUITE DE TESTS - Agent 3 Multi-Step Workflow")
    print("=" * 70)
    if skip_api_tests:
        print("⚠️  Mode: Tests unitaires uniquement (API tests skippés)")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Outils (unitaire)
    try:
        test_tools()
        results["tools"] = True
    except Exception as e:
        print(f"❌ Test outils échoué: {e}")
        results["tools"] = False
    
    # Test 2: Modèles structurés (unitaire)
    try:
        test_structured_models()
        results["models"] = True
    except Exception as e:
        print(f"❌ Test modèles échoué: {e}")
        results["models"] = False
    
    # Test 3: Extraction tool calls (unitaire)
    try:
        results["tool_extraction"] = test_tool_extraction()
    except Exception as e:
        print(f"❌ Test extraction échoué: {e}")
        results["tool_extraction"] = False
    
    # Tests API (optionnels)
    if not skip_api_tests:
        results["risk_analyst"] = await test_risk_analyst_quick()
        results["tax_advisor"] = await test_tax_advisor_quick()
        results["portfolio_optimizer"] = await test_portfolio_optimizer_quick()
    else:
        print("\n⏭️  Tests API skippés (utilisez --skip-api pour forcer)")
        results["risk_analyst"] = None
        results["tax_advisor"] = None
        results["portfolio_optimizer"] = None
    
    # Résumé
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 70)
    
    for test_name, passed in results.items():
        if passed is None:
            status = "⏭️  SKIP"
        elif passed:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"  {test_name:25} {status}")
    
    # Compter seulement les tests exécutés
    executed = {k: v for k, v in results.items() if v is not None}
    total = len(executed)
    passed = sum(1 for v in executed.values() if v)
    
    print(f"\n  Total: {passed}/{total} tests réussis")
    
    if passed == total and total > 0:
        print("\n🎉 Tous les tests sont passés!")
    elif total > 0:
        print(f"\n⚠️  {total - passed} test(s) ont échoué")
    
    return all(v for v in executed.values() if v is not None)


if __name__ == "__main__":
    import sys
    
    skip_api = "--skip-api" in sys.argv or "-s" in sys.argv
    
    if skip_api:
        print("⚠️  Mode: Tests unitaires uniquement")
        # Tests unitaires seulement
        test_tools()
        test_structured_models()
        test_tool_extraction()
        print("\n✅ Tests unitaires terminés")
    else:
        # Tests complets
        success = asyncio.run(run_all_tests(skip_api_tests=False))
        exit(0 if success else 1)
