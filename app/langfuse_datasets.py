"""Langfuse evaluation datasets for agent testing.

This module creates and manages evaluation datasets in Langfuse with varying difficulty levels.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy_financial as npf

from app.langfuse_config import get_langfuse_client

logger = logging.getLogger(__name__)


# Dataset definitions with difficulty levels
AGENT_1_DATASET = [
    # Easy: Simple portfolio with 2-3 positions, clear format
    {
        "prompt": "Extrais le portfolio: 50 AIR.PA à 120€, 30 SAN.PA à 85€",
        "expected_output": {"total_value": 8550.0, "positions_count": 2},
        "difficulty": "easy",
        "category": "portfolio_extraction",
        "metadata": {"expected_positions": ["AIR.PA", "SAN.PA"]},
    },
    {
        "prompt": "Mon portfolio: 100 TTE.PA à 55€, 25 LVMH.PA à 800€",
        "expected_output": {"total_value": 100*55 + 25*800, "positions_count": 2},
        "difficulty": "easy",
        "category": "portfolio_extraction",
    },
    {
        "prompt": "Extrais: 40 OR.PA à 65€, 60 MC.PA à 600€, 20 KER.PA à 50€",
        "expected_output": {"total_value": 40*65 + 60*600 + 20*50, "positions_count": 3},
        "difficulty": "easy",
        "category": "portfolio_extraction",
    },
    {
        "prompt": "Portfolio: 75 SAF.PA à 90€, 50 VIV.PA à 25€",
        "expected_output": {"total_value": 75*90 + 50*25, "positions_count": 2},
        "difficulty": "easy",
        "category": "portfolio_extraction",
    },
    # Medium: Portfolio with 4-6 positions, mixed formats, dates
    {
        "prompt": "J'ai acheté 50 actions Airbus (AIR.PA) à 120€ le 15 mars 2024, 30 actions Sanofi (SAN.PA) à 85€ le 20 février 2024, 100 actions TotalEnergies (TTE.PA) à 55€ le 10 janvier 2024",
        "expected_output": {"total_value": 50*120 + 30*85 + 100*55, "positions_count": 3},
        "difficulty": "medium",
        "category": "portfolio_extraction",
        "metadata": {"has_dates": True, "has_full_names": True},
    },
    {
        "prompt": "Mon portfolio actuel: 40 actions LVMH (LVMH.PA) à 800€ achetées le 1er janvier 2024, 60 actions Hermès (RMS.PA) à 2000€ le 15 février, 25 actions Kering (KER.PA) à 400€ le 1er mars, 80 actions L'Oréal (OR.PA) à 350€ le 10 mars",
        "expected_output": {"total_value": 40*800 + 60*2000 + 25*400 + 80*350, "positions_count": 4},
        "difficulty": "medium",
        "category": "portfolio_extraction",
    },
    {
        "prompt": "Portfolio boursier: 35 actions Danone (BN.PA) à 55€ (achat: 2024-01-05), 45 actions Carrefour (CA.PA) à 18€ (2024-02-10), 55 actions Engie (ENGI.PA) à 13€ (2024-03-01), 20 actions Veolia (VIE.PA) à 28€ (2024-03-15)",
        "expected_output": {"total_value": 35*55 + 45*18 + 55*13 + 20*28, "positions_count": 4},
        "difficulty": "medium",
        "category": "portfolio_extraction",
    },
    {
        "prompt": "J'ai investi dans: 50 AIR.PA (120€, mars 2024), 30 SAN.PA (85€, février), 100 TTE.PA (55€, janvier), 25 LVMH.PA (800€, décembre 2023), 40 OR.PA (65€, novembre)",
        "expected_output": {"total_value": 50*120 + 30*85 + 100*55 + 25*800 + 40*65, "positions_count": 5},
        "difficulty": "medium",
        "category": "portfolio_extraction",
    },
    # Hard: Complex portfolio with 7+ positions, ambiguous formats
    {
        "prompt": "Mon portfolio diversifié comprend plusieurs positions: environ 50 actions d'Airbus achetées à 120 euros l'unité en début d'année, une trentaine de Sanofi à 85€, cent TotalEnergies à 55€, vingt-cinq LVMH à environ 800€, quarante L'Oréal à 65€, soixante-dix Hermès à 2000€, et trente Kering à 400€. Certaines ont été achetées en 2023, d'autres cette année.",
        "expected_output": {"total_value": 50*120 + 30*85 + 100*55 + 25*800 + 40*65 + 70*2000 + 30*400, "positions_count": 7},
        "difficulty": "hard",
        "category": "portfolio_extraction",
        "metadata": {"ambiguous_formats": True, "natural_language": True},
    },
    {
        "prompt": "Portfolio complexe: 45 AIR.PA (120€, 15/03/2024), 35 SAN.PA (85€, 20/02/2024), 110 TTE.PA (55€, 10/01/2024), 28 LVMH.PA (800€, 01/12/2023), 42 OR.PA (65€, 15/11/2023), 65 RMS.PA (2000€, 01/10/2023), 32 KER.PA (400€, 20/09/2023), 50 MC.PA (600€, 05/08/2023)",
        "expected_output": {"total_value": 45*120 + 35*85 + 110*55 + 28*800 + 42*65 + 65*2000 + 32*400 + 50*600, "positions_count": 8},
        "difficulty": "hard",
        "category": "portfolio_extraction",
    },
    {
        "prompt": "J'ai un portfolio avec de nombreuses positions: 50 AIR.PA à 120€, 30 SAN.PA à 85€, 100 TTE.PA à 55€, 25 LVMH.PA à 800€, 40 OR.PA à 65€, 70 RMS.PA à 2000€, 30 KER.PA à 400€, 50 MC.PA à 600€, 20 BN.PA à 55€, 45 CA.PA à 18€",
        "expected_output": {"total_value": 50*120 + 30*85 + 100*55 + 25*800 + 40*65 + 70*2000 + 30*400 + 50*600 + 20*55 + 45*18, "positions_count": 10},
        "difficulty": "hard",
        "category": "portfolio_extraction",
    },
    {
        "prompt": "Portfolio mixte avec positions variées: environ 50 Airbus, 30 Sanofi, 100 TotalEnergies, 25 LVMH, 40 L'Oréal, 70 Hermès, 30 Kering, 50 Michelin, 20 Danone, 45 Carrefour, 55 Engie, 20 Veolia. Les prix varient entre 13€ et 2000€ par action.",
        "expected_output": {"positions_count": 12},
        "difficulty": "hard",
        "category": "portfolio_extraction",
        "metadata": {"missing_prices": True, "natural_language": True},
    },
]

AGENT_2_DATASET = [
    # Easy: Simple future value calculation
    {
        "prompt": "50000€ à 4% sur 5 ans?",
        "expected_output": abs(npf.fv(rate=0.04, nper=5, pmt=0, pv=-50000)),
        "difficulty": "easy",
        "category": "future_value",
        "metadata": {"expected_tool": "calculer_valeur_future"},
    },
    {
        "prompt": "J'ai 100000€ à placer à 3% par an pendant 10 ans. Quelle sera la valeur finale?",
        "expected_output": abs(npf.fv(rate=0.03, nper=10, pmt=0, pv=-100000)),
        "difficulty": "easy",
        "category": "future_value",
    },
    {
        "prompt": "25000€ investis à 5% sur 7 ans, combien aurai-je?",
        "expected_output": abs(npf.fv(rate=0.05, nper=7, pmt=0, pv=-25000)),
        "difficulty": "easy",
        "category": "future_value",
    },
    {
        "prompt": "Valeur future de 75000€ à 2.5% sur 15 ans",
        "expected_output": abs(npf.fv(rate=0.025, nper=15, pmt=0, pv=-75000)),
        "difficulty": "easy",
        "category": "future_value",
    },
    # Medium: Loan payment calculation
    {
        "prompt": "Emprunt de 200000€ à 3.5% sur 20 ans. Mensualité?",
        "expected_output": abs(npf.pmt(rate=0.035/12, nper=20*12, pv=200000)),
        "difficulty": "medium",
        "category": "loan_payment",
        "metadata": {"expected_tool": "calculer_versement_mensuel"},
    },
    {
        "prompt": "Quelle est la mensualité pour un prêt de 300000€ à 4% sur 25 ans?",
        "expected_output": abs(npf.pmt(rate=0.04/12, nper=25*12, pv=300000)),
        "difficulty": "medium",
        "category": "loan_payment",
    },
    {
        "prompt": "Je veux emprunter 150000€ à 2.8% sur 15 ans. Combien paierai-je par mois?",
        "expected_output": abs(npf.pmt(rate=0.028/12, nper=15*12, pv=150000)),
        "difficulty": "medium",
        "category": "loan_payment",
    },
    {
        "prompt": "Prêt immobilier: 400000€ à 3.2% sur 30 ans. Calculer la mensualité.",
        "expected_output": abs(npf.pmt(rate=0.032/12, nper=30*12, pv=400000)),
        "difficulty": "medium",
        "category": "loan_payment",
    },
    # Hard: Complex calculations, present value, rate finding
    {
        "prompt": "Quel taux d'intérêt pour avoir 50000€ dans 8 ans avec 25000€ aujourd'hui?",
        "expected_output": npf.rate(nper=8, pv=-25000, fv=50000, pmt=0),
        "difficulty": "hard",
        "category": "rate_finding",
        "metadata": {"expected_tool": "calculer_taux_necessaire"},
    },
    {
        "prompt": "Quelle est la valeur actuelle de 100000€ dans 15 ans avec un taux d'actualisation de 3%?",
        "expected_output": abs(npf.pv(rate=0.03, nper=15, fv=100000, pmt=0)),
        "difficulty": "hard",
        "category": "present_value",
    },
    {
        "prompt": "Combien dois-je investir aujourd'hui à 4% pour avoir 200000€ dans 20 ans?",
        "expected_output": abs(npf.pv(rate=0.04, nper=20, fv=200000, pmt=0)),
        "difficulty": "hard",
        "category": "present_value",
    },
    {
        "prompt": "J'ai 10000€ aujourd'hui et je veux 30000€ dans 12 ans. Quel taux me faut-il?",
        "expected_output": npf.rate(nper=12, pv=-10000, fv=30000, pmt=0),
        "difficulty": "hard",
        "category": "rate_finding",
    },
    # Expert: Multi-step calculations, portfolio performance
    {
        "prompt": "Performance d'un portfolio: 40% actions (rendement 7%), 30% obligations (3%), 20% immobilier (5%), 10% autres (10%). Investissement initial 100000€ sur 10 ans.",
        "expected_output": {"expected_return": 0.40*0.07 + 0.30*0.03 + 0.20*0.05 + 0.10*0.10},
        "difficulty": "expert",
        "category": "portfolio_performance",
        "metadata": {"multi_step": True, "expected_tool": "calculer_rendement_portfolio"},
    },
    {
        "prompt": "Calculer le rendement attendu: 50% actions (8%), 25% obligations (2.5%), 15% immobilier (4%), 10% liquidités (1%)",
        "expected_output": {"expected_return": 0.50*0.08 + 0.25*0.025 + 0.15*0.04 + 0.10*0.01},
        "difficulty": "expert",
        "category": "portfolio_performance",
    },
    {
        "prompt": "Portfolio 60% actions (rendement 6%), 20% obligations (2%), 10% immobilier (3.5%), 10% crypto (15%). Rendement total?",
        "expected_output": {"expected_return": 0.60*0.06 + 0.20*0.02 + 0.10*0.035 + 0.10*0.15},
        "difficulty": "expert",
        "category": "portfolio_performance",
    },
    {
        "prompt": "Allocation: 35% actions (7%), 35% obligations (3%), 20% immobilier (5%), 10% matières premières (12%). Calculer rendement.",
        "expected_output": {"expected_return": 0.35*0.07 + 0.35*0.03 + 0.20*0.05 + 0.10*0.12},
        "difficulty": "expert",
        "category": "portfolio_performance",
    },
]

AGENT_3_DATASET = [
    # Easy: Simple risk analysis
    {
        "prompt": "40% actions, 30% obligations, 20% immobilier, 10% autres. Investissement 100k€, horizon 30 ans.",
        "expected_output": {"has_risk_analysis": True, "has_tax_analysis": True},
        "difficulty": "easy",
        "category": "risk_tax_workflow",
    },
    {
        "prompt": "Analyse le risque: 50% actions, 30% obligations, 20% autres. 50000€, 20 ans.",
        "expected_output": {"has_risk_analysis": True},
        "difficulty": "easy",
        "category": "risk_analysis",
    },
    {
        "prompt": "Portfolio: 60% actions, 40% obligations. Investissement 75000€, horizon 25 ans.",
        "expected_output": {"has_risk_analysis": True},
        "difficulty": "easy",
        "category": "risk_analysis",
    },
    {
        "prompt": "30% actions, 50% obligations, 20% immobilier. 200000€, 15 ans.",
        "expected_output": {"has_risk_analysis": True},
        "difficulty": "easy",
        "category": "risk_analysis",
    },
    # Medium: Complex allocation with tax optimization
    {
        "prompt": "Portfolio: 45% actions, 25% obligations, 20% immobilier, 10% autres. 150000€, 30 ans. Optimisation fiscale PEA vs compte-titres.",
        "expected_output": {"has_risk_analysis": True, "has_tax_analysis": True, "has_optimization": True},
        "difficulty": "medium",
        "category": "tax_optimization",
        "metadata": {"tax_regime_comparison": True},
    },
    {
        "prompt": "Stratégie: 50% actions (PEA), 30% obligations (assurance-vie), 20% immobilier (SCPI). 300000€, 20 ans. Analyse fiscale.",
        "expected_output": {"has_tax_analysis": True, "has_regime_details": True},
        "difficulty": "medium",
        "category": "tax_optimization",
    },
    {
        "prompt": "Allocation: 40% actions PEA, 30% obligations assurance-vie, 20% immobilier, 10% autres. 250000€, 25 ans. Recommandations fiscales.",
        "expected_output": {"has_tax_analysis": True},
        "difficulty": "medium",
        "category": "tax_optimization",
    },
    {
        "prompt": "Portfolio mixte: 35% actions (PEA), 35% obligations (assurance-vie), 20% immobilier, 10% crypto. 180000€, 30 ans. Optimisation.",
        "expected_output": {"has_risk_analysis": True, "has_tax_analysis": True},
        "difficulty": "medium",
        "category": "tax_optimization",
    },
    # Hard: Full workflow with constraints
    {
        "prompt": "Portfolio complet: 40% actions (PEA max 150k€), 30% obligations (assurance-vie), 20% immobilier (SCPI), 10% autres. Investissement 500000€, horizon 30 ans, profil modéré. Analyse risque, fiscalité, et optimisation.",
        "expected_output": {"has_risk_analysis": True, "has_tax_analysis": True, "has_optimization": True, "has_constraints": True},
        "difficulty": "hard",
        "category": "full_workflow",
        "metadata": {"multi_step": True, "constraints": True},
    },
    {
        "prompt": "Stratégie d'investissement: 45% actions (PEA + compte-titres), 25% obligations (assurance-vie), 20% immobilier, 10% matières premières. Budget 400000€, horizon 25 ans, objectif retraite. Analyse complète.",
        "expected_output": {"has_risk_analysis": True, "has_tax_analysis": True, "has_optimization": True},
        "difficulty": "hard",
        "category": "full_workflow",
    },
    {
        "prompt": "Portfolio diversifié: 50% actions (30% PEA, 20% compte-titres), 30% obligations (assurance-vie), 15% immobilier, 5% autres. 600000€, 30 ans, profil équilibré. Recommandations complètes.",
        "expected_output": {"has_risk_analysis": True, "has_tax_analysis": True, "has_optimization": True},
        "difficulty": "hard",
        "category": "full_workflow",
    },
    {
        "prompt": "Allocation complexe: 40% actions (PEA 150k€ max, reste compte-titres), 30% obligations (assurance-vie), 20% immobilier (SCPI), 10% autres. 800000€, 35 ans, profil dynamique. Analyse détaillée.",
        "expected_output": {"has_risk_analysis": True, "has_tax_analysis": True, "has_optimization": True},
        "difficulty": "hard",
        "category": "full_workflow",
    },
]

AGENT_4_DATASET = [
    # Easy: Standard European call
    {
        "prompt": "Call européen: Spot 100, Strike 105, Maturité 0.5 an, Taux 0.02, Volatilité 0.25",
        "expected_output": {"has_price": True, "price_range": (2.0, 8.0)},
        "difficulty": "easy",
        "category": "option_pricing",
        "metadata": {"expected_tool": "calculer_prix_call_black_scholes"},
    },
    {
        "prompt": "Prix d'un call: Spot 50, Strike 55, Maturité 1 an, Taux 0.03, Volatilité 0.20",
        "expected_output": {"has_price": True},
        "difficulty": "easy",
        "category": "option_pricing",
    },
    {
        "prompt": "Call européen: Spot 200, Strike 210, Maturité 0.25 an, Taux 0.025, Volatilité 0.30",
        "expected_output": {"has_price": True},
        "difficulty": "easy",
        "category": "option_pricing",
    },
    {
        "prompt": "Calculer prix call: Spot 75, Strike 80, Maturité 0.75 an, Taux 0.02, Volatilité 0.22",
        "expected_output": {"has_price": True},
        "difficulty": "easy",
        "category": "option_pricing",
    },
    # Medium: Options with dividends
    {
        "prompt": "Call avec dividende 0.01, Spot 100, Strike 110, Maturité 1 an, Taux 0.02, Volatilité 0.25",
        "expected_output": {"has_price": True, "has_dividend": True},
        "difficulty": "medium",
        "category": "option_pricing",
    },
    {
        "prompt": "Call européen avec dividende: Spot 150, Strike 160, Maturité 0.5 an, Taux 0.03, Volatilité 0.28, Dividende 0.015",
        "expected_output": {"has_price": True},
        "difficulty": "medium",
        "category": "option_pricing",
    },
    {
        "prompt": "Prix call avec dividende 0.02: Spot 80, Strike 85, Maturité 1 an, Taux 0.025, Volatilité 0.24",
        "expected_output": {"has_price": True},
        "difficulty": "medium",
        "category": "option_pricing",
    },
    {
        "prompt": "Call: Spot 120, Strike 125, Maturité 0.6 an, Taux 0.02, Volatilité 0.26, Dividende 0.012",
        "expected_output": {"has_price": True},
        "difficulty": "medium",
        "category": "option_pricing",
    },
    # Hard: Complex options, Greeks
    {
        "prompt": "Calcule Delta, Gamma, Vega, Theta pour un call européen: Spot 100, Strike 105, Maturité 0.5 an, Taux 0.02, Volatilité 0.25",
        "expected_output": {"has_price": True, "has_greeks": True, "greeks": ["delta", "gamma", "vega", "theta"]},
        "difficulty": "hard",
        "category": "option_greeks",
        "metadata": {"requires_greeks": True},
    },
    {
        "prompt": "Call européen avec tous les Greeks: Spot 200, Strike 210, Maturité 1 an, Taux 0.03, Volatilité 0.30, Dividende 0.02",
        "expected_output": {"has_price": True, "has_greeks": True},
        "difficulty": "hard",
        "category": "option_greeks",
    },
    {
        "prompt": "Calculer prix et Greeks (Delta, Gamma, Vega, Theta) pour call: Spot 50, Strike 55, Maturité 0.75 an, Taux 0.025, Volatilité 0.22",
        "expected_output": {"has_price": True, "has_greeks": True},
        "difficulty": "hard",
        "category": "option_greeks",
    },
    {
        "prompt": "Call avec analyse complète (prix + Greeks): Spot 300, Strike 310, Maturité 0.5 an, Taux 0.02, Volatilité 0.28, Dividende 0.01",
        "expected_output": {"has_price": True, "has_greeks": True},
        "difficulty": "hard",
        "category": "option_greeks",
    },
]

AGENT_5_DATASET = [
    # Easy: Simple SWIFT MT103
    {
        "prompt": "Convert this SWIFT MT103 to ISO 20022:\n{1:F01BANKFRPPAXXX1234567890}\n{2:O1031201234567BANKFRPPAXXX1234567890123456}\n{4:\n:20:REF123\n:32A:240112EUR100000,00\n:50K:/FR1420041010050500013M02606\nJohn Doe\n:59:/FR7630001007941234567890185\nJane Smith\n:70:Payment\n-}",
        "expected_output": {"has_iso20022": True, "has_conversion": True},
        "difficulty": "easy",
        "category": "swift_conversion",
    },
    {
        "prompt": "Convert SWIFT to ISO 20022:\n{1:F01BANKFRPPAXXX}\n{2:O1031201234567}\n{4:\n:20:REF456\n:32A:240201EUR50000,00\n:50K:John Doe\n:59:Jane Smith\n-}",
        "expected_output": {"has_iso20022": True},
        "difficulty": "easy",
        "category": "swift_conversion",
    },
    {
        "prompt": "SWIFT MT103 to ISO 20022:\n{1:F01BANKFRPPAXXX}\n{2:O1031201234567}\n{4:\n:20:REF789\n:32A:240301EUR25000,00\n:50K:Debtor\n:59:Creditor\n-}",
        "expected_output": {"has_iso20022": True},
        "difficulty": "easy",
        "category": "swift_conversion",
    },
    {
        "prompt": "Convert: {1:F01BANKFRPPAXXX}\n{2:O1031201234567}\n{4:\n:20:REF001\n:32A:240401EUR75000,00\n:50K:ABC Corp\n:59:XYZ Ltd\n-}",
        "expected_output": {"has_iso20022": True},
        "difficulty": "easy",
        "category": "swift_conversion",
    },
    # Medium: Complex message with multiple parties
    {
        "prompt": "Convert SWIFT MT103 to ISO 20022:\n{1:F01BANKFRPPAXXX1234567890}\n{2:O1031201234567BANKFRPPAXXX1234567890123456}\n{3:{113:SEPA}}\n{4:\n:20:REF123456789\n:23B:CRED\n:32A:240112EUR100000,00\n:50K:/FR1420041010050500013M02606\nJohn Doe\n123 Main Street\nParis\n:52A:BANKFRPP\n:53A:BANKDEFF\n:57A:BANKUS33\n:59:/FR7630001007941234567890185\nJane Smith\n456 Oak Avenue\nLyon\n:70:Payment for services\n:71A:SHA\n-}",
        "expected_output": {"has_iso20022": True, "has_multiple_parties": True},
        "difficulty": "medium",
        "category": "swift_conversion",
    },
    {
        "prompt": "SWIFT to ISO 20022 conversion:\n{1:F01BANKFRPPAXXX}\n{2:O1031201234567}\n{3:{113:SEPA}}\n{4:\n:20:REF999\n:23B:CRED\n:32A:240215EUR200000,00\n:50K:/FR1420041010050500013M02606\nCompany A\nAddress A\n:52A:INTERMEDIARY\n:53A:ACCOUNT_SERVICING\n:57A:RECEIVER\n:59:/FR7630001007941234567890185\nCompany B\nAddress B\n:70:Invoice payment\n:71A:SHA\n:72:/INS/BANKFRPP\n-}",
        "expected_output": {"has_iso20022": True},
        "difficulty": "medium",
        "category": "swift_conversion",
    },
    {
        "prompt": "Convert complex SWIFT:\n{1:F01BANKFRPPAXXX}\n{2:O1031201234567}\n{4:\n:20:REF888\n:32A:240320EUR150000,00\n:50K:/FR1420041010050500013M02606\nDebtor Corp\n:52A:Intermediary Bank\n:53A:Account Servicing\n:57A:Beneficiary Bank\n:59:/FR7630001007941234567890185\nCreditor Corp\n:70:Structured remittance\n:71A:SHA\n:72:Additional info\n-}",
        "expected_output": {"has_iso20022": True},
        "difficulty": "medium",
        "category": "swift_conversion",
    },
    {
        "prompt": "SWIFT MT103 with structured remittance:\n{1:F01BANKFRPPAXXX}\n{2:O1031201234567}\n{4:\n:20:REF777\n:32A:240425EUR300000,00\n:50K:Debtor\n:59:Creditor\n:70:/RFB/REF123456\n/CNT/INV001\n/DM/2024-01-15\n:71A:SHA\n-}",
        "expected_output": {"has_iso20022": True, "has_structured_remittance": True},
        "difficulty": "medium",
        "category": "swift_conversion",
    },
    # Hard: Bidirectional conversion, validation, risk assessment
    {
        "prompt": "Convert ISO 20022 pacs.008 to SWIFT MT103:\n<Document><FIToFICstmrCdtTrf><GrpHdr><MsgId>MSG001</MsgId></GrpHdr><CdtTrfTxInf><PmtId><InstrId>INSTR001</InstrId></PmtId><Amt><InstdAmt Ccy=\"EUR\">100000.00</InstdAmt></Amt><Dbtr><Nm>John Doe</Nm></Dbtr><Cdtr><Nm>Jane Smith</Nm></Cdtr></CdtTrfTxInf></FIToFICstmrCdtTrf></Document>",
        "expected_output": {"has_swift": True, "has_conversion": True},
        "difficulty": "hard",
        "category": "iso20022_conversion",
        "metadata": {"bidirectional": True},
    },
    {
        "prompt": "Validate this SWIFT MT103 message:\n{1:F01BANKFRPPAXXX}\n{2:O1031201234567}\n{4:\n:20:REF123\n:32A:240112EUR100000,00\n:50K:John Doe\n:59:Jane Smith\n-}",
        "expected_output": {"has_validation": True, "validation_result": "valid"},
        "difficulty": "hard",
        "category": "message_validation",
    },
    {
        "prompt": "Assess risk of transaction: Amount 500000 EUR, Debtor: John Doe (BIC: BANKFRPPAXXX), Creditor: Jane Smith, Reference: REF123456789, Date: 2024-01-12",
        "expected_output": {"has_risk_assessment": True, "has_risk_score": True},
        "difficulty": "hard",
        "category": "risk_assessment",
        "metadata": {"expected_tool": "evaluer_risque_message"},
    },
    {
        "prompt": "Full workflow: Convert SWIFT to ISO 20022, validate structure, and assess AML/KYC risk for: {1:F01BANKFRPPAXXX}\n{2:O1031201234567}\n{4:\n:20:REF999\n:32A:240112EUR500000,00\n:50K:Debtor\n:59:Creditor\n-}",
        "expected_output": {"has_conversion": True, "has_validation": True, "has_risk_assessment": True},
        "difficulty": "hard",
        "category": "full_workflow",
        "metadata": {"multi_step": True},
    },
]


def create_evaluation_datasets() -> Dict[str, str]:
    """
    Create all evaluation datasets in Langfuse.
    
    Returns:
        Dictionary mapping dataset names to dataset IDs
    """
    langfuse = get_langfuse_client()
    if not langfuse:
        logger.warning("Langfuse not configured, skipping dataset creation")
        return {}
    
    datasets = {}
    
    try:
        # Agent 1: Portfolio Extraction
        dataset_1 = langfuse.create_dataset(name="agent_1_portfolio_extraction")
        for item in AGENT_1_DATASET:
            dataset_1.create_item(
                input=item["prompt"],
                expected_output=item["expected_output"],
                metadata={
                    "difficulty": item["difficulty"],
                    "category": item["category"],
                    **(item.get("metadata", {})),
                },
            )
        datasets["agent_1"] = dataset_1.id
        logger.info(f"Created dataset: agent_1_portfolio_extraction ({len(AGENT_1_DATASET)} items)")
        
        # Agent 2: Financial Calculator
        dataset_2 = langfuse.create_dataset(name="agent_2_financial_calculator")
        for item in AGENT_2_DATASET:
            dataset_2.create_item(
                input=item["prompt"],
                expected_output=item["expected_output"],
                metadata={
                    "difficulty": item["difficulty"],
                    "category": item["category"],
                    **(item.get("metadata", {})),
                },
            )
        datasets["agent_2"] = dataset_2.id
        logger.info(f"Created dataset: agent_2_financial_calculator ({len(AGENT_2_DATASET)} items)")
        
        # Agent 3: Multi-Step Workflow
        dataset_3 = langfuse.create_dataset(name="agent_3_multi_step_workflow")
        for item in AGENT_3_DATASET:
            dataset_3.create_item(
                input=item["prompt"],
                expected_output=item["expected_output"],
                metadata={
                    "difficulty": item["difficulty"],
                    "category": item["category"],
                    **(item.get("metadata", {})),
                },
            )
        datasets["agent_3"] = dataset_3.id
        logger.info(f"Created dataset: agent_3_multi_step_workflow ({len(AGENT_3_DATASET)} items)")
        
        # Agent 4: Option Pricing
        dataset_4 = langfuse.create_dataset(name="agent_4_option_pricing")
        for item in AGENT_4_DATASET:
            dataset_4.create_item(
                input=item["prompt"],
                expected_output=item["expected_output"],
                metadata={
                    "difficulty": item["difficulty"],
                    "category": item["category"],
                    **(item.get("metadata", {})),
                },
            )
        datasets["agent_4"] = dataset_4.id
        logger.info(f"Created dataset: agent_4_option_pricing ({len(AGENT_4_DATASET)} items)")
        
        # Agent 5: SWIFT/ISO 20022
        dataset_5 = langfuse.create_dataset(name="agent_5_swift_iso20022")
        for item in AGENT_5_DATASET:
            dataset_5.create_item(
                input=item["prompt"],
                expected_output=item["expected_output"],
                metadata={
                    "difficulty": item["difficulty"],
                    "category": item["category"],
                    **(item.get("metadata", {})),
                },
            )
        datasets["agent_5"] = dataset_5.id
        logger.info(f"Created dataset: agent_5_swift_iso20022 ({len(AGENT_5_DATASET)} items)")
        
        logger.info(f"Successfully created {len(datasets)} datasets in Langfuse")
        return datasets
        
    except Exception as e:
        logger.error(f"Error creating Langfuse datasets: {e}", exc_info=True)
        return {}


def get_dataset_items(agent_name: str, difficulty: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve dataset items from Langfuse.
    
    Args:
        agent_name: Agent name ("agent_1", "agent_2", etc.)
        difficulty: Optional difficulty filter ("easy", "medium", "hard", "expert")
        
    Returns:
        List of dataset items
    """
    langfuse = get_langfuse_client()
    if not langfuse:
        logger.warning("Langfuse not configured")
        return []
    
    try:
        dataset_name_map = {
            "agent_1": "portfolio_extraction",
            "agent_2": "financial_calculator",
            "agent_3": "multi_step_workflow",
            "agent_4": "option_pricing",
            "agent_5": "swift_iso20022",
        }
        dataset_name = f"{agent_name}_{dataset_name_map.get(agent_name, 'unknown')}"
        
        # Get dataset (this is a simplified approach - actual Langfuse API may differ)
        # For now, return the hardcoded dataset items
        dataset_map = {
            "agent_1": AGENT_1_DATASET,
            "agent_2": AGENT_2_DATASET,
            "agent_3": AGENT_3_DATASET,
            "agent_4": AGENT_4_DATASET,
            "agent_5": AGENT_5_DATASET,
        }
        
        items = dataset_map.get(agent_name, [])
        if difficulty:
            items = [item for item in items if item.get("difficulty") == difficulty]
        
        return items
        
    except Exception as e:
        logger.error(f"Error retrieving dataset items: {e}", exc_info=True)
        return []


def add_dataset_item(
    agent_name: str,
    prompt: str,
    expected_output: Any,
    difficulty: str,
    metadata: Dict[str, Any],
) -> bool:
    """
    Add a new test case to a dataset.
    
    Args:
        agent_name: Agent name
        prompt: Input prompt
        expected_output: Expected output
        difficulty: Difficulty level
        metadata: Additional metadata
        
    Returns:
        True if successful
    """
    langfuse = get_langfuse_client()
    if not langfuse:
        logger.warning("Langfuse not configured")
        return False
    
    try:
        dataset_name_map = {
            "agent_1": "portfolio_extraction",
            "agent_2": "financial_calculator",
            "agent_3": "multi_step_workflow",
            "agent_4": "option_pricing",
            "agent_5": "swift_iso20022",
        }
        dataset_name = f"{agent_name}_{dataset_name_map.get(agent_name, 'unknown')}"
        
        # This would use Langfuse API to add item
        # For now, just log
        logger.info(f"Would add item to dataset {dataset_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error adding dataset item: {e}", exc_info=True)
        return False

