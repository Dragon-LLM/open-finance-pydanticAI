# Améliorations de l'extraction SWIFT

## Résumé des améliorations

L'extraction de messages SWIFT a été complètement révisée et améliorée avec:

### 1. Parser robuste avec validation Pydantic

**Fichier:** `swift_extractor.py`

- Nouveau module dédié à l'extraction SWIFT avec validation stricte
- Utilisation de modèles Pydantic pour garantir la cohérence des données
- Validation automatique des formats (dates, devises, montants, BIC)

### 2. Support complet des champs SWIFT MT103

**Champs gérés:**
- `:20:` - Référence du transfert
- `:23B:` - Code instruction (CRED, etc.)
- `:32A:` - Date de valeur, devise, montant (avec parsing intelligent)
- `:50K:`, `:50A:`, `:50F:` - Ordre donneur (multi-lignes)
- `:52A:`, `:52D:` - Banque ordonnateur
- `:56A:`, `:56D:` - Banque intermédiaire
- `:57A:`, `:57D:` - Banque bénéficiaire
- `:59:`, `:59A:` - Bénéficiaire (multi-lignes)
- `:70:` - Information pour bénéficiaire (multi-lignes)
- `:71A:` - Frais (OUR/SHA/BEN)
- `:72:` - Information banque à banque (multi-lignes)

### 3. Gestion des champs multi-lignes

Le parser gère correctement les champs qui s'étendent sur plusieurs lignes:
- Lire toutes les lignes jusqu'au prochain tag SWIFT
- Préserver les sauts de ligne dans les adresses et noms
- Extraire les informations structurées (IBAN, BIC) depuis le texte libre

### 4. Extraction automatique

**IBAN:**
- Détection automatique des IBAN dans les champs `:50K:` et `:59:`
- Validation de la longueur (15-34 caractères)
- Nettoyage automatique (suppression des espaces)

**BIC:**
- Extraction depuis les champs `:52A:`, `:56A:`, `:57A:`
- Validation du format (8 ou 11 caractères)
- Pattern matching robuste

### 5. Support des formats de date

**Format :32A:**
- Support YYMMDD (6 chiffres) → conversion automatique en YYYYMMDD
- Support YYYYMMDD (8 chiffres)
- Logique intelligente pour les années (YY < 50 → 20YY, sinon 19YY)

### 6. Validation stricte

**Validations implémentées:**
- Dates: format YYYYMMDD avec vérification des valeurs
- Devises: codes ISO 3 lettres majuscules
- Montants: nombres positifs avec gestion des virgules/points
- BIC: longueur 8 ou 11 caractères
- Charges: valeurs strictes (OUR, SHA, BEN)

### 7. Structure de données typée

**Modèle Pydantic:** `SwiftMT103Parsed`

```python
class SwiftMT103Parsed(BaseModel):
    field_20: str  # Référence
    field_32A: SwiftField32A  # Date, devise, montant (validé)
    field_50K: str  # Ordre donneur
    field_59: str  # Bénéficiaire
    # ... tous les champs optionnels
    ordering_customer_account: Optional[str]  # IBAN extrait
    beneficiary_account: Optional[str]  # IBAN extrait
```

### 8. Fonctionnalités supplémentaires

**Formatage inverse:**
- `format_swift_mt103_from_parsed()` - Reconstitution du message SWIFT depuis une structure parsée

**Gestion d'erreurs:**
- Messages d'erreur détaillés pour faciliter le débogage
- Fallback vers extraction LLM si le parsing échoue

## Utilisation

### Parser basique (ancienne fonction)

```python
from examples.agent_swift import parse_swift_mt103

swift_text = """
:20:NONREF
:23B:CRED
:32A:241215EUR15000.00
:50K:/FR76300040000100000000000123
ORDRE DUPONT JEAN
:59:/FR1420041010050500013M02606
BENEFICIAIRE MARTIN
:71A:OUR
"""

parsed = parse_swift_mt103(swift_text)
```

### Parser avancé (recommandé)

```python
from examples.swift_extractor import parse_swift_mt103_advanced

parsed = parse_swift_mt103_advanced(swift_text)

# Accès aux données validées
print(parsed.field_32A.amount)  # 15000.0
print(parsed.field_32A.currency)  # EUR
print(parsed.field_32A.value_date)  # 20241215
print(parsed.ordering_customer_account)  # FR76300040000100000000000123
```

### Avec agent PydanticAI

```python
from examples.agent_swift import swift_parser

result = await swift_parser.run(f"Parse ce message SWIFT:\n{swift_text}")
# L'agent utilise le parser avancé en arrière-plan
```

## Améliorations futures possibles

1. **Support MT940** (relevés bancaires)
2. **Support MT202** (transferts interbancaires)
3. **Validation IBAN** (algorithme de contrôle)
4. **Cache de parsing** pour performance
5. **Mode strict vs permissif** pour différents niveaux de validation

## Tests

Tous les parsers sont testés avec:
- Messages SWIFT standards
- Formats YYMMDD et YYYYMMDD
- Champs multi-lignes complexes
- Champs optionnels
- Cas limites (montants avec virgules, IBAN avec espaces, etc.)









