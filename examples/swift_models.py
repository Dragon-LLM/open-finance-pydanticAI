"""
Modèles Pydantic pour messages SWIFT.

Ces modèles peuvent être utilisés avec output_type pour valider
automatiquement la structure des messages SWIFT générés.
"""

from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class SWIFTFielBase(BaseModel):
    """Classe de base pour les champs SWIFT."""
    pass


class MT103Field32A(BaseModel):
    """Champ :32A: Date de valeur, devise, montant."""
    value_date: str = Field(description="Date de valeur YYYYMMDD")
    currency: str = Field(description="Code devise ISO 3 lettres")
    amount: float = Field(description="Montant", gt=0)
    
    @field_validator("value_date")
    def validate_date(cls, v):
        if len(v) != 8 or not v.isdigit():
            raise ValueError("Date must be YYYYMMDD format")
        try:
            datetime.strptime(v, "%Y%m%d")
        except ValueError:
            raise ValueError("Invalid date")
        return v
    
    @field_validator("currency")
    def validate_currency(cls, v):
        if len(v) != 3 or not v.isalpha():
            raise ValueError("Currency must be 3 letter ISO code")
        return v.upper()


class SWIFTMT103Structured(BaseModel):
    """Message SWIFT MT103 avec validation complète."""
    
    field_20: str = Field(description=":20: Référence du transfert")
    field_23B: str = Field(default="CRED", description=":23B: Code instruction")
    field_32A: MT103Field32A = Field(description=":32A: Date, devise, montant")
    field_50K: str = Field(description=":50K: Ordre donneur")
    field_59: str = Field(description=":59: Bénéficiaire")
    field_70: str | None = Field(default=None, description=":70: Information pour bénéficiaire")
    field_71A: str = Field(default="OUR", description=":71A: Frais (OUR/SHA/BEN)")
    
    @field_validator("field_71A")
    def validate_charges(cls, v):
        valid = ["OUR", "SHA", "BEN"]
        if v not in valid:
            raise ValueError(f"Charges must be one of {valid}")
        return v
    
    def to_swift_format(self) -> str:
        """Convertit en format SWIFT standard."""
        lines = [
            f":20:{self.field_20}",
            f":23B:{self.field_23B}",
            f":32A:{self.field_32A.value_date}{self.field_32A.currency}{self.field_32A.amount:.2f}",
            f":50K:/{self.field_50K}",
            f":59:/{self.field_59}",
        ]
        
        if self.field_70:
            lines.append(f":70:{self.field_70}")
        
        lines.append(f":71A:{self.field_71A}")
        
        return "\n".join(lines)


# Exemple d'utilisation avec validation
def example_with_validation():
    """Exemple d'utilisation avec validation Pydantic."""
    try:
        mt103 = SWIFTMT103Structured(
            field_20="TXN-2025-001",
            field_32A=MT103Field32A(
                value_date="20250120",
                currency="EUR",
                amount=15000.00
            ),
            field_50K="FR76300040000100000000000123\nORDRE DUPONT",
            field_59="FR1420041010050500013M02606\nBENEFICIAIRE MARTIN",
            field_70="Paiement facture",
            field_71A="OUR"
        )
        
        print("✅ Message SWIFT validé:")
        print(mt103.to_swift_format())
        
    except Exception as e:
        print(f"❌ Erreur de validation: {e}")









