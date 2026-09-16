from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field

EVIDENCE_DESCRIPTION = (
    "Shortest verbatim quote from the document (max 20 words) that supports value; null if value is null."
)


class ContractType(StrEnum):
    """Contract categories from CUAD, normalised, plus OTHER."""
    AFFILIATE = "Affiliate"
    AGENCY = "Agency"
    CO_BRANDING = "Co-Branding"
    COLLABORATION = "Collaboration"
    DEVELOPMENT = "Development"
    DISTRIBUTOR = "Distributor"
    ENDORSEMENT = "Endorsement"
    FRANCHISE = "Franchise"
    HOSTING = "Hosting"
    IP = "IP"
    JOINT_VENTURE = "Joint Venture"
    LICENSE = "License"
    MAINTENANCE = "Maintenance"
    MANUFACTURING = "Manufacturing"
    MARKETING = "Marketing"
    NON_COMPETE = "Non-Compete"
    OUTSOURCING = "Outsourcing"
    PROMOTION = "Promotion"
    RESELLER = "Reseller"
    SERVICE = "Service"
    SPONSORSHIP = "Sponsorship"
    STRATEGIC_ALLIANCE = "Strategic Alliance"
    SUPPLY = "Supply"
    TRANSPORTATION = "Transportation"
    OTHER = "Other"


class Party(BaseModel):
    """One contracting party."""
    name: str = Field(description="Legal name exactly as written.")
    role: str | None = Field(description="Defined term the contract uses for this party, e.g. 'Licensor'.")


class DateField(BaseModel):
    """Extracted date with its supporting quote."""
    evidence: str | None = Field(description=EVIDENCE_DESCRIPTION)
    value: date | None = Field(description="Calendar date as YYYY-MM-DD. Null unless stated explicitly.")


class TextField(BaseModel):
    """Extracted free-text value with its supporting quote."""
    evidence: str | None = Field(description=EVIDENCE_DESCRIPTION)
    value: str | None = Field(description="Free-text value. Null unless stated explicitly.")


class ContractTypeField(BaseModel):
    """Extracted contract type with its supporting quote."""
    evidence: str | None = Field(description=EVIDENCE_DESCRIPTION)
    value: ContractType | None = Field()


class PartiesField(BaseModel):
    """Extracted parties with their supporting quote."""
    evidence: str | None = Field(description=EVIDENCE_DESCRIPTION)
    value: list[Party] | None = Field()


class ContractExtraction(BaseModel):
    """Schema the LLM output is constrained to; also served by GET /schema."""
    contract_type: ContractTypeField = Field(
        description="Closest agreement type; 'Other' for other contracts; null if not a contract.")
    parties: PartiesField = Field(
        description="All parties that sign the contract.")
    agreement_date: DateField = Field(
        description="Date the contract is signed or dated 'as of'.")
    effective_date: DateField = Field(
        description="Date the contract takes effect, only if stated explicitly.")
    expiration_date: DateField = Field(
        description="Calendar date the initial term ends. "
        "Null if only a duration is given; never compute it.")
    renewal_term: TextField = Field(
        description="Automatic renewal period as written, e.g. 'successive 1 year'. Null if none.")
    governing_law: TextField = Field(
        description="Jurisdiction name only, e.g. 'Nevada' or 'England and Wales'.")
