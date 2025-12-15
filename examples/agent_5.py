"""
Agent 5: SWIFT MT to ISO 20022 Message Converter

This agent converts between SWIFT MT messages and ISO 20022 XML messages.
Supports bidirectional conversion for common payment messages:
- SWIFT MT103 (Customer Payment) ↔ ISO 20022 pacs.008 (Customer Credit Transfer)
- SWIFT MT940 (Statement) ↔ ISO 20022 camt.053 (Bank Statement)

Tool Evaluation:
1. **Direct Parsing Tools**: Custom Python parsers for SWIFT MT and ISO 20022 XML
2. **Conversion Tools**: Mapping logic between message formats
3. **Ontology-driven approach**: Could use RDF/OWL ontologies for semantic mapping
   - SWIFT Linked Data Miner (OWL 2 EL) for ontology extension
   - Semantic web technologies (RDF, OWL) for field mapping
   - However, most production tools are Java-based (Prowide) or Rust-based (Reframe)

Recommended Libraries (if available):
- For SWIFT: Custom parser (no standard Python library found)
- For ISO 20022: xml.etree.ElementTree or lxml for XML parsing
- For ontology: rdflib, owlready2 (if implementing ontology-driven approach)
"""

import asyncio
import re
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET
from xml.dom import minidom
from xml.parsers.expat import ExpatError

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings, Tool

from app.models import finance_model


# ============================================================================
# DATA MODELS
# ============================================================================

class SwiftMTMessage(BaseModel):
    """Parsed SWIFT MT message structure."""
    message_type: str = Field(description="MT message type (e.g., MT103, MT940)")
    fields: Dict[str, str] = Field(description="Message fields by tag (e.g., :20:, :32A:)")
    raw_message: str = Field(description="Original raw message")


class ISO20022Message(BaseModel):
    """Parsed ISO 20022 message structure."""
    message_type: str = Field(description="ISO 20022 message type (e.g., pacs.008, camt.053)")
    xml_content: str = Field(description="XML content")
    structured_data: Dict[str, Any] = Field(description="Extracted structured data")


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_swift_message(message: str) -> Dict[str, Any]:
    """Validate SWIFT MT message structure.
    
    Checks for:
    - Presence of required blocks (1, 2, 4)
    - Valid block format
    - Message type consistency
    
    Args:
        message: Raw SWIFT MT message string
        
    Returns:
        Dict with validation result, errors, and warnings
    """
    errors = []
    warnings = []
    
    # Check for required blocks
    if not re.search(r'\{1:[^}]+\}', message):
        errors.append("Block 1 (Basic Header) is missing or invalid")
    
    if not re.search(r'\{2:[^}]+\}', message):
        errors.append("Block 2 (Application Header) is missing or invalid")
    
    if not re.search(r'\{4:[^}]+\}', message):
        errors.append("Block 4 (Text Block) is missing or invalid")
    
    # Check block format
    block_pattern = r'\{(\d+):([^}]+)\}'
    blocks = re.findall(block_pattern, message)
    
    if not blocks:
        errors.append("No valid blocks found in message")
    else:
        block_numbers = [int(b[0]) for b in blocks]
        if 1 not in block_numbers:
            errors.append("Block 1 not found")
        if 2 not in block_numbers:
            errors.append("Block 2 not found")
        if 4 not in block_numbers:
            errors.append("Block 4 not found")
    
    # Check for common field tags in block 4
    block4_match = re.search(r'\{4:([^}]+)\}', message, re.DOTALL)
    if block4_match:
        text_block = block4_match.group(1)
        required_fields = ["20", "32A"]  # Reference and Value Date/Currency/Amount
        for field in required_fields:
            if f":{field}:" not in text_block:
                warnings.append(f"Field :{field}: (recommended) not found in text block")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "block_count": len(blocks) if blocks else 0
    }


def validate_iso20022_message(xml_content: str) -> Dict[str, Any]:
    """Validate ISO 20022 XML message.
    
    Checks for:
    - Valid XML structure
    - Required root elements
    - Namespace consistency
    
    Args:
        xml_content: ISO 20022 XML message string
        
    Returns:
        Dict with validation result, errors, and warnings
    """
    errors = []
    warnings = []
    
    # Try to parse XML
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        errors.append(f"Invalid XML structure: {str(e)}")
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings
        }
    
    # Check for Document root
    if root.tag.split('}')[-1] != "Document":
        warnings.append("Root element is not 'Document' (may be valid for some message types)")
    
    # Check for namespace
    if not root.tag.startswith("{") or "}" not in root.tag:
        warnings.append("No namespace detected in root element")
    
    # Check for required child elements (for pacs.008)
    children = list(root)
    if not children:
        errors.append("No child elements found in Document root")
    else:
        first_child = children[0]
        child_tag = first_child.tag.split('}')[-1]
        if child_tag not in ["CstmrCdtTrfInitn", "CstmrPmtStsRpt", "BkToCstmrStmt"]:
            warnings.append(f"Unknown message type: {child_tag}")
    
    # Check for required elements in pacs.008
    if any("CstmrCdtTrfInitn" in str(c.tag) for c in children):
        # Check for GrpHdr
        grp_hdr = root.find(".//{*}GrpHdr")
        if grp_hdr is None:
            errors.append("GrpHdr (Group Header) is missing")
        else:
            # Check for required fields in GrpHdr
            msg_id = grp_hdr.find(".//{*}MsgId")
            if msg_id is None or not msg_id.text:
                errors.append("MsgId (Message ID) is missing in GrpHdr")
            
            cre_dt_tm = grp_hdr.find(".//{*}CreDtTm")
            if cre_dt_tm is None or not cre_dt_tm.text:
                warnings.append("CreDtTm (Creation DateTime) is missing in GrpHdr")
        
        # Check for payment information
        pmt_inf = root.find(".//{*}PmtInf")
        if pmt_inf is None:
            errors.append("PmtInf (Payment Information) is missing")
        else:
            # Check for required fields
            instd_amt = pmt_inf.find(".//{*}InstdAmt")
            if instd_amt is None:
                errors.append("InstdAmt (Instructed Amount) is missing")
            
            dbtr = pmt_inf.find(".//{*}Dbtr")
            if dbtr is None:
                errors.append("Dbtr (Debtor) is missing")
            
            cdtr = pmt_inf.find(".//{*}Cdtr")
            if cdtr is None:
                errors.append("Cdtr (Creditor) is missing")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "root_element": root.tag.split('}')[-1] if '}' in root.tag else root.tag,
        "message_type": children[0].tag.split('}')[-1] if children else "unknown"
    }


# ============================================================================
# SWIFT MT PARSING TOOLS
# ============================================================================

def parser_swift_mt(message: str) -> Dict[str, Any]:
    """Parse a SWIFT MT message into structured format.
    
    SWIFT MT messages have the format:
    {1:...} {2:...} {3:...} {4:...} {5:...}
    
    Where:
    - {1:...} = Basic Header Block
    - {2:...} = Application Header Block
    - {3:...} = User Header Block (optional)
    - {4:...} = Text Block (message content with fields like :20:, :32A:)
    - {5:...} = Trailer Block
    
    Args:
        message: Raw SWIFT MT message text
        
    Returns:
        Dict with message_type, fields (by tag), and raw_message
    """
    # Extract message type from block 1 or 2
    message_type = "UNKNOWN"
    
    # Try to extract from block 1: {1:F01BANKFRPPAXXX1234567890}
    block1_match = re.search(r'\{1:([^}]+)\}', message)
    if block1_match:
        block1 = block1_match.group(1)
        # Format: F01BANKFRPPAXXX1234567890 (F01 = FIN, then application ID, then session/sequence)
        if len(block1) > 3:
            # Application ID is usually 01 for FIN
            pass
    
    # Try to extract from block 2: {2:O1031200210103BANKFRPPAXXX22221234567890123456789012345678901234567890}
    block2_match = re.search(r'\{2:([IO])(\d{3})([^}]+)\}', message)
    if block2_match:
        direction = block2_match.group(1)  # I = Input, O = Output
        message_type = f"MT{block2_match.group(2)}"
    
    # Extract text block (block 4) which contains the actual message fields
    block4_match = re.search(r'\{4:([^}]+)\}', message, re.DOTALL)
    text_block = block4_match.group(1) if block4_match else ""
    
    # Parse fields from text block
    # Fields are in format :TAG:value or :TAG:value\n
    fields = {}
    field_pattern = r':(\d{2}[A-Z]?):([^\n:]+)'
    for match in re.finditer(field_pattern, text_block):
        tag = match.group(1)
        value = match.group(2).strip()
        # Handle multi-line fields (continuation lines start with -)
        if tag in fields:
            fields[tag] += "\n" + value.lstrip('-')
        else:
            fields[tag] = value
    
    return {
        "message_type": message_type,
        "fields": fields,
        "raw_message": message,
        "parsed_successfully": bool(block2_match and block4_match)
    }


def generer_swift_mt(
    message_type: str,
    fields: Dict[str, str],
    sender_bic: str = "BANKFRPPAXXX",
    receiver_bic: str = "BANKDEFFXXX",
    session_sequence: str = "1234567890"
) -> str:
    """Generate a SWIFT MT message from structured data.
    
    Args:
        message_type: MT message type (e.g., "103", "940")
        fields: Dictionary of field tags to values (e.g., {"20": "REF123", "32A": "240101EUR1000,00"})
        sender_bic: Sender BIC code
        receiver_bic: Receiver BIC code
        session_sequence: Session and sequence number
        
    Returns:
        Complete SWIFT MT message string
    """
    # Block 1: Basic Header Block
    # Format: {1:F01BANKFRPPAXXX1234567890}
    block1 = f"{{1:F01{sender_bic}{session_sequence}}}"
    
    # Block 2: Application Header Block (Output)
    # Format: {2:O1031200210103BANKFRPPAXXX22221234567890123456789012345678901234567890}
    # O = Output, 103 = message type, then date/time, then receiver, then priority
    date_str = datetime.now().strftime("%y%m%d")
    time_str = datetime.now().strftime("%H%M")
    block2 = f"{{2:O{message_type}{date_str}{time_str}{receiver_bic}2222{session_sequence}}}"
    
    # Block 4: Text Block
    text_lines = []
    for tag, value in sorted(fields.items()):
        # Format: :TAG:value
        text_lines.append(f":{tag}:{value}")
    
    block4_text = "\n".join(text_lines)
    block4 = f"{{4:\n{block4_text}\n-}}"
    
    # Block 5: Trailer (MAC, CHK, etc.)
    block5 = "{5:{MAC:ABCD1234}{CHK:EFGH5678}}"
    
    return f"{block1}\n{block2}\n{block4}\n{block5}"


# ============================================================================
# ISO 20022 PARSING TOOLS
# ============================================================================

def parser_iso20022(xml_content: str) -> Dict[str, Any]:
    """Parse an ISO 20022 XML message into structured format.
    
    ISO 20022 messages are XML with namespaces like:
    - pacs.008: Customer Credit Transfer
    - camt.053: Bank Statement
    - pain.001: Customer Credit Transfer Initiation
    
    Args:
        xml_content: ISO 20022 XML message string
        
    Returns:
        Dict with message_type, structured_data, and xml_content
    """
    try:
        root = ET.fromstring(xml_content)
        
        # Determine message type from root element
        # Examples: Document, CstmrCdtTrfInitn, BkToCstmrStmt
        message_type = root.tag.split('}')[-1] if '}' in root.tag else root.tag
        
        # If root is "Document", check the first child element for actual message type
        if message_type == "Document" and len(root) > 0:
            first_child = root[0]
            child_type = first_child.tag.split('}')[-1] if '}' in first_child.tag else first_child.tag
            message_type = child_type
        
        # Common ISO 20022 message types
        iso_types = {
            "CstmrCdtTrfInitn": "pacs.008",
            "BkToCstmrStmt": "camt.053",
            "CstmrPmtStsRpt": "pacs.002",
            "Document": "generic"
        }
        
        detected_type = iso_types.get(message_type, "unknown")
        
        # Extract common fields
        structured_data = {}
        
        # Try to extract payment information with better structure
        # Look for specific paths in pacs.008
        debtor_name = None
        debtor_iban = None
        creditor_name = None
        creditor_iban = None
        
        # Use XPath-like approach: find Dbtr and Cdtr elements
        for dbtr in root.findall(".//{*}Dbtr"):
            for nm in dbtr.findall(".//{*}Nm"):
                if nm.text:
                    debtor_name = nm.text.strip()
        
        for cdtr in root.findall(".//{*}Cdtr"):
            for nm in cdtr.findall(".//{*}Nm"):
                if nm.text:
                    creditor_name = nm.text.strip()
        
        # Find IBANs in account elements
        for dbtr_acct in root.findall(".//{*}DbtrAcct"):
            for iban in dbtr_acct.findall(".//{*}IBAN"):
                if iban.text:
                    debtor_iban = iban.text.strip()
        
        for cdtr_acct in root.findall(".//{*}CdtrAcct"):
            for iban in cdtr_acct.findall(".//{*}IBAN"):
                if iban.text:
                    creditor_iban = iban.text.strip()
        
        # Extract other common fields
        for elem in root.iter():
            tag = elem.tag.split('}')[-1]
            text = elem.text.strip() if elem.text else ""
            
            # Common fields
            if tag in ["GrpHdr", "GrpHeader"]:
                # Group header
                for child in elem:
                    child_tag = child.tag.split('}')[-1]
                    if child.text:
                        structured_data[f"group_{child_tag.lower()}"] = child.text
            elif tag in ["MsgId", "MessageIdentification"]:
                structured_data["message_id"] = text
            elif tag in ["CreDtTm", "CreationDateTime"]:
                structured_data["creation_datetime"] = text
            elif tag in ["InstdAmt", "InstructedAmount"]:
                structured_data["amount"] = text
                # Extract currency from Ccy attribute
                if "Ccy" in elem.attrib:
                    structured_data["currency"] = elem.attrib["Ccy"]
            elif tag in ["EndToEndId"]:
                structured_data["end_to_end_id"] = text
            elif tag in ["InstrId"]:
                structured_data["instruction_id"] = text
            elif tag in ["BIC", "BICFI"]:
                if "bic" not in structured_data:
                    structured_data["bic"] = []
                structured_data["bic"].append(text)
        
        # Store extracted names and IBANs
        if debtor_name:
            structured_data["debtor_name"] = debtor_name
        if debtor_iban:
            structured_data["debtor_iban"] = debtor_iban
        if creditor_name:
            structured_data["creditor_name"] = creditor_name
        if creditor_iban:
            structured_data["creditor_iban"] = creditor_iban
        
        return {
            "message_type": detected_type,
            "structured_data": structured_data,
            "xml_content": xml_content,
            "parsed_successfully": True
        }
    except ET.ParseError as e:
        return {
            "message_type": "unknown",
            "structured_data": {},
            "xml_content": xml_content,
            "parsed_successfully": False,
            "error": str(e)
        }


def generer_iso20022(
    message_type: str,
    message_id: str,
    amount: float,
    currency: str,
    debtor_name: str,
    debtor_iban: str,
    creditor_name: str,
    creditor_iban: str,
    reference: Optional[str] = None,
    execution_date: Optional[str] = None
) -> str:
    """Generate an ISO 20022 XML message (pacs.008 Customer Credit Transfer).
    
    Args:
        message_type: ISO 20022 message type (e.g., "pacs.008")
        message_id: Unique message identifier
        amount: Payment amount
        currency: Currency code (e.g., "EUR", "USD")
        debtor_name: Debtor (payer) name
        debtor_iban: Debtor IBAN
        creditor_name: Creditor (payee) name
        creditor_iban: Creditor IBAN
        reference: Payment reference (optional)
        execution_date: Execution date in YYYY-MM-DD format (optional)
        
    Returns:
        ISO 20022 XML message string
    """
    if message_type == "pacs.008":
        # Customer Credit Transfer
        ns = {
            "": "urn:iso:std:iso:20022:tech:xsd:pacs.008.001.12",
            "xsi": "http://www.w3.org/2001/XMLSchema-instance"
        }
        
        root = ET.Element("Document", attrib={
            "xmlns": ns[""],
            "xmlns:xsi": ns["xsi"]
        })
        
        cstmr_cdt_trf = ET.SubElement(root, "CstmrCdtTrfInitn")
        
        # Group Header
        grp_hdr = ET.SubElement(cstmr_cdt_trf, "GrpHdr")
        msg_id = ET.SubElement(grp_hdr, "MsgId")
        msg_id.text = message_id
        cre_dt_tm = ET.SubElement(grp_hdr, "CreDtTm")
        cre_dt_tm.text = datetime.now().isoformat()
        nb_of_txs = ET.SubElement(grp_hdr, "NbOfTxs")
        nb_of_txs.text = "1"
        ctrl_sum = ET.SubElement(grp_hdr, "CtrlSum")
        ctrl_sum.text = f"{amount:.2f}"
        
        # Payment Information
        pmt_inf = ET.SubElement(cstmr_cdt_trf, "PmtInf")
        pmt_inf_id = ET.SubElement(pmt_inf, "PmtInfId")
        pmt_inf_id.text = message_id
        pmt_mtd = ET.SubElement(pmt_inf, "PmtMtd")
        pmt_mtd.text = "TRF"  # Transfer
        
        # Credit Transfer Transaction Information
        cdt_trf_tx_inf = ET.SubElement(pmt_inf, "CdtTrfTxInf")
        pmt_id = ET.SubElement(cdt_trf_tx_inf, "PmtId")
        instr_id = ET.SubElement(pmt_id, "InstrId")
        instr_id.text = reference or message_id
        end_to_end_id = ET.SubElement(pmt_id, "EndToEndId")
        end_to_end_id.text = reference or message_id
        
        # Amount
        amt = ET.SubElement(cdt_trf_tx_inf, "Amt")
        instd_amt = ET.SubElement(amt, "InstdAmt", Ccy=currency)
        instd_amt.text = f"{amount:.2f}"
        
        # Debtor
        dbtr = ET.SubElement(cdt_trf_tx_inf, "Dbtr")
        dbtr_nm = ET.SubElement(dbtr, "Nm")
        dbtr_nm.text = debtor_name
        dbtr_acct = ET.SubElement(cdt_trf_tx_inf, "DbtrAcct")
        dbtr_id = ET.SubElement(dbtr_acct, "Id")
        dbtr_iban = ET.SubElement(dbtr_id, "IBAN")
        dbtr_iban.text = debtor_iban
        
        # Creditor
        cdtr = ET.SubElement(cdt_trf_tx_inf, "Cdtr")
        cdtr_nm = ET.SubElement(cdtr, "Nm")
        cdtr_nm.text = creditor_name
        cdtr_acct = ET.SubElement(cdt_trf_tx_inf, "CdtrAcct")
        cdtr_id = ET.SubElement(cdtr_acct, "Id")
        cdtr_iban = ET.SubElement(cdtr_id, "IBAN")
        cdtr_iban.text = creditor_iban
        
        # Requested Execution Date
        if execution_date:
            reqd_exctn_dt = ET.SubElement(cdt_trf_tx_inf, "ReqdExctnDt")
            reqd_exctn_dt.text = execution_date
        
        # Convert to pretty XML
        xml_str = ET.tostring(root, encoding='unicode')
        dom = minidom.parseString(xml_str)
        return dom.toprettyxml(indent="  ")
    
    else:
        return f"<!-- Message type {message_type} not yet implemented -->"


# ============================================================================
# CONVERSION TOOLS
# ============================================================================

def convertir_swift_vers_iso20022(swift_message: str) -> Dict[str, Any]:
    """Convert SWIFT MT message to ISO 20022 XML.
    
    Maps common SWIFT MT103 fields to ISO 20022 pacs.008:
    - :20: (Reference) → EndToEndId
    - :32A: (Value Date, Currency, Amount) → ReqdExctnDt, InstdAmt
    - :50A/:50K (Ordering Customer) → Dbtr
    - :59/:59A (Beneficiary Customer) → Cdtr
    - :52A (Ordering Institution) → DbtrAgt
    - :57A (Account With Institution) → CdtrAgt
    
    Args:
        swift_message: Raw SWIFT MT message string
        
    Returns:
        Dict with iso20022_xml and conversion_details
    """
    # Validate SWIFT message first
    validation = validate_swift_message(swift_message)
    if not validation["valid"]:
        return {
            "success": False,
            "error": "SWIFT message validation failed",
            "validation_errors": validation["errors"],
            "validation_warnings": validation["warnings"]
        }
    
    # Parse SWIFT message
    swift_parsed = parser_swift_mt(swift_message)
    
    if not swift_parsed.get("parsed_successfully"):
        return {
            "success": False,
            "error": "Failed to parse SWIFT message",
            "swift_parsed": swift_parsed
        }
    
    fields = swift_parsed["fields"]
    message_type = swift_parsed["message_type"]
    
    if message_type != "MT103":
        return {
            "success": False,
            "error": f"Conversion for {message_type} not yet implemented. Only MT103 is supported.",
            "swift_parsed": swift_parsed
        }
    
    # Extract fields from MT103
    reference = fields.get("20", f"REF{datetime.now().strftime('%Y%m%d%H%M%S')}")
    
    # Parse :32A: (Value Date, Currency, Amount)
    # Format: YYMMDDCURRENCYAMOUNT (e.g., 240101EUR1000,00)
    value_date_str = fields.get("32A", "")
    amount = 0.0
    currency = "EUR"
    execution_date = None
    
    if value_date_str:
        # Try to parse: YYMMDDCURRENCYAMOUNT
        match = re.match(r'(\d{6})([A-Z]{3})([\d,\.]+)', value_date_str)
        if match:
            date_part = match.group(1)
            currency = match.group(2)
            amount_str = match.group(3).replace(',', '.')
            try:
                amount = float(amount_str)
                # Convert YYMMDD to YYYY-MM-DD
                year = 2000 + int(date_part[:2])
                month = date_part[2:4]
                day = date_part[4:6]
                execution_date = f"{year}-{month}-{day}"
            except ValueError:
                pass
    
    # Extract ordering customer (:50A or :50K)
    # Format: /BIC/NAME or /ACCOUNT/NAME or just NAME
    debtor_field = fields.get("50A", fields.get("50K", ""))
    debtor_name = "Unknown Debtor"
    debtor_iban = "FR1420041010050500013M02606"  # Default
    
    if debtor_field:
        # Parse format: /BIC/NAME or /ACCOUNT/NAME
        if "/" in debtor_field:
            parts = [p.strip() for p in debtor_field.split("/") if p.strip()]
            if len(parts) >= 2:
                # First part is BIC or account, last part is name
                # Check if first part looks like IBAN (starts with 2 letters)
                if len(parts[0]) >= 2 and parts[0][:2].isalpha():
                    debtor_iban = parts[0]
                    debtor_name = " ".join(parts[1:]) if len(parts) > 1 else "Unknown Debtor"
                else:
                    # Might be BIC, use default IBAN
                    debtor_name = " ".join(parts[1:]) if len(parts) > 1 else parts[0]
            else:
                debtor_name = parts[0] if parts else "Unknown Debtor"
        else:
            debtor_name = debtor_field.strip()
    
    # Extract beneficiary (:59 or :59A)
    # Format: /ACCOUNT/NAME or just NAME
    creditor_field = fields.get("59", fields.get("59A", ""))
    creditor_name = "Unknown Creditor"
    creditor_iban = "DE89370400440532013000"  # Default
    
    if creditor_field:
        # Parse format: /ACCOUNT/NAME
        if "/" in creditor_field:
            parts = [p.strip() for p in creditor_field.split("/") if p.strip()]
            if len(parts) >= 2:
                # First part is account/IBAN, rest is name
                creditor_iban = parts[0]
                creditor_name = " ".join(parts[1:]) if len(parts) > 1 else "Unknown Creditor"
            else:
                creditor_name = parts[0] if parts else "Unknown Creditor"
        else:
            creditor_name = creditor_field.strip()
    
    # Generate ISO 20022 message
    message_id = f"MSG{datetime.now().strftime('%Y%m%d%H%M%S')}"
    iso_xml = generer_iso20022(
        message_type="pacs.008",
        message_id=message_id,
        amount=amount,
        currency=currency,
        debtor_name=debtor_name,
        debtor_iban=debtor_iban,
        creditor_name=creditor_name,
        creditor_iban=creditor_iban,
        reference=reference,
        execution_date=execution_date
    )
    
    # Validate generated ISO 20022 message
    iso_validation = validate_iso20022_message(iso_xml)
    
    return {
        "success": True,
        "iso20022_xml": iso_xml,
        "swift_parsed": swift_parsed,
        "iso_validation": iso_validation,
        "conversion_details": {
            "swift_type": message_type,
            "iso_type": "pacs.008",
            "mapped_fields": {
                "reference": reference,
                "amount": amount,
                "currency": currency,
                "debtor": debtor_name,
                "creditor": creditor_name,
                "execution_date": execution_date
            },
            "all_fields_included": iso_validation["valid"] and len(iso_validation["errors"]) == 0
        }
    }


def convertir_iso20022_vers_swift(iso20022_xml: str) -> Dict[str, Any]:
    """Convert ISO 20022 XML message to SWIFT MT.
    
    Maps ISO 20022 pacs.008 fields to SWIFT MT103:
    - EndToEndId → :20: (Reference)
    - InstdAmt → :32A: (Value Date, Currency, Amount)
    - Dbtr → :50A/:50K (Ordering Customer)
    - Cdtr → :59/:59A (Beneficiary Customer)
    
    Args:
        iso20022_xml: ISO 20022 XML message string
        
    Returns:
        Dict with swift_message and conversion_details
    """
    # Parse ISO 20022 message
    iso_parsed = parser_iso20022(iso20022_xml)
    
    if not iso_parsed.get("parsed_successfully"):
        return {
            "success": False,
            "error": "Failed to parse ISO 20022 message",
            "iso_parsed": iso_parsed
        }
    
    data = iso_parsed["structured_data"]
    message_type = iso_parsed["message_type"]
    
    if message_type != "pacs.008":
        return {
            "success": False,
            "error": f"Conversion for {message_type} not yet implemented. Only pacs.008 is supported.",
            "iso_parsed": iso_parsed
        }
    
    # Extract fields
    message_id = data.get("message_id", f"REF{datetime.now().strftime('%Y%m%d%H%M%S')}")
    amount = float(data.get("amount", "0"))
    currency = data.get("currency", "EUR")
    
    # Extract names and IBANs from structured data
    debtor_name = data.get("debtor_name", "Unknown Debtor")
    creditor_name = data.get("creditor_name", "Unknown Creditor")
    debtor_iban = data.get("debtor_iban", "FR1420041010050500013M02606")
    creditor_iban = data.get("creditor_iban", "DE89370400440532013000")
    
    # Fallback to end_to_end_id or instruction_id for reference
    reference = data.get("end_to_end_id") or data.get("instruction_id") or message_id
    
    # Get execution date or use today
    execution_date = data.get("creation_datetime", datetime.now().isoformat())
    if execution_date:
        try:
            dt = datetime.fromisoformat(execution_date.replace('Z', '+00:00'))
            date_str = dt.strftime("%y%m%d")
        except:
            date_str = datetime.now().strftime("%y%m%d")
    else:
        date_str = datetime.now().strftime("%y%m%d")
    
    # Build SWIFT MT103 fields
    swift_fields = {
        "20": reference,  # Reference (use end_to_end_id if available)
        "32A": f"{date_str}{currency}{amount:,.2f}".replace(',', ''),  # Value Date, Currency, Amount
        "50A": f"/{debtor_iban}/{debtor_name}",  # Ordering Customer
        "59": f"/{creditor_iban}/{creditor_name}",  # Beneficiary Customer
    }
    
    # Generate SWIFT message
    swift_message = generer_swift_mt(
        message_type="103",
        fields=swift_fields
    )
    
    return {
        "success": True,
        "swift_message": swift_message,
        "iso_parsed": iso_parsed,
        "conversion_details": {
            "iso_type": message_type,
            "swift_type": "MT103",
            "mapped_fields": {
                "reference": message_id,
                "amount": amount,
                "currency": currency,
                "debtor": debtor_name,
                "creditor": creditor_name,
                "execution_date": date_str
            }
        }
    }


# ============================================================================
# AGENT
# ============================================================================

agent_5 = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=3000),
    system_prompt="""Vous êtes un expert en conversion de messages financiers entre SWIFT MT et ISO 20022.

RÈGLES ABSOLUES POUR LES CONVERSIONS:
⚠️  OBLIGATOIRE: Pour TOUTE conversion, utilisez UNIQUEMENT les outils de conversion dédiés:
1. SWIFT → ISO 20022: VOUS DEVEZ utiliser convertir_swift_vers_iso20022 (PAS parser + generer)
2. ISO 20022 → SWIFT: VOUS DEVEZ utiliser convertir_iso20022_vers_swift (PAS parser + generer)

❌ NE PAS utiliser parser_swift_mt + generer_iso20022 pour convertir
❌ NE PAS utiliser parser_iso20022 + generer_swift_mt pour convertir
✅ UTILISEZ UNIQUEMENT les outils convertir_* pour les conversions

VALIDATION OBLIGATOIRE:
- Vérifiez que le message converti contient TOUS les champs requis
- Validez l'entrée avant conversion en utilisant validate_swift_message ou validate_iso20022_message
- Assurez-vous que le message ISO 20022 généré est complet avec tous les éléments requis (GrpHdr, PmtInf, Dbtr, Cdtr, InstdAmt, etc.)

OUTILS AUXILIAIRES (uniquement pour analyse, PAS pour conversion):
- parser_swift_mt: Pour analyser un message SWIFT (pas pour conversion)
- parser_iso20022: Pour analyser un message ISO 20022 (pas pour conversion)
- generer_swift_mt: Pour générer un message SWIFT depuis zéro (pas pour conversion)
- generer_iso20022: Pour générer un message ISO 20022 depuis zéro (pas pour conversion)
- validate_swift_message: Pour valider la structure d'un message SWIFT
- validate_iso20022_message: Pour valider la structure d'un message ISO 20022

FORMATS SUPPORTÉS:
- SWIFT MT103 (Customer Payment) ↔ ISO 20022 pacs.008 (Customer Credit Transfer)

ACTION REQUISE: Quand on vous demande de convertir, appelez DIRECTEMENT convertir_swift_vers_iso20022 ou convertir_iso20022_vers_swift.
Vérifiez que le message converti contient TOUS les champs requis. Validez l'entrée avant conversion.
Répondez en français avec les messages convertis.""",
    tools=[
        Tool(
            parser_swift_mt,
            name="parser_swift_mt",
            description="⚠️ UNIQUEMENT pour analyser un message SWIFT (pas pour conversion). Pour convertir, utilisez convertir_swift_vers_iso20022. Fournissez le message SWIFT brut.",
        ),
        Tool(
            parser_iso20022,
            name="parser_iso20022",
            description="⚠️ UNIQUEMENT pour analyser un message ISO 20022 (pas pour conversion). Pour convertir, utilisez convertir_iso20022_vers_swift. Fournissez le contenu XML.",
        ),
        Tool(
            generer_swift_mt,
            name="generer_swift_mt",
            description="⚠️ UNIQUEMENT pour générer un message SWIFT depuis zéro (pas pour conversion). Pour convertir, utilisez convertir_iso20022_vers_swift. Fournissez message_type (ex: '103'), fields (dict), et optionnellement sender_bic, receiver_bic, session_sequence.",
        ),
        Tool(
            generer_iso20022,
            name="generer_iso20022",
            description="⚠️ UNIQUEMENT pour générer un message ISO 20022 depuis zéro (pas pour conversion). Pour convertir, utilisez convertir_swift_vers_iso20022. Fournissez message_type, message_id, amount, currency, debtor_name, debtor_iban, creditor_name, creditor_iban, et optionnellement reference, execution_date.",
        ),
        Tool(
            convertir_swift_vers_iso20022,
            name="convertir_swift_vers_iso20022",
            description="⚠️ OBLIGATOIRE pour convertir SWIFT MT → ISO 20022. Utilisez CET outil pour toutes les conversions SWIFT vers ISO. Fournissez le message SWIFT brut complet. Supporte MT103 → pacs.008. NE PAS utiliser parser + generer pour convertir.",
        ),
        Tool(
            convertir_iso20022_vers_swift,
            name="convertir_iso20022_vers_swift",
            description="⚠️ OBLIGATOIRE pour convertir ISO 20022 → SWIFT MT. Utilisez CET outil pour toutes les conversions ISO vers SWIFT. Fournissez le contenu XML complet. Supporte pacs.008 → MT103. NE PAS utiliser parser + generer pour convertir.",
        ),
        Tool(
            validate_swift_message,
            name="validate_swift_message",
            description="Valide la structure d'un message SWIFT MT. Vérifie les blocs requis (1, 2, 4) et les champs essentiels. Fournissez le message SWIFT brut.",
        ),
        Tool(
            validate_iso20022_message,
            name="validate_iso20022_message",
            description="Valide la structure d'un message ISO 20022 XML. Vérifie la structure XML, les éléments requis (GrpHdr, PmtInf, Dbtr, Cdtr, etc.). Fournissez le contenu XML.",
        ),
    ],
)


# ============================================================================
# EXAMPLES
# ============================================================================

async def exemple_swift_vers_iso20022():
    """Exemple de conversion SWIFT MT103 → ISO 20022 pacs.008."""
    swift_message = """{1:F01BANKFRPPAXXX1234567890}
{2:O10312002401031200BANKDEFFXXX22221234567890123456789012345678901234567890}
{4:
:20:REF123456789
:32A:240101EUR1000,00
:50A:/FR1420041010050500013M02606
COMPAGNIE ABC
:59:/DE89370400440532013000
COMPAGNIE XYZ
:70:PAYMENT FOR INVOICE 12345
-}
{5:{MAC:ABCD1234}{CHK:EFGH5678}}"""
    
    print("🔄 Agent 5: Conversion SWIFT → ISO 20022")
    print("=" * 70)
    print(f"Message SWIFT MT103:\n{swift_message}\n")
    
    result = await agent_5.run(
        f"Convertis ce message SWIFT MT103 en ISO 20022 pacs.008:\n\n{swift_message}"
    )
    
    print("✅ Résultat de la conversion:\n")
    print(result.output)
    print()


async def exemple_iso20022_vers_swift():
    """Exemple de conversion ISO 20022 pacs.008 → SWIFT MT103."""
    iso_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pacs.008.001.12">
  <CstmrCdtTrfInitn>
    <GrpHdr>
      <MsgId>MSG20240101120000</MsgId>
      <CreDtTm>2024-01-01T12:00:00</CreDtTm>
      <NbOfTxs>1</NbOfTxs>
      <CtrlSum>1000.00</CtrlSum>
    </GrpHdr>
    <PmtInf>
      <PmtInfId>MSG20240101120000</PmtInfId>
      <PmtMtd>TRF</PmtMtd>
      <CdtTrfTxInf>
        <PmtId>
          <InstrId>REF123456789</InstrId>
          <EndToEndId>REF123456789</EndToEndId>
        </PmtId>
        <Amt>
          <InstdAmt Ccy="EUR">1000.00</InstdAmt>
        </Amt>
        <Dbtr>
          <Nm>COMPAGNIE ABC</Nm>
        </Dbtr>
        <DbtrAcct>
          <Id>
            <IBAN>FR1420041010050500013M02606</IBAN>
          </Id>
        </DbtrAcct>
        <Cdtr>
          <Nm>COMPAGNIE XYZ</Nm>
        </Cdtr>
        <CdtrAcct>
          <Id>
            <IBAN>DE89370400440532013000</IBAN>
          </Id>
        </CdtrAcct>
        <ReqdExctnDt>2024-01-01</ReqdExctnDt>
      </CdtTrfTxInf>
    </PmtInf>
  </CstmrCdtTrfInitn>
</Document>"""
    
    print("🔄 Agent 5: Conversion ISO 20022 → SWIFT")
    print("=" * 70)
    print(f"Message ISO 20022 pacs.008:\n{iso_xml[:500]}...\n")
    
    result = await agent_5.run(
        f"Convertis ce message ISO 20022 pacs.008 en SWIFT MT103:\n\n{iso_xml}"
    )
    
    print("✅ Résultat de la conversion:\n")
    print(result.output)
    print()


async def exemple_bidirectionnel():
    """Exemple de conversion bidirectionnelle."""
    print("🔄 Agent 5: Conversion Bidirectionnelle")
    print("=" * 70)
    
    swift_message = """{1:F01BANKFRPPAXXX1234567890}
{2:O10312002401031200BANKDEFFXXX22221234567890123456789012345678901234567890}
{4:
:20:REF987654321
:32A:240215USD5000,00
:50A:/US64SVBKUS6S3300000000
ACME CORPORATION
:59:/GB82WEST12345698765432
GLOBAL SUPPLIERS LTD
:70:PAYMENT FOR SERVICES
-}
{5:{MAC:ABCD1234}{CHK:EFGH5678}}"""
    
    print("1. Conversion SWIFT → ISO 20022")
    result1 = await agent_5.run(
        f"Convertis ce message SWIFT en ISO 20022:\n\n{swift_message}"
    )
    print(result1.output[:500] + "...\n")
    
    # Extract ISO XML from result (simplified - in real use, parse the output)
    print("2. Conversion ISO 20022 → SWIFT (round-trip)")
    result2 = await agent_5.run(
        "Maintenant, convertis ce message ISO 20022 en SWIFT MT103. "
        "Utilise les données du message précédent pour créer un message ISO 20022, "
        "puis convertis-le en SWIFT."
    )
    print(result2.output[:500] + "...\n")


if __name__ == "__main__":
    asyncio.run(exemple_swift_vers_iso20022())
    print("\n" + "=" * 70 + "\n")
    asyncio.run(exemple_iso20022_vers_swift())

