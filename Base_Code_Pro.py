import pubchempy as pcp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem , Draw
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
import os
os.environ["CHROMADB_DEFAULT_DATABASE"] = "duckdb"
#from crewai import Agent, Task, Crew, LLM
#from crewai_tools import SerperDevTool
import base64
from io import BytesIO
import datetime
import google.generativeai as genai
from google.generativeai import types
from mistralai import Mistral




# Suppress warnings
warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

import streamlit as st
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski, Draw, QED
import matplotlib.pyplot as plt
import math
from PIL import Image
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def compare_fingerprints_sy(smiles, radius=6, nBits=150):
    morgan_gen = GetMorganGenerator(radius=radius, fpSize=nBits)
    mol = Chem.MolFromSmiles(smiles)
    fp = morgan_gen.GetFingerprint(mol)
    return fp.ToBitString()

def dose_response(x, IC50):
    A1, A2, B = 0, 100, 1
    return A1 + (A2 - A1) / (1 + (x / IC50)**(-B))

def compare_fingerprints(smiles1, radius=6, nBits=150):
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    mol1 = Chem.MolFromSmiles(smiles1)
    
    if mol1 is None:
        # Return zero vector if SMILES is invalid
        return '0' * nBits
    
    fp1 = morgan_gen.GetFingerprint(mol1)
    bit_str1 = fp1.ToBitString()
    
    # Ensure exactly nBits length
    if len(bit_str1) != nBits:
        bit_str1 = bit_str1[:nBits] + '0' * (nBits - len(bit_str1))
    
    return bit_str1

class ClassificationModel(nn.Module):
    def __init__(self, input_dim=150, num_classes=3):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)  # No activation for logits
        return x
          
class CNNRegressionModel(nn.Module):
    def __init__(self, input_dim=150):
        super(CNNRegressionModel, self).__init__()
        # Assumes input is (batch_size, channels, input_dim), here channels=1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(64 * (input_dim // 8), 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#chnanged 29 july
def bit_string_to_tensor(bit_string, n_bits=150, device='cpu'):
    bit_list = [int(bit) for bit in bit_string]
    # Ensure we have exactly n_bits
    bit_list = bit_list[:n_bits] + [0] * (n_bits - len(bit_list))
    bit_tensor = torch.tensor(bit_list, dtype=torch.float32, device=device).unsqueeze(0)
    return bit_tensor


# Define the function to predict using the trained model
def predict_y(model, smiles1, n_bits=150):
    bit_string = compare_fingerprints(smiles1, nBits=n_bits)
    bit_tensor = bit_string_to_tensor(bit_string, n_bits)
    
    model.eval()
    with torch.no_grad():
        prediction = model(bit_tensor)
    
    return prediction.item()

#changed : 29-07-2025 :
def predict_cyp(model, smiles1, n_bits):
    device = next(model.parameters()).device  # detect model's device
    bit_string = compare_fingerprints(smiles1, nBits=n_bits)
    bit_tensor = bit_string_to_tensor(bit_string, n_bits).to(device)
    model.eval()
    with torch.no_grad():
        output = model(bit_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class




def bit_string_to_tensor_sy(bit_string1, bit_string2, n_bits=150):
    bit_list1 = [int(bit) for bit in bit_string1]
    bit_list2 = [int(bit) for bit in bit_string2]
    combined_bits = bit_list1 + bit_list2
    bit_tensor = torch.tensor(combined_bits, dtype=torch.float32).unsqueeze(0)
    return bit_tensor

def predict_sy(model, smiles1, smiles2, n_bits=150):
    bit_string1 = compare_fingerprints_sy(smiles1, nBits=n_bits)
    bit_string2 = compare_fingerprints_sy(smiles2, nBits=n_bits)
    bit_tensor = bit_string_to_tensor_sy(bit_string1, bit_string2, n_bits * 2)
    model_sy.eval()
    with torch.no_grad():
        prediction = model_sy(bit_tensor)
    return prediction.item()

model_sy= CNNRegressionModel(input_dim=300)
model = CNNRegressionModel(input_dim=150)
model_cyp = ClassificationModel(input_dim=150)
def lipinski_rule_of_five(mol):
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    h_donors = Lipinski.NumHDonors(mol)
    h_acceptors = Lipinski.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, h_donors > 5, h_acceptors > 10])
    return violations, violations <= 1

def water_solubility(mol):
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    mw = Descriptors.MolWt(mol)
    logS = -0.74 * logp - 0.006 * mw + 0.003 * tpsa + 0.63
    return logS

def synthetic_accessibility(mol):
    return 1 - QED.qed(mol)  # Lower means easier synthesis

def bioavailability_score(mol):
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    return int(tpsa <= 140 and rot_bonds <= 10)

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        return img
    return None

def predict_adme(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {"Error": "Invalid SMILES"}

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    h_donors = Lipinski.NumHDonors(mol)
    h_acceptors = Lipinski.NumHAcceptors(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)

    lipinski_violations, lipinski_pass = lipinski_rule_of_five(mol)
    logS = water_solubility(mol)
    sa_score = synthetic_accessibility(mol)
    bio_score = bioavailability_score(mol)
    
    bbb_permeability = -0.3 < logp < 6 and mw < 450 and h_donors <= 3 and h_acceptors <= 7

    result = [
        ("Molecular Weight", mw, "Should be < 500 for good permeability"),
        ("logP", logp, "Measures hydrophobicity; affects absorption"),
        ("TPSA", tpsa, "Below 140 Å² favors permeability"),
        ("H-bond Donors", h_donors, "Should be ≤ 5 for drug-likeness"),
        ("H-bond Acceptors", h_acceptors, "Should be ≤ 10 for permeability"),
        ("Rotatable Bonds", rot_bonds, "Flexibility affects oral bioavailability"),
        ("Lipinski Violations", lipinski_violations, "≤ 1 violation preferred"),
        ("Lipinski Rule of Five Pass", lipinski_pass, "Indicates drug-likeness"),
        ("Water Solubility (LogS)", logS, "Lower LogS = better solubility"),
        ("Synthetic Accessibility", sa_score, "Lower value = easier synthesis"),
        ("Bioavailability Score", bio_score, "1 indicates good oral bioavailability"),
        ("BBB Permeability (Heuristic)", bbb_permeability, "Predicts CNS drug potential")
    ]
    return result

def generate_comprehensive_ai_insight(smiles: str, adme_properties: list, cyp_predictions: dict) -> dict:
    """
    Generate both a narrative summary AND structured ADME inference in one Mistral call.
    Returns a dict with keys: 'narrative_summary', 'adme_inference'
    """
    try:
        prop_dict = {name: value for name, value, _ in adme_properties}
        mw = prop_dict.get("Molecular Weight", "N/A")
        logp = prop_dict.get("logP", "N/A")
        tpsa = prop_dict.get("TPSA", "N/A")
        h_donors = prop_dict.get("H-bond Donors", "N/A")
        h_acceptors = prop_dict.get("H-bond Acceptors", "N/A")
        rot_bonds = prop_dict.get("Rotatable Bonds", "N/A")
        lipinski_violations = prop_dict.get("Lipinski Violations", "N/A")
        lipinski_pass = prop_dict.get("Lipinski Rule of Five Pass", "N/A")
        water_solubility = prop_dict.get("Water Solubility (LogS)", "N/A")
        bioavailability = prop_dict.get("Bioavailability Score", "N/A")
        bbb_permeability = prop_dict.get("BBB Permeability (Heuristic)", "N/A")

        inhibited = [enz for enz, pred in cyp_predictions.items() if pred == 1]
        cyp_text = f"inhibits {', '.join(inhibited)}" if inhibited else "does not inhibit major CYP450 enzymes"

        prompt = f"""Analyze the following molecular data for SMILES: {smiles}

Molecular Properties:
- Molecular Weight: {mw:.2f} Da
- LogP (Hydrophobicity): {logp:.2f}
- TPSA: {tpsa:.2f} Å²
- H-bond Donors: {h_donors}
- H-bond Acceptors: {h_acceptors}
- Rotatable Bonds: {rot_bonds}
- Lipinski Violations: {lipinski_violations}
- Passes Lipinski Rule of Five: {lipinski_pass}
- Water Solubility (LogS): {water_solubility:.2f}
- Bioavailability Score: {bioavailability}
- BBB Permeability: {bbb_permeability}
- CYP Enzyme Interactions: {cyp_text}

Your task is to return a JSON object with two fields:
1. "narrative_summary": A concise scientific paragraph (150–200 words) covering drug-likeness, ADME behavior, therapeutic relevance, and red flags.
2. "adme_inference": A structured breakdown under four subheaders: Absorption, Distribution, Metabolism, Excretion — each as a list of bullet points (no sub-bullet points, no markdown, plain text bullets like "- ...").

Write in a professional, medicinal chemistry tone. Return ONLY valid JSON. Do not include any other text.
"""

        client = Mistral(api_key="wsX20qDpw5DmnFf0vc8PBU7fOdzTfttc")
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.3
        )

        raw_output = response.choices[0].message.content.strip()
        # Safely parse JSON (Mistral sometimes adds markdown code fences)
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]

        import json
        result = json.loads(raw_output)
        return {
            "narrative_summary": result.get("narrative_summary", "Summary not available."),
            "adme_inference": result.get("adme_inference", "Inference not available.")
        }

    except Exception as e:
        error_msg = f"⚠️ AI insight unavailable: {str(e)}"
        return {
            "narrative_summary": error_msg,
            "adme_inference": error_msg
        }

def generate_binding_affinity_insight(smiles: str, protein: str, binding_affinity_kcal: float, kd_value: float) -> str:
    """
    Generate a concise, expert-level inference for binding affinity results.
    """
    try:
        prompt = f"""You are a medicinal chemist analyzing a drug-target interaction.
SMILES: {smiles}
Target Protein: {protein}
Predicted Binding Affinity (ΔG): {binding_affinity_kcal:.2f} kcal/mol
Predicted Kd (Equilibrium Dissociation Constant): {kd_value:.4f} µM

Interpret this result in 3–5 sentences:
- What does this binding strength imply about drug potency?
- Is this Kd value typical for a drug candidate?
- Are there any red flags or notable strengths?
- Keep tone professional, concise, and evidence-based.

Do not include disclaimers or markdown. Return only the paragraph."""
        
        client = Mistral(api_key="wsX20qDpw5DmnFf0vc8PBU7fOdzTfttc")
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"⚠️ AI inference unavailable: {str(e)}"

def generate_synergy_insight(smiles1: str, smiles2: str, cell_line: str, bliss_score: float) -> str:
    """
    Generate a concise, expert-level interpretation of drug synergy prediction.
    """
    try:
        # Determine synergy category
        if bliss_score > 1:
            synergy_type = "synergistic"
        elif bliss_score >= -1:
            synergy_type = "additive"
        else:
            synergy_type = "antagonistic"

        prompt = f"""You are a pharmacologist analyzing a drug combination.
Drug 1 SMILES: {smiles1}
Drug 2 SMILES: {smiles2}
Cell Line: {cell_line}
Predicted Bliss Synergy Score: {bliss_score:.2f}

This score is interpreted as **{synergy_type}** interaction.

Provide a concise 3–5 sentence expert interpretation:
- What does this synergy type imply for therapeutic use?
- Is this result promising or concerning?
- Any caveats or next steps for experimental validation?
- Keep tone professional and evidence-based.

Return only the paragraph. No disclaimers, no markdown."""
        
        client = Mistral(api_key="wsX20qDpw5DmnFf0vc8PBU7fOdzTfttc")
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"⚠️ AI synergy insight unavailable: {str(e)}"

# --- PDF GENERATION FUNCTION (TOP-LEVEL) ---
def render_pdf_report():
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from io import BytesIO
    import base64
    
    # Create a BytesIO buffer for the PDF
    buffer = BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                           rightMargin=0.75*inch, leftMargin=0.75*inch,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    # Get styles and create custom ones
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1e40af')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#1e40af'),
        borderWidth=1,
        borderColor=colors.HexColor('#ddd'),
        borderPadding=4
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    )
    
    # Build the story (content)
    story = []
    
    # Header
    story.append(Paragraph("DrugXplorer", title_style))
    story.append(Paragraph("Smart AI/ML Drug Analysis Tool — MEDxAI Innovations Pvt. Ltd.", 
                          ParagraphStyle('subtitle', parent=styles['Normal'], 
                                       fontSize=10, alignment=TA_CENTER, 
                                       textColor=colors.HexColor('#555'))))
    story.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%B %d, %Y')}", 
                          ParagraphStyle('date', parent=styles['Normal'], 
                                       fontSize=9, alignment=TA_CENTER, 
                                       textColor=colors.HexColor('#777'))))
    story.append(Spacer(1, 20))
    
    # Get data from session state
    adme_data = st.session_state.report_data.get('adme')
    binding_data = st.session_state.report_data.get('binding')
    synergy_data = st.session_state.report_data.get('synergy')
    
    # ADME Analysis Section
    if adme_data:
        story.append(Paragraph("🔬 ADME Analysis Results", heading_style))
        story.append(Paragraph(f"<b>Drug:</b> {adme_data.get('drug_name', 'N/A')} | <b>SMILES:</b> {adme_data.get('smiles', 'N/A')}", normal_style))
        
        # Add molecule image if available
        if adme_data.get('mol_img'):
            try:
                img_buffer = BytesIO()
                adme_data['mol_img'].save(img_buffer, format="PNG")
                img_buffer.seek(0)
                img = Image(img_buffer, width=2*inch, height=2*inch)
                story.append(img)
                story.append(Spacer(1, 12))
            except:
                pass
        
        # ADME Properties Table
        if adme_data.get('properties'):
            table_data = [['Property', 'Value', 'Interpretation']]
            for prop in adme_data['properties']:
                table_data.append([str(prop[0]), str(prop[1]), str(prop[2])])
            
            table = Table(table_data, colWidths=[2*inch, 1.5*inch, 3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f5ff')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#ccc'))
            ]))
            story.append(table)
            story.append(Spacer(1, 12))
        
        # CYP Enzyme Inhibition
        if adme_data.get('cyp_predictions'):
            story.append(Paragraph("CYP Enzyme Inhibition", ParagraphStyle('subheading', parent=styles['Heading3'], fontSize=14)))
            cyp_table_data = [['Enzyme', 'Prediction']]
            for enzyme, pred in adme_data['cyp_predictions'].items():
                prediction = 'Inhibitor' if pred == 1 else 'Non-inhibitor'
                cyp_table_data.append([enzyme, prediction])
            
            cyp_table = Table(cyp_table_data, colWidths=[2*inch, 2*inch])
            cyp_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f5ff')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#ccc'))
            ]))
            story.append(cyp_table)
            story.append(Spacer(1, 12))
        
        # AI Summary
        if adme_data.get('ai_summary'):
            story.append(Paragraph("AI-Generated Scientific Summary", ParagraphStyle('subheading', parent=styles['Heading3'], fontSize=14)))
            story.append(Paragraph(adme_data['ai_summary'], normal_style))
            story.append(Spacer(1, 20))
    
    # Binding Affinity Section
    if binding_data:
        story.append(Paragraph("⚛️ Binding Affinity Results", heading_style))
        story.append(Paragraph(f"<b>Drug:</b> {binding_data.get('drug_name', 'N/A')} | <b>SMILES:</b> {binding_data.get('smiles', 'N/A')}", normal_style))
        story.append(Paragraph(f"<b>Target Protein:</b> {binding_data.get('protein', 'N/A')} ({binding_data.get('protein_group', 'N/A')})", normal_style))
        story.append(Paragraph(f"<b>Binding Affinity (ΔG):</b> {binding_data.get('binding_affinity', 'N/A')}", normal_style))
        story.append(Paragraph(f"<b>Equilibrium Constant (Kd):</b> {binding_data.get('equilibrium_constant', 'N/A')}", normal_style))
        
        if binding_data.get('inference'):
            story.append(Paragraph("AI Interpretation", ParagraphStyle('subheading', parent=styles['Heading3'], fontSize=14)))
            story.append(Paragraph(binding_data['inference'], normal_style))
        story.append(Spacer(1, 20))
    
    # Drug Synergy Section
    if synergy_data:
        story.append(Paragraph("🧪 Drug Synergy Results", heading_style))
        story.append(Paragraph(f"<b>Drug 1:</b> {synergy_data.get('drug_name_1', 'N/A')} | <b>SMILES:</b> {synergy_data.get('smiles_1', 'N/A')}", normal_style))
        story.append(Paragraph(f"<b>Drug 2:</b> {synergy_data.get('drug_name_2', 'N/A')} | <b>SMILES:</b> {synergy_data.get('smiles_2', 'N/A')}", normal_style))
        story.append(Paragraph(f"<b>Cell Line:</b> {synergy_data.get('cell_line', 'N/A')}", normal_style))
        story.append(Paragraph(f"<b>Bliss Synergy Score:</b> {synergy_data.get('bliss_score', 'N/A')}", normal_style))
        
        if synergy_data.get('inference'):
            story.append(Paragraph("AI Interpretation", ParagraphStyle('subheading', parent=styles['Heading3'], fontSize=14)))
            story.append(Paragraph(synergy_data['inference'], normal_style))
        story.append(Spacer(1, 20))
    
    # Footer
    story.append(Spacer(1, 30))
    story.append(Paragraph("Confidential — Generated by DrugXplorer | MEDxAI Innovations Pvt. Ltd.", 
                          ParagraphStyle('footer', parent=styles['Normal'], 
                                       fontSize=9, alignment=TA_CENTER, 
                                       textColor=colors.HexColor('#666'))))
    
    # Build the PDF
    doc.build(story)
    
    # Get the PDF data
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data


userchoice=["Name","SMILES"]
userchoice2=["Name","SMILES"]
userchoice3=["Name","SMILES"]
cell_line=["MCF7","7860","A549","DU145","HCT116","K562","OVCAR3","SNB75"]
st.set_page_config(page_title="💊DrugXplorer",page_icon=":pill")
st.markdown("# Drug:blue[X]plorer")


#modified
tab = st.sidebar.radio(
    "**Navigation**",
    ["🏠 Home", "🔬 ADME Analysis", "⚛️ Binding Affinity", "🧪 Drug Synergy", "📊 Generate Report"]
)
protein_groups = {
    "Nuclear Receptors": ["PPARD","PPARG", "AR", "ESR1", "NR3C1"],
    "Kinases & Cell Signaling": ["ABL1", "JAK2", "AKT1", "MAPK1", "PLK1","EGFR"],
    "Enzymes and Metabolic Targets": ["HMGCR", "PTGS2", "CYP3A4", "DPP4"],
    "Neurotransmitter and Neurological Targets": ["ADRB1", "ADORA2A", "DRD2", "ACHE", "BACE1"],
    "Cancer Therapeutic Targets": ["CASP3", "PARP1", "ROCK1","KDR"]
}
#newly added
def generate_report_data():
    """Store report data in session state"""
    if 'report_data' not in st.session_state:
        st.session_state.report_data = {
            'adme': None,
            'binding': None,
            'synergy': None,
            'timestamp': None
        }

if tab =="🏠 Home":
    st.markdown("#### :blue[*Welcome to DrugXplorer, your AI-powered companion for drug discovery*]")

    st.markdown(
    """
    <div style="text-align: justify;">
    
    :blue[**DrugXplorer**] is an advanced web application designed to streamline the drug discovery process by providing insights into key molecular properties. \n
    Whether you're a researcher, chemist, or biotechnology enthusiast, **DrugXplorer** enables you to predict crucial pharmacokinetic properties, assess drug-protein interactions, and analyze potential drug synergies—all in one platform.
    
    ### :blue[Features:]
    🔬 **ADME Prediction** - :gray[Evaluate the Absorption, Distribution, Metabolism, and Excretion (ADME) properties of drug-like molecules.]  
    🧬 **Binding Affinity Analysis** - :gray[Predict the binding strength between a drug molecule and various target proteins.]  
    💊 **Drug Synergy Prediction** - :gray[Analyze potential synergistic effects between drug combinations.]  
    
    :gray[Navigate through the different tabs to perform specific analyses:]
    - **ADME Properties**: :gray[Input your molecule's name or SMILES representation and obtain detailed ADME predictions.]
    - **Binding Affinity**: :gray[Select a target protein and provide a drug molecule to predict binding affinity values.]
    - **Drug Synergy**: :gray[Explore drug pair interactions and their potential for combination therapy.]
    
    **Harness the power of AI-driven drug discovery with DrugXplorer and accelerate your research with data-driven insights!** 
    
    </div>
    """, 
    unsafe_allow_html=True
)
if tab =="🔬 ADME Analysis":
    generate_report_data() #add this line
    st.subheader("ADME Analysis")
    st.markdown(
    """
    - ### **🔬Molecular Properties:**
        - **Molecular Weight (MW):** Measures the size of the molecule.  
        - **logP (Hydrophobicity):** Indicates lipid solubility; affects absorption.  
        - **Topological Polar Surface Area (TPSA):** Predicts permeability & solubility.  
        - **H-bond Donors:** Number of hydrogen bond donors in the molecule.  
        - **H-bond Acceptors:** Number of hydrogen bond acceptors in the molecule.  
        - **Rotatable Bonds:** Determines molecule flexibility; impacts oral bioavailability.
        - **Lipinski Violations:** Rules for drug-likeness (≤1 violation is preferred).
        - **Lipinski Rule of Five Pass:** Whether the molecule meets Lipinski’s criteria.    
        - **Water Solubility (LogS):** Predicts solubility; lower LogS = better solubility.  
        - **Synthetic Accessibility Score:** Estimates ease of synthesis (lower is better).
        - **Bioavailability Score:** Probability of good oral bioavailability.
        - **Blood-Brain Barrier (BBB) Permeability:** Predicts CNS drug potential.
    --- 
    - ### **💊Drug-Likeness & CYP Inhibition:**
        - **Lipinski Rule of Five Pass:** Evaluates overall drug-likeness.  
        - **Bioavailability Score:** Assesses potential for oral absorption.  
        - **Blood-Brain Barrier (BBB) Permeability:** Predicts CNS drug capability.
        - **CYP Inhibition:** Predicts whether molecule will inhibit CYP1A2, CYP2C9, CYP2C19, CYP2D6, and CYP34.
    --- 
    - ### **⚡How to Use This App**
        1. Enter drug name or its SMILES representation
        2. Click **Analyze** to get the ADME properties.
    """
    )
    user_ch=st.selectbox("Do you want to enter Name or SMILES representation",options=userchoice)
    if user_ch=="Name":
        drug_name= st.text_input("Enter molecule's name")
        def get_smiles(drug_name):
            try:
                compound = pcp.get_compounds(drug_name, 'name')[0]
                return compound.isomeric_smiles
            except:
                return 0
        smiles_ip=get_smiles(drug_name)
       
    elif user_ch=="SMILES":
        smiles_ip=st.text_input("Enter molecule's SMILES representation in capital letters")

    if st.button("Analyze"):
        if smiles_ip:
            result = predict_adme(smiles_ip)
            st.subheader("Molecular Structure")
            mol_img = draw_molecule(smiles_ip)
            if mol_img:
                st.image(mol_img, caption="Generated from SMILES", use_container_width=False)
            else:
                st.warning("Invalid SMILES provided. Please enter a valid SMILES string.")
            st.subheader("ADME Properties")
            df = pd.DataFrame(result, columns=["Property", "Value","Interpretation"])
            st.table(df)
            lipinski_pass = result[7][1]  # Extracts the boolean value (True/False)
            bioavailability_score = result[10][1]  # Extracts the score (1 or 0)
            bbb_permeability = result[11][1]  # Extracts the boolean value (True/False)

        # Additional Summary
       
            if lipinski_pass:
                st.success("✅ This molecule **passes** Lipinski's Rule of Five (drug-like).")
            else:
                st.warning("⚠️ This molecule **violates** Lipinski's Rule of Five.")

            if bioavailability_score:
                st.success("✅ This molecule **meets** Veber's bioavailability criteria.")
            else:
                st.warning("⚠️ This molecule **may have poor oral bioavailability**.")

            if bbb_permeability:
                st.success("✅ This molecule has **good potential** for Blood-Brain Barrier (BBB) permeability.")
            else:
                st.warning("⚠️ This molecule **may have limited** BBB permeability.")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            try:
                model_cyp.load_state_dict(torch.load('CYP1A2_model_2.pth', map_location=device))
                model_cyp.to(device)
                model_cyp.eval()
                predicted_1A2 = predict_cyp(model_cyp, smiles_ip, 150)  # Make sure smiles_ip is on device if tensor
            except Exception as e:
                st.error(f"Error predicting CYP1A2: {str(e)}")
                predicted_1A2 = 0

            try:
                model_cyp.load_state_dict(torch.load('CYP2C9_model_2.pth', map_location=device))
                model_cyp.to(device)
                model_cyp.eval()
                predicted_2C9 = predict_cyp(model_cyp, smiles_ip, 150)
            except Exception as e:
                st.error(f"Error predicting CYP2C9: {str(e)}")
                predicted_2C9 = 0

            try:
                model_cyp.load_state_dict(torch.load('CYP2C19_model_2.pth', map_location=device))
                model_cyp.to(device)
                model_cyp.eval()
                predicted_2C19 = predict_cyp(model_cyp, smiles_ip, 150)
            except Exception as e:
                st.error(f"Error predicting CYP2C19: {str(e)}")
                predicted_2C19 = 0

            try:
                model_cyp.load_state_dict(torch.load('CYP2D6_model_2.pth', map_location=device))
                model_cyp.to(device)
                model_cyp.eval()
                predicted_2D6 = predict_cyp(model_cyp, smiles_ip, 150)
            except Exception as e:
                st.error(f"Error predicting CYP1A2: {str(e)}")
                predicted_2D6 = 0

            try:
                model_cyp.load_state_dict(torch.load('CYP3A4_model_2.pth', map_location=device))
                model_cyp.to(device)
                model_cyp.eval()
                predicted_3A4 = predict_cyp(model_cyp, smiles_ip, 150)
            except Exception as e:
                st.error(f"Error predicting CYP3A4: {str(e)}")
                predicted_3A4 = 0

            if predicted_1A2 == 1:
                st.warning("⚠️ This molecule is an inhibitor of CYP1A2.")
            else:
                st.success("✅ This molecule is not an inhibitor of CYP1A2.")
            if predicted_2C9 == 1:
                st.warning("⚠️ This molecule is an inhibitor of CYP2C9.")
            else:
                st.success("✅ This molecule is not an inhibitor of CYP2C9.")
            if predicted_2C19 == 1:
                st.warning("⚠️ This molecule is an inhibitor of CYP2C19.")
            else:
                st.success("✅ This molecule is not an inhibitor of CYP2C19.")
            if predicted_2D6 == 1:
                st.warning("⚠️ This molecule is an inhibitor of CYP2D6.")
            else:
                st.success("✅ This molecule is not an inhibitor of CYP2D6.")
            if predicted_3A4 == 1:
                st.warning("⚠️ This molecule is an inhibitor of CYP3A4.")
            else:
                st.success("✅ This molecule is not an inhibitor of CYP3A4.")
            
            # Generate AI Summary 
            # === SINGLE AI CALL ===
            st.subheader("Molecular Summary")
            with st.spinner("Generating AI insights... Please wait."):
                cyp_predictions = {
                    'CYP1A2': predicted_1A2,
                    'CYP2C9': predicted_2C9,
                    'CYP2C19': predicted_2C19,
                    'CYP2D6': predicted_2D6,
                    'CYP3A4': predicted_3A4
                }
                ai_result = generate_comprehensive_ai_insight(smiles_ip, result, cyp_predictions)

                # start
                # --- Scientific Summary ---
                st.markdown("### 🧪 Scientific Summary")
                st.markdown(ai_result["narrative_summary"])

                # --- ADME Inference (Formatted) ---
                st.markdown("### 📊 ADME Inference")
                inference = ai_result["adme_inference"]

                # Handle case where inference is still a stringified dict (fallback)
                if isinstance(inference, str):
                    try:
                        import json
                        inference = json.loads(inference)
                    except:
                        # If parsing fails, just show as plain text
                        st.markdown(inference)
                        inference = None

                if isinstance(inference, dict):
                    for section, bullets in inference.items():
                        st.markdown(f"**{section}:**")
                        if isinstance(bullets, list):
                            for bullet in bullets:
                                st.markdown(f"- {bullet}")
                        else:
                            st.markdown(str(bullets))
                        st.markdown("")  # Add spacing
                #end
            # ADD THIS BLOCK HERE ⬇️
            st.session_state.report_data['adme'] = {
                'smiles': smiles_ip,
                'drug_name': drug_name if user_ch == "Name" else "SMILES Input",
                'properties': result,
                'mol_img': mol_img,
                'lipinski_pass': lipinski_pass,
                'bioavailability_score': bioavailability_score,
                'bbb_permeability': bbb_permeability,
                'cyp_predictions': cyp_predictions,
                'ai_summary': ai_result["narrative_summary"],  # Store AI summary
                'inference': ai_result["adme_inference"],  # Will be updated after AI inference
                'timestamp': datetime.datetime.now()
            }

if tab=="⚛️ Binding Affinity":
    generate_report_data()  # ADD THIS LINE
    st.subheader("Binding Affinity")
    st.markdown("""

- ### **🧬Protein Groups**

    **Nuclear Receptors (Hormone-Responsive)**
    - **PPARD** – Peroxisome Proliferator-Activated Receptor Delta.
    - **PPARG** – Peroxisome Proliferator-Activated Receptor Gamma.
    - **AR** – Androgen receptor
    - **ESR1** – Estrogen Receptor Alpha
    - **NR3C1** – Glucocorticoid receptor

    **Kinases & Cell Signaling**
    - **ABL1** – Abelson Murine Leukemia Viral Oncogene Homolog 1
    - **JAK2** – Janus Kinase 2
    - **AKT1** – AKT Serine/Threonine Kinase 1
    - **MAPK1** – Mitogen-Activated Protein Kinase 1
    - **PLK1** – Polo-Like Kinase 1
    - **EGFR** – Epidermal Growth Factor Receptor
                
    **Enzymes and Metabolic Targets**
    - **HMGCR** – 3-Hydroxy-3-Methylglutaryl-CoA Reductase
    - **PTGS2** – Prostaglandin-Endoperoxide Synthase 2 (COX-2)
    - **CYP3A4** – Cytochrome P450 3A4
    - **DPP4** – Dipeptidyl Peptidase 4

    **Neurotransmitter and Neurological Targets**
    - **ADRB1** – Beta-1 Adrenergic Receptor
    - **ADORA2A** – Adenosine A2A Receptor
    - **DRD2** – Dopamine Receptor D2
    - **ACHE** – Acetylcholinesterase
    - **BACE1** – Beta-Site Amyloid Precursor Protein-Cleaving Enzyme 1

    **Cancer Therapeutic Targets**
    - **CASP3** – Caspase-3
    - **PARP1** – Poly (ADP-Ribose) Polymerase 1
    - **ROCK1** – Rho-Associated Protein Kinase 1
    - **KDR** – Kinase Insert Domain Receptor (VEGFR-2)
---

- ### **📊How to interpret results**
        
    **Binding Affinity (ΔG)**
    Binding affinity (**ΔG**, in kcal/mol) represents how strongly a drug binds to a target protein.

    | **Binding Affinity (ΔG)** | **Binding Strength** |
    |--------------------------|--------------------|
    | **ΔG ≤ -10 kcal/mol** | Very strong binding |
    | **-10 < ΔG ≤ -8 kcal/mol** | Strong binding |
    | **-8 < ΔG ≤ -6 kcal/mol** | Moderate binding |
    | **-6 < ΔG ≤ -4 kcal/mol** | Weak binding |
    | **ΔG > -4 kcal/mol** | Very weak/no binding |

    **Equilibrium Constant (K)**
    The equilibrium constant (**K** in µM) represents the ratio of bound and unbound states of a drug-protein interaction:

    | **Kd Value (µM)** | **Biological Interpretation** |
    |------------------|------------------------------------|
    | **< 0.001 µM** | Likely irreversible inhibition or very strong target modulation |
    | **0.001 – 0.1 µM** | Highly potent modulator, strong effect at low concentrations |
    | **0.1 – 1 µM** | Effective modulation, commonly seen in drug candidates |
    | **1 – 10 µM** | Moderate effect, may require optimization for potency |
    | **10 – 100 µM** | Weak modulation, may be non-specific or require high doses |
    | **> 100 µM** | Very weak or no modulation, likely not effective |

---                
- ### **⚡How to Use This App**
    1. Enter drug name or its SMILES representation
    2. Select a protein group from the dropdown menu.
    3. Choose a target protein within the selected group.
    4. Click **Predict** to get the predicted interaction strength.

""")

    user_ch_2=st.selectbox("Do you want to enter Name or SMILES representation of a molecule",options=userchoice2)
    if user_ch_2=="Name":
        drug_name_2= st.text_input("Enter name of the molecule")
        def get_smiles(drug_name_2):
            try:
                compound = pcp.get_compounds(drug_name_2, 'name')[0]
                return compound.isomeric_smiles
            except:
                return 0
        smiles_ip_2=get_smiles(drug_name_2)
       
    elif user_ch_2=="SMILES":
        smiles_ip_2=st.text_input("Enter molecule's SMILES representation in capital")
    
    if smiles_ip_2:
        group = st.selectbox("Select Protein Group", list(protein_groups.keys()))
        if group:
            protein = st.selectbox("Select Target Protein", protein_groups[group])

    if smiles_ip_2:
        if st.button("Predict"):
            model.load_state_dict(torch.load(f"{protein}_model_best.pth", map_location=torch.device('cpu')))
            model.eval()
            predicted_y = predict_y(model, smiles_ip_2)
            integer_value = round(predicted_y,2)
            del_g=str(integer_value)
            K=integer_value*4184
            K=K/(298*8.314)
            K=np.exp(K)
            K=K * 10**6
            K=round(K,4)
            eqb=str(K)
            L=" µM"
            cal= " kcal/mol"
            del_g=del_g+cal
            eqb=eqb+L
            st.write("Binding affinity of your drug molecule with selected protein is")
            st.write(del_g)
            st.write("Amount of drug in micromolar needed to modulate selected protein is")
            st.write(eqb)
            concentration = np.logspace(-2, 2, 1000)
            inhibition = dose_response(concentration, K)
            st.title('Dose-Response Curve')
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.semilogx(concentration, inhibition, 'g-', linewidth=2, label=f'Compound Kd = {K:.3f} µM)')
            y_K = dose_response(K, K)
            ax.plot(K, y_K, 'mo', markersize=8, label='_nolegend_')
            ax.axvline(K, color='m', linestyle='--', linewidth=1)
            ax.set_xlabel('Drug Concentration (µM)')
            ax.set_ylabel('% Inhibition')
            ax.set_title('Dose-Response Curve')
            ax.legend(loc='best')
            ax.grid(True)
            ax.set_xlim([min(concentration), max(concentration)])
            ax.set_ylim([0, 100])
            st.pyplot(fig)
            # Generate AI inference
            with st.spinner("Generating expert interpretation..."):
                binding_inference = generate_binding_affinity_insight(
                    smiles=smiles_ip_2,
                    protein=protein,
                    binding_affinity_kcal=integer_value,  # raw float, not string with units
                    kd_value=K  # raw Kd in µM
                )

            # Display inference
            st.markdown("### 🔍 AI-Generated Interpretation")
            st.markdown(binding_inference)

            # Save to session state
            st.session_state.report_data['binding'] = {
                'smiles': smiles_ip_2,
                'drug_name': drug_name_2 if user_ch_2 == "Name" else "SMILES Input",
                'protein_group': group,
                'protein': protein,
                'binding_affinity': del_g,
                'equilibrium_constant': eqb,
                'kd_value': K,
                'inference': binding_inference,
                'timestamp': datetime.datetime.now()
            }

if tab=="🧪 Drug Synergy":
    generate_report_data()  # ADD THIS LINE
    st.subheader("Drug Synergy Prediction")
    st.markdown("""  

- ### **🧪Cell Lines**  

    **Cell Lines & Cancer Types**  
    - **MCF7** – Breast cancer (ER⁺, Luminal A).  
    - **A549** – Lung adenocarcinoma (NSCLC).  
    - **HCT116** – Colorectal carcinoma.  
    - **DU145** – Prostate cancer (androgen-independent).  
    - **K562** – Chronic myelogenous leukemia (CML).  
    - **OVCAR3** – Ovarian adenocarcinoma.  
    - **SNB75** – Glioblastoma (brain tumor).   
    - **786-O** – Renal cell carcinoma (RCC, kidney cancer).  

---  

- ### **📊 How to Interpret Bliss Synergy Score**  

    The **Bliss Synergy Score** quantifies the interaction between two drugs compared to their expected independent effects.  

    | **Bliss Synergy Score** | **Interpretation** |  
    |-------------------------|--------------------|  
    | **> 1**  | Synergistic – Drugs work significantly better together than expected. |   
    | **1 to -1**  | Additive – Drugs work as expected without interaction. |  
    | **< -1**  | Antagonistic – Drug combination reduces effectiveness. |  

---  

- ### **⚡ How to Use This App**  

    1. Select two drugs from the input list or enter SMILES.  
    2. Choose a cancer cell line from the dropdown menu.  
    3. Click **Predict** to calculate the **Bliss Synergy Score**.    

""")
    user_ch_3=st.selectbox("Do you want to enter the Name or SMILES representation",options=userchoice3)
    if user_ch_3=="Name":
        drug_name_sy1= st.text_input("Enter name of the first molecule")
        drug_name_sy2=st.text_input("Enter name of the second molecule")
        def get_smiles(drug_name):
            try:
                compound = pcp.get_compounds(drug_name, 'name')[0]
                return compound.isomeric_smiles
            except:
                return 0
        smiles_sy_1=get_smiles(drug_name_sy1)
        smiles_sy_2=get_smiles(drug_name_sy2)
       
    elif user_ch_3=="SMILES":
        smiles_sy_1=st.text_input("Enter first molecule's SMILES representation in capital letters")
        smiles_sy_2=st.text_input("Enter second molecule's SMILES representation in capital letters")
        
    if smiles_sy_1 and smiles_sy_2:
        cell=st.selectbox("Choose your desired cell line",options=cell_line)
    if smiles_sy_1 and smiles_sy_2:
        if st.button("Predict Synergy"):
            model_sy.load_state_dict(torch.load(f"{cell}_MODEL.pth", map_location = torch.device("cpu")))
            model_sy.eval()
            predicted_sy = predict_sy(model_sy, smiles_sy_1, smiles_sy_2)
            synergy_value = round(predicted_sy,2)
            bliss_score=str(synergy_value)
            st.write("Bliss score of the two molecules with desired cell line is")
            st.write(bliss_score)
            # Generate AI inference
            with st.spinner("Generating expert synergy interpretation..."):
                synergy_inference = generate_synergy_insight(
                    smiles1=smiles_sy_1,
                    smiles2=smiles_sy_2,
                    cell_line=cell,
                    bliss_score=synergy_value  # use raw float, not string
                )

            # Display inference
            st.markdown("### 🔍 AI-Generated Interpretation")
            st.markdown(synergy_inference)

            # Save to session state
            st.session_state.report_data['synergy'] = {
                'smiles_1': smiles_sy_1,
                'smiles_2': smiles_sy_2,
                'drug_name_1': drug_name_sy1 if user_ch_3 == "Name" else "SMILES Input 1",
                'drug_name_2': drug_name_sy2 if user_ch_3 == "Name" else "SMILES Input 2",
                'cell_line': cell,
                'bliss_score': bliss_score,
                'synergy_value': synergy_value,
                'inference': synergy_inference,
                'timestamp': datetime.datetime.now()
            }
           
if tab == "📊 Generate Report":
    st.subheader("📄 Professional PDF Report")
    
    # Ensure session state is initialized
    generate_report_data()
    
    # Check if any analysis data exists
    has_data = any([
        st.session_state.report_data['adme'],
        st.session_state.report_data['binding'],
        st.session_state.report_data['synergy']
    ])
    
    if not has_data:
        st.warning("⚠️ No analysis data found. Please run analyses in other tabs first.")
    else:
        st.success("✅ Report data is ready. Click below to download a professional PDF.")
        
        # ONE BUTTON ONLY
        try:
            pdf_data = render_pdf_report()
            st.download_button(
                label="📄 Download Professional PDF Report",
                data=pdf_data,
                file_name=f"DrugXplorer_Report_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                type="primary"
            )
        except Exception as e:
            st.error(f"❌ Failed to generate PDF: {str(e)}")
            st.info("Please ensure all required dependencies are installed and try again.")

#bg image



# Simple Animated Dark Blue Gradient Background
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(-45deg, #0c1629, #1e3a5f, #0f2847, #1a365d, #2563eb, #1e40af);
        background-size: 600% 600%;
        animation: gradientShift 12s ease-in-out infinite;
    }
    
    @keyframes gradientShift {
        0% {
            background-position: 0% 50%;
        }
        25% {
            background-position: 100% 0%;
        }
        50% {
            background-position: 100% 100%;
        }
        75% {
            background-position: 0% 100%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
</style>
""", unsafe_allow_html=True)
