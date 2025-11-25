"""
Updated smiles_utils.py with higher resolution and clearer chemical structure generation
"""

import logging
import base64
from io import BytesIO
from typing import Optional

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

def smiles_to_image_base64(smiles: str, width: int = 400, height: int = 400) -> Optional[str]:
    """
    Convert a SMILES string to a high-resolution base64-encoded PNG image.
    
    Args:
        smiles: SMILES string representing the molecule
        width: Width of the generated image in pixels (default increased to 400)
        height: Height of the generated image in pixels (default increased to 400)
        
    Returns:
        str: Base64-encoded PNG image data URI, or None if conversion fails
    """
    if not RDKIT_AVAILABLE:
        logger.warning("RDKit not available - cannot convert SMILES to structure")
        return None
    
    if not smiles or str(smiles).strip() in ['', 'nan', 'None']:
        logger.debug("Empty or invalid SMILES string provided")
        return None
    
    try:
        # Parse the SMILES string
        mol = Chem.MolFromSmiles(str(smiles))
        
        if mol is None:
            logger.warning(f"Could not parse SMILES: {smiles}")
            return None
        
        # Use the newer high-quality drawer for better resolution
        try:
            # Try to use the modern high-resolution drawer
            drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
            
            # Set drawing options for better quality
            opts = drawer.drawOptions()
            opts.bondLineWidth = 2  # Thicker bonds for clarity
            opts.atomLabelFontSize = 24  # Larger font size
            opts.minFontSize = 20  # Minimum font size
            opts.maxFontSize = 40  # Maximum font size
            
            # Generate coordinates if not present
            if not mol.GetNumConformers():
                from rdkit.Chem import rdDepictor
                rdDepictor.Compute2DCoords(mol)
            
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            
            # Get the image data
            img_data = drawer.GetDrawingText()
            
            # Convert to base64
            img_b64 = base64.b64encode(img_data).decode()
            return f"data:image/png;base64,{img_b64}"
            
        except (ImportError, AttributeError):
            # Fall back to the standard method if Cairo drawer not available
            logger.debug("Cairo drawer not available, using standard drawer")
            
            # Generate the image with higher DPI equivalent
            img = Draw.MolToImage(
                mol, 
                size=(width, height),
                fitImage=True,
                options=None
            )
            
            # Convert to base64
            buffer = BytesIO()
            # Save with higher quality
            img.save(buffer, format='PNG', optimize=False, quality=95)
            img_data = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_data}"
            
    except Exception as e:
        logger.error(f"Error converting SMILES '{smiles}' to image: {str(e)}")
        return None

def is_valid_smiles(smiles: str) -> bool:
    """
    Check if a SMILES string is valid.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not RDKIT_AVAILABLE:
        return False
    
    if not smiles or str(smiles).strip() in ['', 'nan', 'None']:
        return False
    
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return mol is not None
    except Exception:
        return False

def get_rdkit_availability() -> bool:
    """
    Check if RDKit is available for SMILES processing.
    
    Returns:
        bool: True if RDKit is available, False otherwise
    """
    return RDKIT_AVAILABLE