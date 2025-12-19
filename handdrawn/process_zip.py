# ruff: noqa
# flake8: noqa

"""
ABAC Configuration Generator from Hand-drawn Distribution Images

This script processes a zip file containing hand-drawn distribution images
for Subject Attributes (SA-*), Object Attributes (OA-*), and Environment 
Attributes (EA-*), extracts the distribution parameters using an LLM, and
generates the ABAC configuration JSON.

Input: A zip file with images named like:
  - SA-1.png, SA-2.png, ... (Subject Attributes)
  - OA-1.png, OA-2.png, ... (Object Attributes)
  - EA-1.png, EA-2.png, ... (Environment Attributes)

Output:
  - config_output.json: The generated ABAC configuration
  - comparisons/: Folder with side-by-side original vs interpreted images
"""

import io
import os
import json
import zipfile
import argparse
from pathlib import Path
from typing import Literal, Union, Optional, Tuple, List, Dict
from dataclasses import dataclass, field

from google import genai
from pydantic import BaseModel, Field, TypeAdapter
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemini-2.5-flash"


# --- PYDANTIC MODELS FOR DISTRIBUTION EXTRACTION ---

class NormalDist(BaseModel):
    dist_type: Literal["normal"]
    mu: float = Field(description="The mean (peak location on X-axis).")
    sigma: float = Field(description="The standard deviation (width).")
    x_axis_min: float = Field(description="Start value of the visual X-axis.")
    x_axis_max: float = Field(description="End value of the visual X-axis.")
    y_axis_max: float = Field(description="The maximum Y-value shown on the chart's Y-axis.")
    visual_peak_height: float = Field(description="The Y-value of the highest point of the curve.")


class PoissonDist(BaseModel):
    dist_type: Literal["poisson"]
    lam: float = Field(description="Lambda (event rate) for the Poisson distribution.")
    x_axis_min: float = Field(description="Start value of the visual X-axis.")
    x_axis_max: float = Field(description="End value of the visual X-axis.")
    y_axis_max: float = Field(description="The maximum Y-value shown on the chart's Y-axis.")
    visual_peak_height: float = Field(description="The Y-value of the highest bar.")


class UniformDist(BaseModel):
    dist_type: Literal["uniform"]
    low_index: int = Field(description="The 1-based integer index of the start of the range.")
    high_index: int = Field(description="The 1-based integer index of the end of the range (inclusive).")
    x_axis_min: float = Field(description="Start value of the visual X-axis.")
    x_axis_max: float = Field(description="End value of the visual X-axis.")
    y_axis_max: float = Field(description="The maximum Y-value shown on the chart's Y-axis.")
    visual_peak_height: float = Field(description="The Y-value of the bars.")


DistributionResponse = Union[NormalDist, PoissonDist, UniformDist]
adapter = TypeAdapter(DistributionResponse)


# --- ATTRIBUTE METADATA EXTRACTION ---

class AttributeMetadata(BaseModel):
    """Metadata extracted from the image about the attribute it represents."""
    attribute_type: Literal["SA", "OA", "EA"] = Field(
        description="The type of attribute: SA (Subject), OA (Object), or EA (Environment)."
    )
    attribute_index: int = Field(
        description="The 1-based index of the attribute (e.g., 1 for SA-1, 2 for OA-2)."
    )
    num_values: int = Field(
        description=(
            "The number of possible values for this attribute (N). "
            "In these sketches, N is typically written as the RIGHTMOST X-axis label at the arrow tip "
            "(e.g., '... 25' at the far right means num_values=25). "
            "Do NOT return the count of written numbers; return the max X-axis endpoint label."
        )
    )
    evidence_text: Optional[str] = Field(
        default=None,
        description=(
            "Optional: the exact text you relied on (e.g., '25' near the x-axis arrow tip), "
            "or a short description like 'rightmost x-axis label'."
        ),
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Optional confidence in [0,1]. Use low confidence if the label is faint/unclear.",
    )


metadata_adapter = TypeAdapter(AttributeMetadata)


# --- DATA STRUCTURES ---

@dataclass
class ProcessedAttribute:
    """Holds all data for a processed attribute image."""
    attr_type: str  # "SA", "OA", "EA"
    index: int      # 1-based index
    num_values: int
    distribution_params: DistributionResponse
    abac_config: dict
    original_image: Image.Image
    interpreted_image: Image.Image
    comparison_image: Image.Image


@dataclass
class ABACConfig:
    """Aggregated ABAC configuration ready for JSON export."""
    subject_attributes_count: int = 0
    object_attributes_count: int = 0
    environment_attributes_count: int = 0
    subject_attributes_values: List[int] = field(default_factory=list)
    object_attributes_values: List[int] = field(default_factory=list)
    environment_attributes_values: List[int] = field(default_factory=list)
    subject_distributions: List[dict] = field(default_factory=list)
    object_distributions: List[dict] = field(default_factory=list)
    environment_distributions: List[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "subject_attributes_count": self.subject_attributes_count,
            "object_attributes_count": self.object_attributes_count,
            "environment_attributes_count": self.environment_attributes_count,
            "subject_attributes_values": self.subject_attributes_values,
            "object_attributes_values": self.object_attributes_values,
            "environment_attributes_values": self.environment_attributes_values,
            "subject_distributions": self.subject_distributions,
            "object_distributions": self.object_distributions,
            "environment_distributions": self.environment_distributions,
        }


# --- PLOTTING ENGINE ---

class Plotter:
    @staticmethod
    def render(params: DistributionResponse, title: str = None) -> Optional[Image.Image]:
        """Renders the plot based on the validated Pydantic object."""
        plt.figure(figsize=(6, 4))
        x_range = np.linspace(params.x_axis_min, params.x_axis_max, 500)
        
        try:
            if isinstance(params, NormalDist):
                y_std = stats.norm.pdf(x_range, params.mu, params.sigma)
                std_peak = np.max(y_std) if np.max(y_std) > 0 else 1.0
                scale_factor = params.visual_peak_height / std_peak
                y_scaled = y_std * scale_factor
                
                plt.plot(x_range, y_scaled, 
                        label=rf'Normal($\mu={params.mu:.1f}, \sigma={params.sigma:.1f}$)', 
                        color='#2563eb', linewidth=2)
                plt.fill_between(x_range, y_scaled, alpha=0.2, color='#2563eb')
                plt.ylim(0, max(params.y_axis_max, params.visual_peak_height * 1.1))
                
            elif isinstance(params, PoissonDist):
                x_discrete = np.arange(int(params.x_axis_min), int(params.x_axis_max) + 1)
                y_std = stats.poisson.pmf(x_discrete, params.lam)
                std_peak = np.max(y_std) if np.max(y_std) > 0 else 1.0
                scale_factor = params.visual_peak_height / std_peak
                y_scaled = y_std * scale_factor
                
                plt.stem(x_discrete, y_scaled, 
                        label=rf'Poisson($\lambda={params.lam:.1f}$)', 
                        basefmt=" ", linefmt='#dc2626', markerfmt='o')
                plt.ylim(0, max(params.y_axis_max, params.visual_peak_height * 1.1))
                
            elif isinstance(params, UniformDist):
                x_discrete = np.arange(params.low_index, params.high_index + 1)
                if len(x_discrete) > 0:
                    y_scaled = np.full_like(x_discrete, params.visual_peak_height, dtype=float)
                else:
                    y_scaled = []
                
                plt.bar(x_discrete, y_scaled, width=0.8, 
                       label=f'Uniform([{params.low_index}, {params.high_index}])', 
                       color='#16a34a', alpha=0.5, edgecolor='#16a34a', linewidth=1.5)
                plt.xticks(np.arange(int(params.x_axis_min), int(params.x_axis_max) + 1))
                plt.ylim(0, max(params.y_axis_max, params.visual_peak_height * 1.1))

            plot_title = title if title else f"Interpreted: {params.dist_type.title()}"
            plt.title(plot_title, fontsize=12, fontweight='bold')
            plt.xlabel("Values", fontsize=10)
            plt.ylabel("Probability Density", fontsize=10)
            plt.legend(loc='upper right', fontsize=9)
            plt.grid(True, alpha=0.3, linestyle='--')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
            plt.close()
            buf.seek(0)
            return Image.open(buf)

        except Exception as e:
            print(f"  [ERROR] Plotting failed: {e}")
            plt.close()
            return None


# --- IMAGE UTILITIES ---

def stitch_images(img_orig: Image.Image, img_generated: Image.Image, 
                  labels: Tuple[str, str] = ("Original", "Interpreted")) -> Image.Image:
    """Creates a side-by-side comparison with labels."""
    # Ensure both images are in RGB mode
    if img_orig.mode != 'RGB':
        img_orig = img_orig.convert('RGB')
    if img_generated.mode != 'RGB':
        img_generated = img_generated.convert('RGB')
    
    # Resize generated image to match original height
    scale = img_orig.height / img_generated.height
    new_width = int(img_generated.width * scale)
    img_generated = img_generated.resize((new_width, img_orig.height), Image.Resampling.LANCZOS)
    
    # Create combined image with padding for labels
    padding = 30
    total_width = img_orig.width + img_generated.width + 20  # 20px gap
    total_height = img_orig.height + padding
    
    combined = Image.new('RGB', (total_width, total_height), 'white')
    
    # Paste images
    combined.paste(img_orig, (0, padding))
    combined.paste(img_generated, (img_orig.width + 20, padding))
    
    return combined


def parse_attribute_from_filename(filename: str) -> Optional[Tuple[str, int]]:
    """
    Deprecated/disabled by design:
    Filenames are assumed to be random; attribute identity MUST be extracted from the image.
    Kept only for backward compatibility/testing.
    """
    return None


# --- CORE EXTRACTION LOGIC ---

def extract_metadata_from_image(image: Image.Image) -> Optional[AttributeMetadata]:
    """Uses LLM to extract attribute metadata from the image itself."""
    prompt = """
    Look at this distribution chart image carefully.

    Your primary task is to find the attribute identifier WRITTEN INSIDE THE IMAGE.
    It will appear as text like:
      - SA-1, SA-2, ... (Subject Attributes)
      - OA-1, OA-2, ... (Object Attributes)
      - EA-1, EA-2, ... (Environment Attributes)

    Requirements:
    - Do NOT use the filename; assume filenames are random.
    - The identifier must be read from visible text in the image (title, legend, axis labels, watermark, etc.).
    - attribute_index is the integer after the dash.

    Also extract num_values (IMPORTANT — common failure mode):
    - In these hand-drawn sketches, num_values is usually written as the RIGHTMOST X-axis label
      at the arrow tip (the maximum endpoint of the X-axis).
      Examples: if you see '... 20' at the far right, num_values=20. If you see '... 13', num_values=13.
    - Do NOT return the count of how many numbers are written on the page.
      Many sketches have two numbers (peak and right endpoint); that DOES NOT mean num_values=2.
    - Prefer the number closest to the x-axis arrow tip (far right). If multiple candidates, pick the largest
      number that is clearly an X-axis endpoint.
    - If the right-end label is unreadable, make your best estimate and set confidence low (<0.5).
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[prompt, image],
            config={
                "response_mime_type": "application/json",
                "response_schema": AttributeMetadata,
            },
        )
        return metadata_adapter.validate_json(response.text)
    except Exception as e:
        print(f"  [WARNING] Metadata extraction failed: {e}")
        return None


def extract_distribution_params(image: Image.Image, attr_label: str) -> Optional[DistributionResponse]:
    """Extracts distribution parameters from an image using LLM."""
    prompt = f"""
    Analyze this distribution chart for attribute {attr_label}.
    Determine if it shows a Normal, Poisson, or Uniform distribution.
    
    IMPORTANT GUIDELINES:
    
    - For Normal Distribution:
      - Read the X-axis labels to determine 'x_axis_min' and 'x_axis_max'.
      - The 'mu' (mean) is the X-value at the peak of the curve.
      - Estimate 'sigma' from the width (where the curve drops to ~60% of peak).
      - The 'visual_peak_height' is the Y-value at the peak.
      - The 'y_axis_max' is the maximum value shown on the Y-axis.
      
    - For Poisson Distribution:
      - Identify the 'lam' (lambda) parameter from where the distribution peaks.
      - Note the bars/stems pattern typical of discrete Poisson.
      
    - For Uniform Distribution:
      - All bars should have roughly equal height.
      - 'low_index' and 'high_index' are 1-based indices of the range.
      - Look at X-axis labels like "1", "2", "3" or "{attr_label}.1", "{attr_label}.2".
    
    Extract the precise parameters from the visual representation.
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[prompt, image],
            config={
                "response_mime_type": "application/json",
                "response_schema": DistributionResponse,
            },
        )
        return adapter.validate_json(response.text)
    except Exception as e:
        print(f"  [ERROR] Distribution extraction failed: {e}")
        return None


def refine_distribution_params(original_img: Image.Image, 
                               current_params: DistributionResponse,
                               iterations: int = 1) -> Tuple[DistributionResponse, Image.Image]:
    """Refines parameters through visual comparison iterations."""
    current_plot = Plotter.render(current_params)
    if not current_plot:
        return current_params, None
    
    for i in range(iterations):
        comparison = stitch_images(original_img, current_plot)
        
        prompt_refine = f"""
        Compare the two images. Left is the ORIGINAL hand-drawn distribution, 
        Right is the INTERPRETED mathematical distribution.
        
        Current Parameters: {current_params.model_dump_json()}
        
        REFINEMENT INSTRUCTIONS:
        - If the shapes don't match, adjust the parameters.
        - For Normal: Check if mu (peak location) and sigma (width) are correct.
        - For Poisson: Check if lambda gives the right peak location.
        - For Uniform: Check if the range [low_index, high_index] is correct.
        
        EDGE CHECK (Critical for Normal):
        - If the original curve is still high at the edges but interpreted drops to zero,
          INCREASE sigma to widen the distribution.
        
        Output the CORRECTED parameters.
        """
        
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[prompt_refine, comparison],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": DistributionResponse,
                },
            )
            current_params = adapter.validate_json(response.text)
            current_plot = Plotter.render(current_params)
            if not current_plot:
                break
        except Exception as e:
            print(f"  [WARNING] Refinement iteration {i+1} failed: {e}")
            break
    
    return current_params, current_plot


def to_abac_config(params: DistributionResponse) -> dict:
    """Converts extracted parameters to ABAC JSON configuration format."""
    if isinstance(params, NormalDist):
        return {
            "distribution": "N",
            "mean": round(params.mu, 2),
            "variance": round(params.sigma ** 2, 2)
        }
    elif isinstance(params, PoissonDist):
        return {
            "distribution": "P",
            "lambda": round(params.lam, 2)
        }
    elif isinstance(params, UniformDist):
        return {
            "distribution": "U"
            # Uniform doesn't need additional params in the ABAC format
        }
    return {}


# --- MAIN PROCESSING PIPELINE ---

def process_single_image(image: Image.Image, 
                         attr_type: str, 
                         attr_index: int,
                         num_values: int,
                         refinement_iterations: int = 1) -> Optional[ProcessedAttribute]:
    """Process a single distribution image and extract all parameters."""
    attr_label = f"{attr_type}-{attr_index}"
    print(f"  Processing {attr_label}...")
    
    # Step 1: Extract distribution parameters
    params = extract_distribution_params(image, attr_label)
    if not params:
        print(f"  [ERROR] Could not extract distribution for {attr_label}")
        return None
    
    print(f"    Initial: {params.dist_type} distribution")
    
    # Step 2: Refine through visual comparison
    if refinement_iterations > 0:
        params, interpreted_img = refine_distribution_params(image, params, refinement_iterations)
        print(f"    Refined: {params.model_dump_json()}")
    else:
        interpreted_img = Plotter.render(params, title=f"{attr_label}: {params.dist_type.title()}")
    
    if not interpreted_img:
        print(f"  [ERROR] Could not render interpretation for {attr_label}")
        return None
    
    # Step 3: Create comparison image
    comparison = stitch_images(image, interpreted_img)
    
    # Step 4: num_values comes from image metadata extraction (LLM).
    # Keep a conservative fallback if the caller passes something invalid.
    if not isinstance(num_values, int) or num_values <= 0:
        if isinstance(params, UniformDist):
            # Uniform sketches in this repo usually indicate N directly (e.g., "13" at the right end).
            # The model sometimes outputs 0-based indices; "max()" is a safe fallback here.
            num_values = max(1, int(max(params.low_index, params.high_index)))
        else:
            # For Normal/Poisson, the endpoint label is more like an axis max than a count of labels.
            # Fall back to rounding x_axis_max (better than counting written numbers).
            num_values = max(1, int(round(getattr(params, "x_axis_max", 1.0))))
    
    # Step 5: Convert to ABAC config format
    abac_config = to_abac_config(params)
    
    return ProcessedAttribute(
        attr_type=attr_type,
        index=attr_index,
        num_values=num_values,
        distribution_params=params,
        abac_config=abac_config,
        original_image=image,
        interpreted_image=interpreted_img,
        comparison_image=comparison
    )


def process_zip_file(zip_path: str, 
                     output_dir: str = "output",
                     refinement_iterations: int = 1) -> Optional[ABACConfig]:
    """
    Main entry point: Process a zip file of distribution images.
    
    Args:
        zip_path: Path to the zip file containing distribution images
        output_dir: Directory to save outputs (config JSON and comparison images)
        refinement_iterations: Number of refinement iterations per image
    
    Returns:
        ABACConfig object with all extracted parameters
    """
    print(f"\n{'='*60}")
    print("ABAC Configuration Generator")
    print(f"{'='*60}")
    print(f"Input: {zip_path}")
    print(f"Output: {output_dir}/")
    print(f"Refinement iterations: {refinement_iterations}")
    print(f"{'='*60}\n")
    
    # Create output directories
    output_path = Path(output_dir)
    comparisons_path = output_path / "comparisons"
    output_path.mkdir(parents=True, exist_ok=True)
    comparisons_path.mkdir(parents=True, exist_ok=True)
    
    # Collect processed attributes
    processed: Dict[str, List[ProcessedAttribute]] = {
        "SA": [],
        "OA": [],
        "EA": []
    }
    
    # Open and process zip file
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Get list of image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        image_files = [
            name for name in zf.namelist() 
            if Path(name).suffix.lower() in image_extensions
            and not name.startswith('__MACOSX')  # Skip macOS metadata
        ]
        
        print(f"Found {len(image_files)} image files in zip\n")
        
        for img_name in sorted(image_files):
            print(f"\n[{img_name}]")
            
            # Extract image
            with zf.open(img_name) as img_file:
                image = Image.open(io.BytesIO(img_file.read()))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            
            # Always use LLM to extract metadata from the image (filenames are random).
            print("  Extracting metadata from image (SA/OA/EA + index + num_values)...")
            metadata = extract_metadata_from_image(image)
            if metadata:
                attr_type = metadata.attribute_type
                attr_index = metadata.attribute_index
                num_values = metadata.num_values
                print(f"  Identified from image: {attr_type}-{attr_index} ({num_values} values)")
            else:
                print(f"  [SKIP] Could not identify attribute for {img_name}")
                continue
            
            # Process the image
            result = process_single_image(
                image=image,
                attr_type=attr_type,
                attr_index=attr_index,
                num_values=num_values,
                refinement_iterations=refinement_iterations
            )
            
            if result:
                processed[attr_type].append(result)
                
                # Save comparison image
                comparison_filename = f"{attr_type}-{attr_index}_comparison.png"
                result.comparison_image.save(comparisons_path / comparison_filename)
                print(f"    Saved: {comparison_filename}")
    
    # Sort by index
    for attr_type, _ in processed.items():
        processed[attr_type].sort(key=lambda x: x.index)

    # Ensure contiguous indices per attribute type (SA-1..SA-N etc.)
    # This guarantees the JSON arrays align to index order.
    def _require_contiguous(processed_list: List[ProcessedAttribute], kind: str) -> List[ProcessedAttribute]:
        if not processed_list:
            return []
        max_idx = max(p.index for p in processed_list)
        by_idx = {p.index: p for p in processed_list}
        missing = [i for i in range(1, max_idx + 1) if i not in by_idx]
        if missing:
            raise ValueError(f"Missing {kind} indices in inputs: {missing}. Provide images for every {kind}-k from 1..{max_idx}.")
        return [by_idx[i] for i in range(1, max_idx + 1)]

    processed["SA"] = _require_contiguous(processed["SA"], "SA")
    processed["OA"] = _require_contiguous(processed["OA"], "OA")
    processed["EA"] = _require_contiguous(processed["EA"], "EA")
    
    # Build ABAC configuration
    config = ABACConfig()
    
    # Subject Attributes
    config.subject_attributes_count = len(processed["SA"])
    config.subject_attributes_values = [p.num_values for p in processed["SA"]]
    config.subject_distributions = [p.abac_config for p in processed["SA"]]
    
    # Object Attributes
    config.object_attributes_count = len(processed["OA"])
    config.object_attributes_values = [p.num_values for p in processed["OA"]]
    config.object_distributions = [p.abac_config for p in processed["OA"]]
    
    # Environment Attributes
    config.environment_attributes_count = len(processed["EA"])
    config.environment_attributes_values = [p.num_values for p in processed["EA"]]
    config.environment_distributions = [p.abac_config for p in processed["EA"]]
    
    # Save configuration JSON
    config_path = output_path / "config_output.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=4)
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print("Processed:")
    print(f"  - Subject Attributes: {config.subject_attributes_count}")
    print(f"  - Object Attributes: {config.object_attributes_count}")
    print(f"  - Environment Attributes: {config.environment_attributes_count}")
    print(f"\nOutputs saved to: {output_path.absolute()}")
    print("  - config_output.json")
    print("  - comparisons/*.png")
    print(f"{'='*60}\n")
    
    return config


def process_folder(folder_path: str,
                   output_dir: str = "output", 
                   refinement_iterations: int = 1) -> Optional[ABACConfig]:
    """
    Alternative entry point: Process a folder of distribution images.
    
    Args:
        folder_path: Path to folder containing distribution images
        output_dir: Directory to save outputs
        refinement_iterations: Number of refinement iterations per image
    
    Returns:
        ABACConfig object with all extracted parameters
    """
    print(f"\n{'='*60}")
    print("ABAC Configuration Generator (Folder Mode)")
    print(f"{'='*60}")
    print(f"Input: {folder_path}")
    print(f"Output: {output_dir}/")
    print(f"{'='*60}\n")
    
    # Create output directories
    output_path = Path(output_dir)
    comparisons_path = output_path / "comparisons"
    output_path.mkdir(parents=True, exist_ok=True)
    comparisons_path.mkdir(parents=True, exist_ok=True)
    
    # Collect processed attributes
    processed: Dict[str, List[ProcessedAttribute]] = {
        "SA": [],
        "OA": [],
        "EA": []
    }
    
    # Find image files
    folder = Path(folder_path)
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    image_files = [
        f for f in folder.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    print(f"Found {len(image_files)} image files\n")
    
    for img_path in sorted(image_files):
        print(f"\n[{img_path.name}]")
        
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Always use LLM to extract metadata from the image (filenames are random).
        print("  Extracting metadata from image (SA/OA/EA + index + num_values)...")
        metadata = extract_metadata_from_image(image)
        if metadata:
            attr_type = metadata.attribute_type
            attr_index = metadata.attribute_index
            num_values = metadata.num_values
            print(f"  Identified from image: {attr_type}-{attr_index} ({num_values} values)")
        else:
            print("  [SKIP] Could not identify attribute")
            continue
        
        # Process the image
        result = process_single_image(
            image=image,
            attr_type=attr_type,
            attr_index=attr_index,
            num_values=num_values,
            refinement_iterations=refinement_iterations
        )
        
        if result:
            processed[attr_type].append(result)
            
            # Save comparison image
            comparison_filename = f"{attr_type}-{attr_index}_comparison.png"
            result.comparison_image.save(comparisons_path / comparison_filename)
            print(f"    Saved: {comparison_filename}")
    
    # Sort by index
    for attr_type in processed:
        processed[attr_type].sort(key=lambda x: x.index)

    # Ensure contiguous indices per attribute type (SA-1..SA-N etc.)
    def _require_contiguous(processed_list: List[ProcessedAttribute], kind: str) -> List[ProcessedAttribute]:
        if not processed_list:
            return []
        max_idx = max(p.index for p in processed_list)
        by_idx = {p.index: p for p in processed_list}
        missing = [i for i in range(1, max_idx + 1) if i not in by_idx]
        if missing:
            raise ValueError(f"Missing {kind} indices in inputs: {missing}. Provide images for every {kind}-k from 1..{max_idx}.")
        return [by_idx[i] for i in range(1, max_idx + 1)]

    processed["SA"] = _require_contiguous(processed["SA"], "SA")
    processed["OA"] = _require_contiguous(processed["OA"], "OA")
    processed["EA"] = _require_contiguous(processed["EA"], "EA")
    
    # Build ABAC configuration
    config = ABACConfig()
    
    config.subject_attributes_count = len(processed["SA"])
    config.subject_attributes_values = [p.num_values for p in processed["SA"]]
    config.subject_distributions = [p.abac_config for p in processed["SA"]]
    
    config.object_attributes_count = len(processed["OA"])
    config.object_attributes_values = [p.num_values for p in processed["OA"]]
    config.object_distributions = [p.abac_config for p in processed["OA"]]
    
    config.environment_attributes_count = len(processed["EA"])
    config.environment_attributes_values = [p.num_values for p in processed["EA"]]
    config.environment_distributions = [p.abac_config for p in processed["EA"]]
    
    # Save configuration JSON
    config_path = output_path / "config_output.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=4)
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  - Subject Attributes: {config.subject_attributes_count}")
    print(f"  - Object Attributes: {config.object_attributes_count}")  
    print(f"  - Environment Attributes: {config.environment_attributes_count}")
    print(f"\nOutputs: {output_path.absolute()}")
    print(f"{'='*60}\n")
    
    return config


# --- CLI ENTRY POINT ---

def main():
    parser = argparse.ArgumentParser(
        description="Generate ABAC configuration from hand-drawn distribution images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_zip.py distributions.zip
  python process_zip.py distributions.zip -o my_output -r 2
  python process_zip.py ./images_folder --folder -o results
        """
    )
    
    parser.add_argument(
        "input",
        help="Path to zip file or folder containing distribution images"
    )
    parser.add_argument(
        "-o", "--output",
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "-r", "--refinement",
        type=int,
        default=1,
        help="Number of refinement iterations per image (default: 1)"
    )
    parser.add_argument(
        "--folder",
        action="store_true",
        help="Treat input as a folder instead of a zip file"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}")
        return 1
    
    if args.folder or input_path.is_dir():
        config = process_folder(
            str(input_path),
            args.output,
            args.refinement
        )
    else:
        config = process_zip_file(
            str(input_path),
            args.output,
            args.refinement
        )
    
    if config:
        print("\n--- Generated Configuration ---")
        print(json.dumps(config.to_dict(), indent=4))
        return 0
    else:
        print("\nProcessing failed!")
        return 1


if __name__ == "__main__":
    exit(main())

