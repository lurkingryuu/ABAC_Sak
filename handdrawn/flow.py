import io
import os
import json
from typing import Literal, Union
from google import genai
from pydantic import BaseModel, Field, TypeAdapter
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
# Replace with your actual API Key
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemini-2.5-flash"

# --- 1. DEFINE PYDANTIC MODELS (The Schema) ---


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
    low_index: int = Field(description="The 1-based integer index of the start of the range (e.g. if bar starts at label '1.1', this is 1).")
    high_index: int = Field(description="The 1-based integer index of the end of the range (inclusive).")
    x_axis_min: float = Field(description="Start value of the visual X-axis (e.g. 0).")
    x_axis_max: float = Field(description="End value of the visual X-axis (e.g. max index).")
    y_axis_max: float = Field(description="The maximum Y-value shown on the chart's Y-axis.")
    visual_peak_height: float = Field(description="The Y-value of the bars.")


# The master union type
DistributionResponse = Union[NormalDist, PoissonDist, UniformDist]
adapter = TypeAdapter(DistributionResponse)

# --- 2. DETERMINISTIC PLOTTING ENGINE ---


class Plotter:
    @staticmethod
    def render(params: DistributionResponse) -> Image.Image:
        """Renders the plot based on the validated Pydantic object."""
        plt.figure(figsize=(6, 4))
        x_range = np.linspace(params.x_axis_min, params.x_axis_max, 500)
        
        try:
            if isinstance(params, NormalDist):
                # 1. Calculate standard PDF
                y_std = stats.norm.pdf(x_range, params.mu, params.sigma)
                
                # 2. Determine Scaling Factor
                std_peak = np.max(y_std) if np.max(y_std) > 0 else 1.0
                scale_factor = params.visual_peak_height / std_peak
                y_scaled = y_std * scale_factor
                
                plt.plot(x_range, y_scaled, label=rf'Normal($\mu={params.mu:.1f}, \sigma={params.sigma:.1f}$)', color='blue')
                plt.fill_between(x_range, y_scaled, alpha=0.1, color='blue')
                
                # Force the Y-axis to include the visual peak
                plt.ylim(0, max(params.y_axis_max, params.visual_peak_height * 1.1))
                
            elif isinstance(params, PoissonDist):
                x_discrete = np.arange(int(params.x_axis_min), int(params.x_axis_max) + 1)
                y_std = stats.poisson.pmf(x_discrete, params.lam)
                
                std_peak = np.max(y_std) if np.max(y_std) > 0 else 1.0
                scale_factor = params.visual_peak_height / std_peak
                y_scaled = y_std * scale_factor
                
                plt.stem(x_discrete, y_scaled, label=rf'Poisson($\lambda={params.lam:.1f}$)', basefmt=" ")
                plt.ylim(0, max(params.y_axis_max, params.visual_peak_height * 1.1))
                
            elif isinstance(params, UniformDist):
                # Discrete Uniform
                x_discrete = np.arange(params.low_index, params.high_index + 1)
                
                if len(x_discrete) > 0:
                    # Uniform height means all bars are at visual_peak_height
                    y_scaled = np.full_like(x_discrete, params.visual_peak_height, dtype=float)
                else:
                    y_scaled = []
                
                plt.bar(x_discrete, y_scaled, width=1.0, label=f'Uniform([{params.low_index}, {params.high_index}])', color='green', alpha=0.3, edgecolor='green')
                
                plt.xticks(np.arange(int(params.x_axis_min), int(params.x_axis_max) + 1))
                plt.ylim(0, max(params.y_axis_max, params.visual_peak_height * 1.1))

            plt.title(f"Extracted: {params.dist_type.title()}")
            plt.legend()
            plt.grid(True, alpha=0.3, linestyle='--')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            buf.seek(0)
            return Image.open(buf)

        except Exception as e:
            print(f"Plotting Error: {e}")
            return None

# --- 3. THE REFINEMENT SYSTEM (Modern Syntax) ---



def stitch_images(img_orig, img_generated):
    """Side-by-side comparison helper."""
    img_generated = img_generated.resize((int(img_generated.width * img_orig.height / img_generated.height), img_orig.height))
    total_width = img_orig.width + img_generated.width
    combined = Image.new('RGB', (total_width, img_orig.height))
    combined.paste(img_orig, (0, 0))
    combined.paste(img_generated, (img_orig.width, 0))
    return combined


def to_abac_config(params: DistributionResponse) -> dict:
    """Converts the extracted Pydantic model to the ABAC JSON configuration format."""
    if isinstance(params, NormalDist):
        return {
            "distribution": "N",
            "mean": params.mu,
            "variance": params.sigma ** 2
        }
    elif isinstance(params, PoissonDist):
        return {
            "distribution": "P",
            "lambda": params.lam
        }
    elif isinstance(params, UniformDist):
        return {
            "distribution": "U",
            "low": params.low_index - 1,  # Convert 1-based index to 0-based
            "high": params.high_index     # 0-based exclusive (equivalent to 1-based inclusive for the count)
        }
    return {}


def analyze_and_refine(image_path, iterations=2):
    original_img = Image.open(image_path)
    
    # --- PASS 1: INITIAL EXTRACTION ---
    prompt_extract = """
    Analyze this image. Determine if it is a Normal, Poisson, or Uniform distribution.
    
    - For Normal Distribution:
      - Read the X-axis labels directly to determine 'x_axis_min', 'x_axis_max'.
      - Identify the peak location on this X-axis for 'mu'.
      - Estimate 'sigma' from the width.
      - Be aware of the edges of the distribution to prevent thinning out the distribution towards the extremes.
      - Identify the peak location on the Y-axis for 'y_axis_max'.
      - Extract 'visual_peak_height' as the Y-value of the peak of the curve.
      
    - For Poisson or Uniform (Discrete):
      - The X-axis often represents discrete items (1-based indices).
      - Labels like "1", "2", "3" or "SA-1.1", "SA-1.2" correspond to items 1, 2, 3...
      - For Uniform: extract 'low_index' and 'high_index' as integers (inclusive range).
      - Identify the peak location on the Y-axis for 'y_axis_max'.
      - Extract 'visual_peak_height' as the Y-value of the highest bar.
    """
    
    print("--- Pass 1: Extracting Parameters ---")
    
    # 1. The clean call using the new SDK pattern
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[prompt_extract, original_img],
        config={
            "response_mime_type": "application/json",
            "response_schema": DistributionResponse, # Pass the class directly (SDK handles schema)
        },
    )
    
    # 2. Validate using Pydantic built-in
    try:
        # Note: In the new SDK, parsed response might be accessible directly or via text
        # We rely on Pydantic to parse the JSON string for safety
        current_params = adapter.validate_json(response.text)
        print(f"Initial Guess: {current_params}")
    except Exception as e:
        print(f"Extraction failed: {e}")
        return

    current_plot = Plotter.render(current_params)
    
    # --- PASS 2: VISUAL FEEDBACK LOOP ---
    for i in range(iterations):
        if not current_plot: break
        print(f"--- Iteration {i+1}: Refining ---")
        
        comparison = stitch_images(original_img, current_plot)
        
        prompt_refine = f"""
        Refine the parameters. Left is Original, Right is Approximation.
        Current Parameters: {current_params.model_dump_json()}
        
        CRITICAL INSTRUCTION: 
        Look at the edges of the curve (at x_axis_min and x_axis_max).
        - If the Original sketch is still high at the edges, but your Approximation has dropped to zero, you MUST INCREASE SIGMA.
        - Do not sacrifice the width of the base just to make the peak sharper. 
        - It is better to have a wider curve that covers the edges than a narrow one that fits the peak perfectly.
        
        Output the CORRECTED object.
        """
        
        response_refine = client.models.generate_content(
            model=MODEL_ID,
            contents=[prompt_refine, comparison],
            config={
                "response_mime_type": "application/json",
                "response_schema": DistributionResponse, 
            },
        )
        
        try:
            current_params = adapter.validate_json(response_refine.text)
            print(f"Refined Params: {current_params}")
            current_plot = Plotter.render(current_params)
        except Exception as e:
            print(f"Refinement failed: {e}")
            break

    if current_plot:
        current_plot.save("final_output.png")
        print("\nSuccess! Saved to final_output.png")
        
    # --- OUTPUT FOR ABAC APP ---
    abac_config = to_abac_config(current_params)
    print("\n--- ABAC Configuration JSON ---")
    print(json.dumps(abac_config, indent=4))


if __name__ == "__main__":
    analyze_and_refine("normal.png", iterations=0)
