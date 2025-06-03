import re
import requests
import gradio as gr
import torch
import random
import time
import base64
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple # Added Tuple for type hinting
from transformers import AutoTokenizer, AutoModelForCausalLM
from g4f.client import Client # As per user's reverted code

# --- Constants and Configuration ---
MODEL_NAME = "Gustavosta/MagicPrompt-Stable-Diffusion"

# Translation settings
TRANSLATE_API_URL = "https://translate.googleapis.com/translate_a/single"
TRANSLATE_TIMEOUT_SECONDS = 10
TRANSLATE_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Prompt generation parameters
PROMPT_WEIGHT_MIN = 1.1
PROMPT_WEIGHT_MAX = 1.8
NUM_PROMPTS_TO_GENERATE = 5
ARTIST_POOL_SIZE = 5
NUM_PROMPTS_WITH_ARTISTS = 3
ARTIST_PICK_MIN = 1
ARTIST_PICK_MAX = 3

# Hugging Face Model Generation Parameters
HF_MAX_LENGTH = 100 # This is used as total max length in user's original generate_with_huggingface
HF_TEMPERATURE = 0.8
HF_TOP_K = 50
HF_TOP_P = 0.95
HF_MIN_ADDITION_LEN = 4  # Corrected non-breaking space

# Device configuration for PyTorch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize G4F client
g4f_client_instance: Optional[Client] = None # Explicitly define type
g4f_available = False
try:
  g4f_client_instance = Client()
  g4f_available = True
  print("G4F client initialized successfully.")
except Exception as e: # Catch generic Exception as in user's code
  g4f_available = False
  print(f"Error initializing G4F client: {e}")
  print("Proceeding without G4F. Prompt enhancement with GPT will not be available.")

# --- Global Variables for Model and Tokenizer ---
hf_tokenizer: Optional[AutoTokenizer] = None
hf_model: Optional[AutoModelForCausalLM] = None

# --- Model Loading ---
try:
  print(f"Attempting to load Hugging Face model '{MODEL_NAME}'...")
  hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  # Set pad token to eos token to avoid attention mask warning
  if hf_tokenizer.pad_token is None or hf_tokenizer.pad_token_id == hf_tokenizer.eos_token_id: # Check pad_token_id against eos_token_id
    hf_tokenizer.pad_token = hf_tokenizer.eos_token # Set the token string
    print("Pad token was None or same as EOS. Set pad token to EOS token string.")
  
  hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
  hf_model.eval()
  hf_model.to(DEVICE)
  print(f"Model '{MODEL_NAME}' loaded successfully on {DEVICE}.")
except Exception as e:
  print(f"Error loading Hugging Face model '{MODEL_NAME}': {e}")
  print("Proceeding without the Hugging Face model. Prompt generation will rely solely on templates and user input.")
  hf_tokenizer = None
  hf_model = None

# --- Prompt Template Categories ---
STYLE_CATEGORIES = {
  "Classic Art Styles": {
    "Realism": ["photorealistic", "ultra realistic", "hyperrealism", "realistic", "naturalistic"],
    "Naturalism": ["scientific accuracy", "detailed realism", "natural light"],
    "Academism": ["academic style", "classical composition", "strict rules"],
    "Classicism": ["classicism", "harmony", "order", "antique motifs"],
    "Baroque": ["baroque", "dramatic", "ornate", "dynamic composition"],
    "Rococo": ["rococo", "lightness", "elegance", "pastel colors"],
    "Romanticism": ["romanticism", "emotional", "heroism", "freedom"],
    "Impressionism": ["impressionism", "fleeting impression", "light and color", "brush strokes"],
    "Post-Impressionism": ["post-impressionism", "expressive colors", "bold forms"]
  },
  "Modern & Avant-Garde Styles": {
    "Expressionism": ["expressionism", "emotional distortion", "bold colors"],
    "Cubism": ["cubism", "geometric", "fragmented forms"],
    "Surrealism": ["surrealism", "dreamlike", "fantasy", "irrational"],
    "Abstract": ["abstract", "non-representational", "colors and shapes"],
    "Futurism": ["futurism", "speed", "technology", "dynamic"],
    "Constructivism": ["constructivism", "structure", "material", "form"],
    "Dadaism": ["dadaism", "absurd", "protest", "anti-art"],
    "Suprematism": ["suprematism", "geometric abstraction", "minimalism"],
    "Minimalism": ["minimalism", "simplicity", "clean", "lack of details"],
    "Art Nouveau": ["art nouveau", "flowing lines", "natural motifs"],
    "Art Deco": ["art deco", "geometry", "luxury", "industrial style"]
  },
  "Contemporary Styles": {
    "Pop Art": ["pop art", "mass culture", "comics", "bright colors"],
    "Neo-Expressionism": ["neo-expressionism", "rough", "energetic"],
    "Conceptualism": ["conceptual art", "idea over form"],
    "Hyperrealism": ["hyperrealism", "photo-like", "extreme detail"],
    "Lowbrow": ["lowbrow", "street art", "humorous"],
    "Street Art": ["street art", "graffiti", "political messages"],
    "Digital Art": ["digital art", "2D", "3D", "collage"],
    "AI Art": ["AI-generated", "neural aesthetics", "machine creativity"]
  },
  "Futuristic & Fantasy": {
    "Cyberpunk": ["cyberpunk", "neon lights", "futuristic city", "dark neon", "techwear", "rainy night"],
    "Neon": ["neon", "bright neon lights", "glowing", "dark background", "vaporwave"],
    "Synthwave": ["synthwave", "80s retro", "neon colors", "grid", "sunset", "futuristic"],
    "Retrofuturism": ["retrofuturism", "vintage sci-fi", "chrome", "futuristic cars"],
    "Fantasy": ["fantasy", "mythical creatures", "magic", "medieval", "epic landscapes"],
    "Post-apocalypse": ["post-apocalyptic", "ruins", "desolate", "wasteland", "survival"],
    "Cybergoth": ["cybergoth", "dark futuristic", "industrial", "neon accents"],
    "Steampunk": ["steampunk", "brass", "steam machines", "Victorian"],
    "Dark Fantasy": ["dark fantasy", "gothic", "grim"],
    "Solarpunk": ["solarpunk", "eco-futurism", "bright green tech"]
  },
  "Stylized & Digital": {
    "Anime": ["anime style", "japanese anime", "manga", "studio ghibli"],
    "Manga": ["manga", "black and white", "line art"],
    "Caricature": ["caricature", "exaggerated features", "humor"],
    "Comic Art": ["comic art", "panels", "stylized characters"],
    "Pixel Art": ["pixel art", "retro graphics", "pixelated"],
    "Line Art": ["line art", "contour drawing"],
    "Vector Art": ["vector art", "sharp shapes", "scalable"],
    "Ink Drawing": ["ink drawing", "black ink", "shading"],
    "Cel Shading": ["cel shading", "flat colors", "toon style"],
    "My Custom Style": ["my unique tag1", "my unique tag2", "special technique", "custom motif"]
  }
}

# Полный словарь художников по стилям
STYLE_ARTISTS = {
  "Realism": ["Gustave Courbet", "Jean-François Millet", "Ilya Repin", "Thomas Eakins", "Edward Hopper"],
  "Naturalism": ["Jules Bastien-Lepage", "Winslow Homer", "Jean-Baptiste-Camille Corot", "Rosa Bonheur"],
  "Academism": ["William-Adolphe Bouguereau", "Alexandre Cabanel", "Jean-Léon Gérôme", "Thomas Couture"],
  "Classicism": ["Nicolas Poussin", "Jacques-Louis David", "Antoine-Jean Gros", "Jean-Auguste-Dominique Ingres"],
  "Baroque": ["Caravaggio", "Peter Paul Rubens", "Rembrandt", "Gian Lorenzo Bernini", "Diego Velázquez"],
  "Rococo": ["François Boucher", "Jean-Honoré Fragonard", "Thomas Gainsborough", "Antoine Watteau"],
  "Romanticism": ["Eugène Delacroix", "Francisco Goya", "Caspar David Friedrich", "J.M.W. Turner", "John Constable"],
  "Impressionism": ["Claude Monet", "Pierre-Auguste Renoir", "Edgar Degas", "Camille Pissarro", "Édouard Manet"],
  "Post-Impressionism": ["Vincent van Gogh", "Paul Cézanne", "Paul Gauguin", "Georges Seurat", "Henri de Toulouse-Lautrec"],
  "Expressionism": ["Edvard Munch", "Egon Schiele", "Franz Marc", "Wassily Kandinsky", "Ernst Ludwig Kirchner"],
  "Cubism": ["Pablo Picasso", "Georges Braque", "Juan Gris", "Fernand Léger", "Robert Delaunay"],
  "Surrealism": ["Salvador Dalí", "René Magritte", "Max Ernst", "André Breton", "Yves Tanguy"],
  "Abstract": ["Wassily Kandinsky", "Piet Mondrian", "Kazimir Malevich", "Mark Rothko", "Jackson Pollock"],
  "Futurism": ["Umberto Boccioni", "Giacomo Balla", "Carlo Carrà", "Gino Severini", "Luigi Russolo"],
  "Constructivism": ["Vladimir Tatlin", "El Lissitzky", "Alexander Rodchenko", "Naum Gabo", "Lyubov Popova"],
  "Dadaism": ["Marcel Duchamp", "Max Ernst", "Hannah Höch", "Tristan Tzara"],
  "Suprematism": ["Kazimir Malevich", "El Lissitzky", "Ilya Chashnik"],
  "Minimalism": ["Donald Judd", "Agnes Martin", "Frank Stella", "Sol LeWitt"],
  "Art Nouveau": ["Gustav Klimt", "Alphonse Mucha", "Antoni Gaudí", "Hector Guimard"],
  "Art Deco": ["Tamara de Lempicka", "Erté", "Jean Dupas", "René Lalique"],
  "Pop Art": ["Andy Warhol", "Roy Lichtenstein", "Keith Haring", "Claes Oldenburg", "Robert Rauschenberg"],
  "Neo-Expressionism": ["Jean-Michel Basquiat", "Anselm Kiefer", "Georg Baselitz"],
  "Conceptualism": ["Sol LeWitt", "Joseph Kosuth", "Yoko Ono", "Bruce Nauman"],
  "Hyperrealism": ["Chuck Close", "Ralph Goings", "Roberto Bernardi", "Taner Ceylan"],
  "Lowbrow": ["Robert Williams", "Gary Baseman", "Todd Schorr", "Mark Ryden"],
  "Street Art": ["Banksy", "Shepard Fairey", "Jean-Michel Basquiat", "Blek le Rat", "Swoon"],
  "Digital Art": ["Beeple", "Ross Tran", "Android Jones", "James White"],
  "AI Art": ["Refik Anadol", "Mario Klingemann", "Sougwen Chung"],
  "Cyberpunk": ["Syd Mead", "Josan Gonzalez", "H. R. Giger", "Beeple"],
  "Neon": ["James White", "Beeple", "Travis Pitts"],
  "Synthwave": ["Patrick Nagel", "James White", "Mitch Marner"],
  "Retrofuturism": ["Chris Foss", "Michael Whelan", "Syd Mead"],
  "Fantasy": ["Brom", "Frank Frazetta", "Julie Bell", "Alan Lee", "John Howe"],
  "Post-apocalypse": ["Simon Stalenhag", "Jakub Rozalski"],
  "Cybergoth": ["Beeple", "Josan Gonzalez"],
  "Steampunk": ["Katsuhiro Otomo", "Hayao Miyazaki"],
  "Dark Fantasy": ["Zdzisław Beksiński", "H. R. Giger"],
  "Solarpunk": ["Timothy O'Brien", "Toby Allen"],
  "Anime": ["Hayao Miyazaki", "Makoto Shinkai", "Satoshi Kon", "Mamoru Hosoda"],
  "Manga": ["Osamu Tezuka", "Akira Toriyama", "CLAMP", "Naoki Urasawa"],
  "Caricature": ["Al Hirschfeld", "Mort Drucker", "David Levine"],
  "Comic Art": ["Jack Kirby", "Jim Lee", "Frank Miller"],
  "Pixel Art": ["Paul Robertson", "Octavi Navarro"],
  "Line Art": ["Egon Schiele", "Aubrey Beardsley"],
  "Vector Art": ["Malika Favre", "Charis Tsevis"],
  "Ink Drawing": ["Hokusai", "Albrecht Dürer"],
  "Cel Shading": ["Mamoru Hosoda", "Sunao Katabuchi"],
  "My Custom Style": ["YourArtist1", "YourArtist2"]
}

# --- AI Model Templates ---
AI_MODEL_TEMPLATES = {
  "Flux": {"prompt_format": "{prompt}", "tags_multiplier": 1.0, "system_prompt": "Enhance this image generation prompt for Flux model: {prompt}"},
  "Midjourney": {"prompt_format": "/imagine prompt: {prompt}", "tags_multiplier": 1.2, "system_prompt": "Enhance this image generation prompt for Midjourney. Create a detailed and vivid description with artistic direction: {prompt}"},
  "Stable Diffusion 1.5": {"prompt_format": "{prompt}", "tags_multiplier": 1.0, "system_prompt": "Enhance this image generation prompt for Stable Diffusion 1.5. Focus on clear descriptive elements: {prompt}"},
  "SDXL": {"prompt_format": "{prompt}", "tags_multiplier": 1.5, "system_prompt": "Enhance this image generation prompt for SDXL model. Include rich details, composition, lighting, and atmosphere: {prompt}"}
}

# --- Support functions ---
def get_artists_for_style(style: str) -> List[str]:
  return STYLE_ARTISTS.get(style, [])

def translate_to_english(text: str) -> str:
  if not text.strip():
    return ""
  if re.match(r'^[a-zA-Z0-9\s.,!?;:"\'\-_\(\)]*$', text):
    return text
  try:
    params = {'client': 'gtx', 'sl': 'auto', 'tl': 'en', 'dt': 't', 'q': text}
    headers = {'User-Agent': TRANSLATE_USER_AGENT}
    response = requests.get(TRANSLATE_API_URL, params=params, headers=headers, timeout=TRANSLATE_TIMEOUT_SECONDS)
    response.raise_for_status()
    translation = response.json()[0][0][0]
    return translation
  except requests.exceptions.RequestException as e:
    print(f"Translation API request error: {e}")
    return text
  except Exception as e:
    print(f"Translation error: {e}")
    return text

def enhance_prompt_with_gpt(prompt: str, model_type: str = "Flux") -> str:
  if not g4f_available or not g4f_client_instance:
    return prompt
  try:
    system_message = "You are an expert prompt engineer for AI image generation models. Your task is to enhance and improve the user's prompt to make it more detailed and effective for image generation. Focus on clarity, descriptive elements, and aesthetic details. DO NOT add any explanations or conversation, ONLY return the improved prompt."
    model_template = AI_MODEL_TEMPLATES.get(model_type, AI_MODEL_TEMPLATES["Flux"])
    system_prompt_formatted = model_template["system_prompt"].format(prompt=prompt)
    response = g4f_client_instance.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": system_prompt_formatted}
      ],
      web_search=False
    )
    enhanced_prompt = response.choices[0].message.content.strip()
    if len(enhanced_prompt) < len(prompt) * 0.8:
      return prompt
    return enhanced_prompt
  except Exception as e:
    print(f"GPT enhancement error: {e}")
    return prompt

def analyze_image_with_gpt(image: Image.Image, model_type: str = "Flux", selected_styles: List[str] = None) -> str:
  if not g4f_available or not g4f_client_instance:
    return "GPT сервис недоступен для анализа изображений."
  try:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Create system message with style information
    style_instruction = ""
    if selected_styles:
      style_list = ", ".join(selected_styles)
      style_instruction = f" Pay special attention to incorporating these artistic styles in your analysis: {style_list}."
    
    system_message = f"""You are an expert in analyzing images and creating detailed prompts for the {model_type} AI image generation model. 
    When given an image, carefully analyze its content, style, lighting, composition, and details.
    Then, create a detailed prompt that could generate a similar image.{style_instruction}
    Focus only on what's visible in the image. Be specific about:
    1. Main subject(s)
    2. Style and artistic influence
    3. Colors and lighting
    4. Composition and perspective
    5. Important details and textures
    Return ONLY the prompt, without any explanations or conversation."""
    
    response = g4f_client_instance.chat.completions.create(
      model="gpt-4o", 
      messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": [
          {"type": "text", "text": f"Analyze this image and create a detailed prompt for {model_type}."},
          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
        ]}
      ],
      web_search=False
    )
    prompt_text = response.choices[0].message.content.strip()
    
    # Add selected styles as tags if available
    if selected_styles:
      style_tags = []
      style_artists = []
      
      for style in selected_styles:
        # Find style tags
        for category_dict in STYLE_CATEGORIES.values():
          if style in category_dict:
            style_tags.extend(category_dict[style][:2])  # Add first 2 tags from each style
            break
        
        # Find style artists
        artists = get_artists_for_style(style)
        if artists:
          style_artists.extend(artists[:1])  # Add first artist from each style
      
      if style_tags:
        prompt_text += ", " + ", ".join(list(set(style_tags)))
      if style_artists:
        prompt_text += ", in the style of " + ", ".join(list(set(style_artists)))
    
    model_template = AI_MODEL_TEMPLATES.get(model_type, AI_MODEL_TEMPLATES["Flux"])
    if model_type == "Midjourney" and not prompt_text.startswith("/imagine prompt:"):
      prompt_text = model_template["prompt_format"].format(prompt=prompt_text)
    return prompt_text
  except Exception as e:
    print(f"GPT image analysis error: {e}")
    return f"Ошибка при анализе изображения: {str(e)}"

def generate_with_huggingface(input_text: str) -> str:
  if not hf_model or not hf_tokenizer:
    return input_text
  try:
    input_ids = hf_tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)
    attention_mask = torch.ones_like(input_ids)
    pad_token_id_to_use = hf_tokenizer.pad_token_id if hf_tokenizer.pad_token_id is not None else hf_tokenizer.eos_token_id
    if pad_token_id_to_use is None:
        print("Warning: pad_token_id is None. Using EOS token ID for padding.")
        pad_token_id_to_use = hf_tokenizer.eos_token_id # Fallback if still None
        if pad_token_id_to_use is None: # Should absolutely not happen with a valid tokenizer
             print("Critical Error: EOS token ID is also None. Cannot generate.")
             return input_text

    with torch.no_grad():
      output = hf_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=HF_MAX_LENGTH,
        temperature=HF_TEMPERATURE,
        top_k=HF_TOP_K,
        top_p=HF_TOP_P,
        do_sample=True,
        pad_token_id=pad_token_id_to_use
      )
    generated_text = hf_tokenizer.decode(output[0], skip_special_tokens=True)
    if len(generated_text) <= len(input_text) + HF_MIN_ADDITION_LEN:
      return input_text
    return generated_text
  except Exception as e:
    print(f"HuggingFace generation error: {e}")
    return input_text

# --- Функция генерации промптов с учетом detail_level ---
def generate_prompts(base_prompt: str, selected_styles: List[str], model_type: str, detail_level: str) -> str:
    base_prompt = translate_to_english(base_prompt.strip())
    if not base_prompt:
        base_prompt = "image"

    model_template = AI_MODEL_TEMPLATES.get(model_type, AI_MODEL_TEMPLATES["Flux"])
    tags_multiplier_from_model = model_template["tags_multiplier"]
    results = []

    if model_type == "Flux":
        base_styles_to_use = 1
        base_tags_per_style = 1
    elif model_type == "Midjourney":
        base_styles_to_use = 2
        base_tags_per_style = 2
    elif model_type == "Stable Diffusion 1.5":
        base_styles_to_use = 2
        base_tags_per_style = 2
    else:  # SDXL
        base_styles_to_use = 3
        base_tags_per_style = 3

    if detail_level == "Basic":
        num_styles_to_use = max(1, int(base_styles_to_use * 0.5))
        num_tags_per_style = max(1, int(base_tags_per_style * 0.5 * tags_multiplier_from_model))
    elif detail_level == "Detailed":
        num_styles_to_use = base_styles_to_use
        num_tags_per_style = max(1, int(base_tags_per_style * 1.0 * tags_multiplier_from_model))
    else:  # Ultra
        num_styles_to_use = int(base_styles_to_use * 1.5)
        num_tags_per_style = max(1, int(base_tags_per_style * 1.5 * tags_multiplier_from_model))

    if selected_styles:
        num_styles_to_use = min(num_styles_to_use, len(selected_styles))
    else:
        num_styles_to_use = 0

    if not selected_styles:
        for i in range(NUM_PROMPTS_TO_GENERATE):
            prompt = base_prompt
            if g4f_available and i < 2:
                enhanced = enhance_prompt_with_gpt(prompt, model_type)
                if model_type == "Midjourney" and not enhanced.startswith("/imagine prompt:"):
                    enhanced = model_template["prompt_format"].format(prompt=enhanced)
                results.append(enhanced)
            elif hf_model and hf_tokenizer:
                enhanced = generate_with_huggingface(prompt)
                if model_type == "Midjourney" and not enhanced.startswith("/imagine prompt:"):
                    enhanced = model_template["prompt_format"].format(prompt=enhanced)
                results.append(enhanced)
            else:
                if model_type == "Midjourney":
                    prompt = model_template["prompt_format"].format(prompt=prompt)
                results.append(prompt)
            if g4f_available:
                time.sleep(0.5)
        return "\n\n".join([f"Prompt {i+1}:\n{prompt}" for i, prompt in enumerate(results)])

    for i in range(NUM_PROMPTS_TO_GENERATE):
        prompt = base_prompt
        current_styles_for_prompt = []
        if selected_styles and num_styles_to_use > 0:
            current_styles_for_prompt = random.sample(selected_styles, num_styles_to_use)

        added_tags_list = []
        added_artists_list = []

        for style_item in current_styles_for_prompt:
            style_specific_tags = []
            for category_dict in STYLE_CATEGORIES.values():
                if style_item in category_dict:
                    style_specific_tags = category_dict[style_item]
                    break
            if style_specific_tags:
                sample_k_tags = min(num_tags_per_style, len(style_specific_tags))
                if sample_k_tags > 0:
                    selected_style_tags = random.sample(style_specific_tags, sample_k_tags)
                    added_tags_list.extend(selected_style_tags)

            style_specific_artists = get_artists_for_style(style_item)
            if style_specific_artists and i < NUM_PROMPTS_WITH_ARTISTS:
                sample_k_artists = min(ARTIST_PICK_MAX, len(style_specific_artists))
                if sample_k_artists > 0:
                    min_artists_to_pick = min(ARTIST_PICK_MIN, sample_k_artists)
                    if min_artists_to_pick <= sample_k_artists:
                        num_artists_to_select = random.randint(min_artists_to_pick, sample_k_artists)
                        if num_artists_to_select > 0:
                            selected_style_artists = random.sample(style_specific_artists, num_artists_to_select)
                            added_artists_list.extend(selected_style_artists)

        if added_tags_list:
            prompt += ", " + ", ".join(list(set(added_tags_list)))
        if added_artists_list:
            prompt += ", in the style of " + ", ".join(list(set(added_artists_list)))

        if g4f_available and i < 2:
            prompt = enhance_prompt_with_gpt(prompt, model_type)
        elif hf_model and hf_tokenizer:
            prompt = generate_with_huggingface(prompt)

        if model_type == "Midjourney" and not prompt.startswith("/imagine prompt:"):
            prompt = model_template["prompt_format"].format(prompt=prompt)
        results.append(prompt)
        if g4f_available:
            time.sleep(0.5)
    return "\n\n".join([f"Prompt {i+1}:\n{prompt}" for i, prompt in enumerate(results)])

# --- Статус индикаторы ---
def get_status_indicator(is_active: bool, service_name: str) -> str:
    color = "green" if is_active else "red"
    return f"""
    <div style="display:inline-flex; align-items:center; margin-right:15px">
        <div style="width:10px; height:10px; border-radius:50%; background-color:{color}; margin-right:5px"></div>
        <span>{service_name}</span>
    </div>
    """

# --- Создание интерфейса Gradio ---
def create_gradio_interface():
    all_checkbox_group_components_for_inputs: List[gr.CheckboxGroup] = []
    all_checkbox_group_components_for_analysis: List[gr.CheckboxGroup] = []

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🪄 Magic Prompt Generator for Image Synthesis")

        hf_status_html = get_status_indicator(hf_model is not None and hf_tokenizer is not None, "MagicPrompt")
        g4f_status_html = get_status_indicator(g4f_available, "GPT4")
        status_html = f"""<div style="display:flex; margin-bottom:10px">{hf_status_html}{g4f_status_html}</div>"""
        gr.HTML(status_html)
        gr.Markdown("Введите базовую идею, выберите художественные стили, или загрузите изображение для анализа!")

        with gr.Tabs():
            with gr.TabItem("Текстовый промпт"):
                with gr.Row():
                    with gr.Column(scale=2):
                        base_prompt_input = gr.Textbox(
                            label="Введите вашу базовую идею (русский или английский будет переведен на английский)",
                            lines=3,
                            placeholder="например, мистический лес в сумерках, кот-космонавт, футуристический городской пейзаж"
                        )
                        detail_level_radio = gr.Radio(
                            ["Basic", "Detailed", "Ultra"],
                            value="Detailed",
                            label="Уровень детализации и разнообразия добавляемых тегов"
                        )
                        model_type_radio = gr.Radio(
                            list(AI_MODEL_TEMPLATES.keys()),
                            value="Flux",
                            label="Модель нейросети (для генерации)"
                        )
                    with gr.Column(scale=1, min_width=180):
                        generate_button = gr.Button("✨ Сгенерировать промпты ✨", variant="primary")
                        gr.Markdown("*Совет: Комбинируйте разные стили для уникальных результатов!*")

                gr.Markdown("--- \n### Выберите художественные стили и влияния")

                style_checkbox_groups: Dict[str, gr.CheckboxGroup] = {}
                num_columns = 3
                category_names = list(STYLE_CATEGORIES.keys())
                for i in range(0, len(category_names), num_columns):
                    with gr.Row():
                        for j in range(num_columns):
                            if i + j < len(category_names):
                                category_name = category_names[i + j]
                                styles_in_category = STYLE_CATEGORIES[category_name]
                                with gr.Column():
                                    gr.Markdown(f"**{category_name}**")
                                    style_checkbox = gr.CheckboxGroup(
                                        choices=list(styles_in_category.keys()), label=""
                                    )
                                    style_checkbox_groups[category_name] = style_checkbox
                                    all_checkbox_group_components_for_inputs.append(style_checkbox)

                generated_prompts_textbox = gr.Textbox(
                    label="Сгенерированные промпты (скопируйте их в ваш генератор изображений)",
                    lines=10,
                    placeholder="Ваши сгенерированные промпты появятся здесь..."
                )

            with gr.TabItem("Анализ изображения"):
                with gr.Row():
                    with gr.Column(scale=1):
                        uploaded_image = gr.Image(label="Загрузите изображение для анализа", type="pil")
                        analyze_model_type_radio = gr.Radio(
                            list(AI_MODEL_TEMPLATES.keys()),
                            value="Flux",
                            label="Выберите модель для анализа"
                        )
                        analyze_button = gr.Button("🔍 Анализировать изображение", variant="primary")
                        gr.Markdown("Загрузите изображение и GPT-4 создаст промпт на его основе")
                    with gr.Column(scale=1):
                        analyzed_prompt_textbox = gr.Textbox(
                            label="Результат анализа (промпт для похожего изображения)",
                            lines=10,
                            placeholder="Результаты анализа появятся здесь..."
                        )

                gr.Markdown("--- \n### Выберите художественные стили для анализа изображения")
                for i in range(0, len(category_names), 3):
                    with gr.Row():
                        for j in range(3):
                            if i + j < len(category_names):
                                category_name = category_names[i + j]
                                styles_in_category = STYLE_CATEGORIES[category_name]
                                with gr.Column():
                                    gr.Markdown(f"**{category_name}**")
                                    style_checkbox = gr.CheckboxGroup(
                                        choices=list(styles_in_category.keys()), label=""
                                    )
                                    all_checkbox_group_components_for_analysis.append(style_checkbox)

        def get_all_selected_styles():
            selected_styles = []
            for checkbox_component in all_checkbox_group_components_for_inputs:
                selected_styles.extend(checkbox_component.value or [])
            return selected_styles

        def get_all_selected_styles_for_analysis():
            selected_styles = []
            for checkbox_component in all_checkbox_group_components_for_analysis:
                selected_styles.extend(checkbox_component.value or [])
            return selected_styles

        def on_generate_button_click(base_prompt, detail_level, model_type):
            selected_styles = get_all_selected_styles()
            return generate_prompts(base_prompt, selected_styles, model_type, detail_level)

        def on_analyze_button_click(image, model_type):
            selected_styles = get_all_selected_styles_for_analysis()
            if image is None:
                return "Пожалуйста, загрузите изображение."
            return analyze_image_with_gpt(image, model_type, selected_styles)

        generate_button.click(
            on_generate_button_click,
            inputs=[base_prompt_input, detail_level_radio, model_type_radio],
            outputs=generated_prompts_textbox,
        )

        analyze_button.click(
            on_analyze_button_click,
            inputs=[uploaded_image, analyze_model_type_radio],
            outputs=analyzed_prompt_textbox,
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
