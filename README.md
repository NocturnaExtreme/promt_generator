# 🪄 Magic Prompt Generator for AI Image Generation

A powerful AI-powered tool for generating enhanced prompts for various image generation models like Flux, Midjourney, SDXL, and Stable Diffusion. The application combines multiple AI services to create detailed, artistic prompts and can analyze existing images to generate similar prompts.

## ✨ Features

- **Multi-Model Support**: Generate prompts optimized for Flux, Midjourney, SDXL, and Stable Diffusion 1.5
- **Artistic Style Integration**: Choose from 50+ artistic styles across 5 categories
- **AI Enhancement**: Uses GPT-4 and MagicPrompt-Stable-Diffusion for prompt enhancement
- **Image Analysis**: Upload images to generate similar prompts using GPT-4 Vision
- **Multi-Language Support**: Automatically translates input text to English
- **Customizable Detail Levels**: Basic, Detailed, and Ultra prompt complexity options
- **Artist Database**: Incorporates famous artists' styles into prompts

## 🎨 Supported Style Categories

- **Classic Art Styles**: Realism, Impressionism, Baroque, Romanticism, etc.
- **Modern & Avant-Garde**: Cubism, Surrealism, Abstract, Expressionism, etc.
- **Contemporary Styles**: Pop Art, Street Art, Digital Art, Hyperrealism, etc.
- **Futuristic & Fantasy**: Cyberpunk, Synthwave, Steampunk, Post-apocalypse, etc.
- **Stylized & Digital**: Anime, Manga, Pixel Art, Vector Art, etc.

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for better performance)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/magic-prompt-generator.git
cd magic-prompt-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The application will launch in your browser at `http://localhost:7860`

## 📋 Usage

### Text Prompt Generation

1. Enter your base idea in Russian or English
2. Select desired artistic styles from the categories
3. Choose your target AI model (Flux, Midjourney, SDXL, or Stable Diffusion 1.5)
4. Set detail level (Basic, Detailed, or Ultra)
5. Click "Generate Prompts" to get 5 enhanced variations

### Image Analysis

1. Upload an image in the "Image Analysis" tab
2. Select artistic styles to incorporate (optional)
3. Choose target AI model
4. Click "Analyze Image" to generate a similar prompt

## 🛠 Technical Details

### AI Models Used

- **MagicPrompt-Stable-Diffusion**: For prompt enhancement
- **GPT-4/GPT-4 Vision**: For advanced prompt enhancement and image analysis
- **Google Translate API**: For automatic translation

### Architecture

The application is built with:
- **Gradio**: Web interface framework
- **Transformers**: Hugging Face model integration
- **PyTorch**: Machine learning backend
- **G4F**: GPT API client
- **PIL**: Image processing

## 🎯 Model-Specific Features

### Flux
- Optimized for natural language prompts
- Balanced tag usage
- Clean, direct prompt format

### Midjourney
- Formatted with `/imagine prompt:` prefix
- Enhanced descriptive language
- Artistic direction emphasis

### SDXL
- Rich detail incorporation
- Multiple style combinations
- Extended tag support

### Stable Diffusion 1.5
- Classic SD prompt structure
- Focused descriptive elements
- Compatibility optimized

## 📊 Examples

### Input
```
мистический лес в сумерках
```

### Output (Flux, Detailed)
```
Prompt 1:
A mystical twilight forest with ancient towering trees, soft ethereal mist floating between dark silhouettes, magical blue and purple ambient lighting, mysterious atmosphere, photorealistic, cinematic composition

Prompt 2:
Enchanted forest at dusk, golden hour lighting filtering through dense canopy, mystical fog, dark fantasy atmosphere, in the style of Caspar David Friedrich, romantic landscape painting
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file (optional):
```
DEVICE=cuda  # or cpu
HF_MAX_LENGTH=100
TEMPERATURE=0.8
```

### Customization

You can customize styles and artists by editing the `STYLE_CATEGORIES` and `STYLE_ARTISTS` dictionaries in `app.py`.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for the MagicPrompt model
- OpenAI for GPT models
- Gradio team for the excellent UI framework
- All the artists whose styles inspire this tool

## 🐛 Known Issues

- G4F service may occasionally be unavailable
- Large images may take longer to analyze
- Some artistic styles work better with specific models

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the troubleshooting section in the wiki
- Review the existing discussions

## 🔄 Updates

This project is actively maintained. Check the releases page for the latest updates and features.
  