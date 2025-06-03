# Contributing to Magic Prompt Generator

Thank you for your interest in contributing to Magic Prompt Generator! This document provides guidelines and information for contributors.

## How to Contribute

### Reporting Issues

1. **Check existing issues** first to avoid duplicates
2. **Use the issue template** if available
3. **Provide detailed information**:
   - Steps to reproduce the problem
   - Expected vs actual behavior
   - Screenshots if applicable
   - System information (OS, Python version, etc.)

### Suggesting Features

1. **Open a feature request** issue
2. **Describe the feature** in detail
3. **Explain the use case** and benefits
4. **Consider implementation complexity**

### Code Contributions

#### Setting Up Development Environment

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/NocturnaExtreme/promt_generator.git
   cd magic-prompt-generator
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

#### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes**
3. **Test your changes** thoroughly
4. **Follow code style guidelines**

#### Code Style Guidelines

- **Use Python PEP 8** style guidelines
- **Add type hints** where appropriate
- **Write docstrings** for functions and classes
- **Keep functions small** and focused
- **Use meaningful variable names**
- **Add comments** for complex logic

#### Submitting Changes

1. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   ```
2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
3. **Create a Pull Request**
4. **Fill out the PR template**
5. **Wait for review**

### Adding New Artistic Styles

To add new artistic styles:

1. **Edit the `STYLE_CATEGORIES` dictionary** in `app.py`
2. **Add corresponding artists** to `STYLE_ARTISTS` dictionary
3. **Test the new styles** with various prompts
4. **Update documentation** if needed

Example:
```python
"My New Category": {
    "New Style": ["tag1", "tag2", "tag3", "tag4", "tag5"]
}

# And in STYLE_ARTISTS:
"New Style": ["Artist 1", "Artist 2", "Artist 3"]
```

### Improving AI Models Integration

When working with AI model integrations:

1. **Test with multiple models** (Flux, Midjourney, SDXL, SD 1.5)
2. **Handle errors gracefully**
3. **Add appropriate fallbacks**
4. **Document any new parameters**

### Documentation Improvements

- **Update README.md** for new features
- **Add examples** for new functionality
- **Keep documentation clear** and concise
- **Include screenshots** where helpful

## Development Guidelines

### Testing

- **Test all functionality** before submitting
- **Test with different inputs** (various languages, long prompts, etc.)
- **Test error conditions**
- **Verify UI responsiveness**

### Performance Considerations

- **Profile memory usage** for large operations
- **Optimize prompt generation** for speed
- **Consider GPU memory limitations**
- **Handle large images efficiently**

### Security

- **Never commit API keys** or secrets
- **Validate user inputs** properly
- **Handle file uploads** securely
- **Follow secure coding practices**

## Community Guidelines

### Code of Conduct

- **Be respectful** and inclusive
- **Welcome newcomers**
- **Provide constructive feedback**
- **Help others learn**

### Communication

- **Use clear, descriptive titles** for issues and PRs
- **Be patient** with review processes
- **Ask questions** if something is unclear
- **Share knowledge** and experiences

## Getting Help

- **Check the README** first
- **Search existing issues**
- **Join community discussions**
- **Ask specific questions** with context

## Recognition

Contributors will be recognized in:
- **Contributors list** in README
- **Release notes** for significant contributions
- **Project documentation**

Thank you for contributing to Magic Prompt Generator! 🪄
  
