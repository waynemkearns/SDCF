# Contributing to SDCF

Thank you for your interest in contributing to the Synthetic Data Compliance Framework!

## Ways to Contribute

### 📊 Empirical Validation
Run SDCF assessments on your datasets and share results to help calibrate thresholds.

### 🔧 Tool Integrations
Build connectors for additional synthetic data tools and metrics libraries.

### 📚 Case Studies
Document real-world SDCF applications in different domains.

### 🐛 Bug Reports
Found an issue? Open a GitHub issue with:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)

### 💡 Feature Requests
Have an idea? Open a GitHub issue describing:
- The use case
- Proposed solution
- Any alternatives considered

---

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/waynemkearns/SDCF.git
cd sdcf
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run tests (if available):
```bash
pytest
```

---

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to all functions and classes
- Include type hints where appropriate

---

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests to ensure nothing is broken
5. Commit with clear messages (`git commit -m "Add feature X"`)
6. Push to your fork (`git push origin feature/your-feature`)
7. Open a Pull Request

---

## License

By contributing, you agree that your contributions will be licensed under:
- MIT License for code contributions
- CC BY-NC-SA 4.0 for documentation contributions

---

## Questions?

Contact: wayne.kearns@nortesconsulting.com

Thank you for helping improve SDCF!

