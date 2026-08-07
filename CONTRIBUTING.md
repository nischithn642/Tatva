# Contributing to TATVA

Thank you for your interest in contributing to **TATVA: Bare-Metal RISC-V Transformer Optimization Toolchain**!

---

> [!IMPORTANT]
> **Out-of-Scope Scope Creep Warning:**
> TATVA is strictly a **local-only developer toolchain and model optimizer**.
> To prevent scope creep, the following feature categories are explicitly **OUT OF SCOPE** and Pull Requests adding them will be rejected:
> - User account systems, authentication platforms, or login portals.
> - SaaS cloud backends, hosted web APIs, or online database dashboards.
> - Model marketplaces, token payments, or social sharing platforms.

---

## 1. Local Developer Environment Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/tatva.git
cd tatva

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install in editable mode with development tools
pip install -e .[dev]

# Pre-commit hook setup
pre-commit install
```

---

## 2. Code Quality & Testing Guidelines

### Code Formatting & Linting (`ruff`)
All code must pass `ruff` lint checks cleanly:

```bash
# Check code formatting and linting
ruff check src/ tests/

# Format code automatically
ruff format src/ tests/
```

### Test Tiers

We enforce two test execution tiers:

1. **Unit Test Suite (Fast, offline, mock-friendly):**
   ```bash
   pytest -m unit -v
   ```
2. **Full Integration Suite (Includes QEMU compiler emulation passes):**
   ```bash
   pytest -v
   ```

---

## 3. Pull Request Guidelines

1. **Keep Changes Scoped:** One feature or bug fix per Pull Request.
2. **Preserve Benchmark Honesty:** Ensure any new benchmark numbers added to documentation are measured under **QEMU system-mode emulation (`-icount shift=0`) at 100MHz nominal frequency** and clearly labeled.
3. **Secret Hygiene:** Ensure no API keys, private credentials, or raw model weight binaries are committed.
4. **All Tests Must Pass:** CI checks (`ruff`, `gitleaks`, `pytest`) must pass cleanly before merge.
