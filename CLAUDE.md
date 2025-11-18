# CLAUDE Development Guidelines

## Core Principles

1. **Do not rush to an answer**
   - If there is a necessary question, ask it before responding
   - Clarify requirements and approach before implementation
   - Take time to understand the full context

2. **Multiple questions format**
   - When asking multiple questions, use NUMBERED or ALPHABETICAL LISTS
   - This allows for unambiguous answers
   - Makes it easy to reference specific questions

3. **Verification and validation**
   - Recheck or verify if you think I'm going for the wrong approach
   - Challenge assumptions that may lead to issues
   - Suggest alternatives when better solutions exist

## Python Environment Management

**IMPORTANT: Always use `uv` for Python package management**

- **DO NOT** use `pip install` - use `uv pip install` instead
- **DO NOT** use `python` directly - use `uv run python` instead
- **DO NOT** use `pytest` directly - use `uv run pytest` instead

### Common uv Commands
```bash
# Install dependencies
uv pip install <package>

# Run Python scripts
uv run python script.py

# Run tests
uv run pytest

# Sync dependencies from pyproject.toml
uv pip sync
```

## Implementation Notes

This document tracks the development approach and decisions made during the AMRx Simulator implementation.

### Key Design Decisions

1. **Project Structure**: Package-based with submodules
   - Main package: `amrx/`
   - Submodules: `amrx/sensors/`, `amrx/robot/`, `amrx/world/`, etc.

2. **Testing Framework**: pytest
   - Modern, feature-rich testing
   - Easy fixture management

3. **User Interface**: PyQt6
   - Primary visualization using PyQt6
   - Real-time rendering and interactive controls

4. **World Maps**: JSON format stored in `maps/` directory
   - Environment A: Empty Box (basic testing)
   - Environment B: Corridor (line following)
   - Environment C: SLAM Arena (full SLAM testing)

5. **Development Workflow**: Commit and push after each phase completion
   - Version control at major milestones
   - Clean git history

6. **Code Style**: PEP 8 with type hints
   - Modern Python practices
   - Type annotations for better IDE support and documentation

### Questions and Clarifications
- To be filled as questions arise

### Known Issues
- To be tracked during development
