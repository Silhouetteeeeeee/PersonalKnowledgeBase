# XMind Parser Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans.

**Goal:** Replace the incompatible xmind package with xmindparser for XMind file parsing.

**Architecture:** Single-function replacement in _extract_text_from_xmind.
**Tech Stack:** Python, xmindparser

---

### Task 1: Update requirements.txt

**Files:**
- Modify: requirements.txt:19

- [ ] Replace xmind with xmindparser

Change: xmind>=1.2.0 -> xmindparser>=1.0.0

- [ ] Commit

### Task 2: Replace _extract_text_from_xmind

**Files:**
- Modify: storage/file_processor.py:72-107

- [ ] Replace function body

Old: import xmind; wb=xmind.load(...); manual tree walk
New: from xmindparser import xmind_to_markdown; return xmind_to_markdown(file_path)

Keep try/except/logger pattern unchanged.

- [ ] Commit

### Task 3: Update XMind test

**Files:**
- Modify: tests/test_file_processor.py:55-82

Old test uses xmind package API to create .xmind fixtures.
Replace with stdlib zipfile + xml.etree.ElementTree.

- [ ] Rewrite test to generate .xmind via zipfile + XML
- [ ] Commit

### Task 4: Run tests and verify

- [ ] Run: python -m pytest tests/test_file_processor.py::test_extract_text_from_xmind -v
- [ ] Run: python -m pytest tests/test_file_processor.py -v
- [ ] Final commit if needed
