"""Fix prompts.py and verify."""
import re

# Read the original file
with open('/app/src/generation/prompts.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Save the two real placeholders as safe markers
content = content.replace('{context}', '\x00CTX\x00')
content = content.replace('{question}', '\x00QST\x00')

# Escape ALL remaining { and } by doubling them.
# We must NOT double already-doubled {{ or }}.
# Strategy: find all occurrences of {{ or }} and temporarily replace them,
# then double all singles, then restore the doubles.

# Step 1: Protect existing {{ and }} with safe markers
protected = content.replace('{{', '\x01DOUBLE_OPEN\x01')
protected = protected.replace('}}', '\x01DOUBLE_CLOSE\x01')

# Step 2: Double all single braces
protected = protected.replace('{', '{{')
protected = protected.replace('}', '}}')

# Step 3: Restore {{ and }}
protected = protected.replace('\x01DOUBLE_OPEN\x01', '{{')
protected = protected.replace('\x01DOUBLE_CLOSE\x01', '}}')

# Step 4: Restore the real placeholders
protected = protected.replace('\x00CTX\x00', '{context}')
protected = protected.replace('\x00QST\x00', '{question}')

# Write fixed file
with open('/app/src/generation/prompts_fixed.py', 'w', encoding='utf-8') as f:
    f.write(protected)

print("Fixed file written.")

# Verify
import sys
sys.path.insert(0, '/app/src')
import importlib
import generation.prompts_fixed as pf
importlib.reload(pf)

from src.generation.prompts_fixed import RAG_PROMPT_TEMPLATE
test_context = "[文档1]\n心肌梗死典型症状"
test_question = "急性心肌梗死的典型症状有哪些？"
try:
    result_str = RAG_PROMPT_TEMPLATE.format(context=test_context, question=test_question)
    print("SUCCESS! format() works. Length:", len(result_str))
    jidx = result_str.find('"analysis"')
    if jidx >= 0:
        print("JSON section:", repr(result_str[jidx-3:jidx+60]))
except Exception as e:
    print("FAILED:", type(e).__name__, str(e))
