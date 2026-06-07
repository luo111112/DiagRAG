"""Find all problematic { } in RAG_PROMPT_TEMPLATE."""
import sys
sys.path.insert(0, '/app')

from src.generation.prompts import RAG_PROMPT_TEMPLATE

lines = RAG_PROMPT_TEMPLATE.split('\n')
for i, line in enumerate(lines, start=1):
    # Find { that's not part of {{
    in_template = False
    j = 0
    while j < len(line):
        if line[j] == '{':
            if j + 1 < len(line) and line[j+1] == '{':
                j += 2  # skip {{
            else:
                print(f"Line {i}: {repr(line)} (pos {j})")
                break
        j += 1

print("\nAll { } that aren't {{ }}:")
import re
for m in re.finditer(r'(?<!{)\{(?!{)([^}]*)\}(?!})', RAG_PROMPT_TEMPLATE):
    print(f"  pos {m.start()}-{m.end()}: {repr(m.group())}")
