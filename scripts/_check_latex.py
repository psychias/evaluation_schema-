#!/usr/bin/env python3
import re

with open('submission/main.tex') as f:
    text = f.read()
    lines = text.split('\n')

# Check matching begin/end
envs = re.findall(r'\\(begin|end)\{(\w+)\}', text)
stack = []
for i, (action, env) in enumerate(envs):
    if action == 'begin':
        stack.append((env, i))
    elif action == 'end':
        if stack and stack[-1][0] == env:
            stack.pop()
        else:
            expected = stack[-1][0] if stack else 'nothing'
            print(f'Mismatch at env #{i}: \\end{{{env}}} but expected \\end{{{expected}}}')

if stack:
    print(f'Unclosed environments: {[s[0] for s in stack]}')
else:
    print('All environments properly matched')

# Check brace balance
total_open = text.count('{')
total_close = text.count('}')
print(f'Braces: {{ = {total_open}, }} = {total_close}, diff = {total_open - total_close}')

# Check for $...$ balance (rough)
dollar_signs = len(re.findall(r'(?<!\\)\$', text))
print(f'Dollar signs (non-escaped): {dollar_signs} ({"even" if dollar_signs % 2 == 0 else "ODD - potential issue"})')

print(f'Total lines: {len(lines)}')
