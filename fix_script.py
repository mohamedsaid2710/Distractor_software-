import sys
# A small test script to help find how _get_german_grammatical_case is defined
with open("/mnt/c/Users/moham/Desktop/maze_automate/sentence_set.py", "r") as f:
    content = f.read()

import re
match = re.search(r'def _get_german_grammatical_case.*?(?=def |\Z)', content, re.DOTALL)
if match:
    print(match.group(0))
