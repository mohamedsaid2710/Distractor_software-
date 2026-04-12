# Future Refactoring Plan: `choose_distractor`

**Context:** The `choose_distractor` method in `sentence_set.py` has grown into a massive monolith over time. It currently mixes database lookups, neural network batch scoring, linguistic grammatical filtering, and layers of desperation fallbacks into a single, deeply nested loop. 

When you have the time and energy in the future (and **only** after your current experiments are completely finished and published!), here is the step-by-step blueprint for dismantling and rebuilding it safely.

---

### Step 1: Lock behavior before refactoring (Golden Tests)
Before changing a single line of logic, you need to prove that the psycholinguistic outputs won't shift underneath you.
* **Goal:** Create a testing script that feeds specific sentences and specific parameters into the system, and saves the exact output distractors to a file.
* **Action:** Run your exact English, German, and Arabic test sets using a fixed `random.seed(42)`. Save the outputs. 
* **Validation:** As you rewrite the function in the later steps, run the test script. If the output words change *at all*, you know your refactor accidentally broke a rule.

### Step 2: Stop mutating `params` inside the selection loop
Currently, the script saves local token states (like `params['target_is_noun']` or `params['target_is_capitalized']`) directly into the global `params` dictionary. 
* **The Problem:** This creates a dangerous landscape where loops accidentally contaminate each other. A word might be treated as a noun simply because the *previous* word in the loop was a noun.
* **Action:** Pass an immutable "context" object or a clear set of local variables (e.g., `target_length`, `is_noun_flag`, `zipf_score`) to the distractor function instead of reading/writing back to the global `params` dictionary dict.

### Step 3: Split `choose_distractor` into Pure Pipeline Stages
Break the giant loop into clear, separate, isolated functions. Each stage should have clear inputs and clear outputs.
1. **Candidate Retrieval Stage:** Go to the dictionary and get all words matching the base length and frequency. (No grammar checking here, just building the list).
2. **Cheap Linguistic Filtering Stage:** Filter out banned words, proper nouns, and bad regex patterns (fast operations).
3. **ML Scoring Stage:** Send the clean list to the Language Model for surprisal scoring.
4. **Threshold Decision Stage:** Compare the scores against `target_surprisal` (Mode A/B thresholds) and pick the best one.
5. **Fallback Strategy Stage:** If Step 4 fails, trigger the fallback rules (relaxing length, ignoring repeats) in a highly structured way.
6. **Post-Processing Stage:** Apply final casing and punctuation to the chosen word.

### Step 4: Make Fallbacks Deterministic and Traceable
Debugging silent failures in the fallback loops took hours because the system didn't leave a trace of *how* it made its decision.
* **Action:** Have the system return a structured "decision trace" along with the word. (e.g., `[Stage 2: Filtered 500 candidates -> 0; Triggered Stage 3 Fallback; Selected 'arg']`).
* **Action:** Use a local random number generator seeded by the specific item/label ID instead of the global `random.shuffle()`. This guarantees your maze generations are 100% reproducible on any computer.

### Step 5: Replace Blanket `try/except` with Scoped Failures
Right now, the code uses broad `try: ... except Exception: pass` blocks.
* **The Problem:** If a completely unrelated bug happens (like a missing variable), the code silently swallows the error, panics, and dumps you into the "Total Desperation" fallback pool without telling you why.
* **Action:** Only catch the exact errors you expect (like `CUDA OutOfMemory Error` during ML scoring). Let other errors gracefully crash the script so you can immediately see and fix them instead of generating bad mazes.

### Step 6: Optimize Last
Only after the structure is clean, easy to read, and passing all the tests from Step 1, should you look into making it faster. 
* Do not mix "performance optimizations" (like batch processing) with "correctness logic" (like grammar checking). Keep them completely separated!