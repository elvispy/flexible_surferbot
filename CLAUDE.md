This is the set of guidelines to be followed to achieve MATLAB/Julia parity
from the MATLAB/old_code. THese guidelines are non-negotiables:

1) The file docs/paper_draft.tex sohuld be treated as the canonical ground truth
for the Julia implementation. If a term in the Julia codebase is demeed as wrong 
because it has a wrong physics term, then it should be compared against the paper draft
If it disagrees with the paper draft, then the julia codebase can be changed
to match the ground truth. If, at any point, via reading the file or by comparing against the code, it is found that the .text file is not mathematically correct or self-consistent, that should be surfaced to the user. No changes to the .tex file are allowed without explicit approval from the user

2) The source of truth for the MATLAB old_code/ implementation is the Benham et al 2024 paper 'On-wave driven propulsion' (the JFM versioin, because the arxiv version has a few typos)
The same applies: if the MATLAB version is thought to be defective, we consult with the paper first. Similarly to the Julia, if a problem is surfaced in the codebase, first it should be comapred against the source truth, to match it. 

3) You are allowed to change the inputs of the Julia version as needed (but not the solver, unless permitted by rule 1)) to achieve parity. 
For example, you can change the forcing width, turn off surface tension, change the Lambda parameter value in Julia, because all of those are physics that are not necesarily present in the MATLAB version. 
However, whatever is mapped from MATLAB to Julia, they SHOULD MATCH their spiritual analogues (motor position, frequency).

4) If, at any point, you believe the rules in this CLAUDE.md file must be changed because they are hindering our ultimate goal of achieving MATLAB/Julia parity, raise it to the user to discuss

5) To avoid repeating mistakes, we should proceed with the scientific method: 
Make an hypothesis, conduct a numerical experiment, analyze data, report conclusions, repeat. Hypothesis and their falsifiability process should be first-class citizens. This is done in MILESTONES.md. The document should contain itmized classess, where each entry forms one iteration of the scientific method as described in this rule. 
MILESTONES is located at /Users/harrislab/.claude/projects/-Users-harrislab-Documents-GitHub-waves-code/memory/MILESTONES.md
6) If, at any point, MILESTONES.md gives contradicting evidence or is not self-consistent, that should be resolved before proposing or adding a new entry. MILESTONES.md should aim to be as close as possible to being ground-truth, high signal hints. After a MILESTONE.md entry is written, the codebase should be commited with a reference to the MILESTONE.md entry..

7) We should have a mental model of what changes have been made to the source code codebase. Julia/src and MATLAB/old_code have cannonical, non-parity versions whose commits can be taken to be the ones before May 1st. That way you can know which changes to the source code that pertain to rules (1) and (2) have been done, and that might help you resolve lack of self-consistency as described un rule (6)

8) We are interested in full MATLAB/Julia parity. That means that all inputs that are present on both models should be the same, and the vector \eta(x_i) should be the same in the whole free-surface/raft domain. We are not interested in scalar diagnostic matchs (they can be used as diagnostics, but cannot be claimed to match a success criterion)
For example, we are not interested in asymptotic parity (a la 'as n -> Inf, as m -> Inf', this scalar quantity like thrust matches)
THe L2 norm of \eta_Julia and \eta_MATLAB is the ONLY success criterion for this parity task.

---

These other set of rules are to be followed for figure visualization


