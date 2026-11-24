/*
Objective: Carry out independent explorations. Open-ended research project using
modern NLP knowledge.

Proposal (10%): 1-page single PDF with unlimited references. No specific style
requirements are mandated, but standard academic report formatting (e.g.,
legible fonts, appropriate spacing) is expected.


Proposal (10%):
- Research question & motivation (2%): States a focused research question and
  explains why it matters for NLP. Makes clear what the project aims to
  investigate or learn.
- Relevant background (2%): Identifies directly relevant prior work and explains
  how the proposed project relates to it.
- Method & evaluation plan (2%): Describes a concrete approach and how it will
  be evaluated, including appropriate data, metrics, and comparisons.
- Feasibility & scope (2%): Presents a realistic scope and timeline given the
  team size, available resources, and remaining time.
- Clarity & format (2%): Communicates the plan clearly, uses readable standard
  academic report formatting, follows the page and file requirements, and
  includes accurate citations.
*/

#show link: set text(fill: blue)

== Motivation, background and research question

A robust understanding how LLMs work will require understanding the relationship
between statistical structure in the training data, the loss landscape and
algorithmic structures in the learned model@biderman2026. Lehalleur and
co-authors argue that understanding this relationship is central to AI
alignment@lehalleur2025. This research agenda has been described as training
dynamics@biderman2026, learning dynamics@shallow2025 and learning
mechanics@simon2026. Where mechanistic interpretability studies the learned
computational units in transformers, learning dynamics studies how and why those
circuits emerge.

One approach based on Singular Learning Theory (SLT)@watanabe2009 tracks the
local learning coefficient (LLC) of a model over the course of the training
process to uncover transitions in the learned algorithmic structures (e.g., the
development of induction heads)@hoogland2025. The LLC describes the geometry of
the loss function. It answers the question "in the space around $w$, how much
does the loss function change". It is related to, but more informative than the
Jacobian/Hessian of the loss function. It can be estimated as

$
  hat(lambda)(w^*) = n beta [ bb(E)^beta_(w | w^*, gamma) [ell_n (w)] - ell_n
    (w^*) ], quad
  p(w; w^*, beta, gamma) prop exp lr({ -n beta ell_n (w) - gamma / 2 norm(w - w^*)_2^2 })
$ <eq:llc>


Where $ell_n$ is the empirical loss over a sample from the data distribution of
size $n$. This quantity is loosely "the difference between the loss at $w^*$ and
the weighted average of the loss in its neighborhood". The expectation of the
nearby loss ($bb(E)^beta_(w | w^*, gamma)$) is weighted under the Gibbs
posterior, $p$, and $beta$ and $gamma$ are hyperparameters.
// TODO: why does a measure of the geometry say anything about complexity?

This approach was extended by Wang et. al (2024)@wang2024 to introduce the
concept of the weight-refined LLC (wrLLC) and data-refined LLC (drLLC). These
are found by calculating the standard LLC on a subset of the weights (e.g., a
single attention head) or on data drawn from a subset of the data distribution.
In section 4.2, they show the drLLC for two induction attention heads evaluated
on data from the training distribution and data from a Github/code dataset. They
remark: _"data-refined LLCs quantitatively illustrate which components of a
model are being shaped in response to what structure in the data. For example,
[the attention head] may have been significantly influenced by code"_. My goal
with this project is to causally validate this claim.

== Methodology, scope and feasibility

I will reproduce Wang et al.'s methodology, but using a training dataset with
all code/programming tokens removed. Like the authors, I will use a subset of
"The Pile"@gao2020 dataset for training. The authors have not published their
code, but they have published their training parameters: #link(
  "https://github.com/timaeus-research/paper-rllcs-2024",
)[Wang 2025 Github]. They also maintain an open source library calculating LLC
and categorizing attention heads: #link(
  "https://pypi.org/project/devinterp/",
)[devinterp 2.0.1] which will be useful. Training will require
$50k "steps" times 100 "batches" times 1024 "sequence length" approx 5 times 10^9$
tokens. The model is 3M parameters, so it will take
$C approx 6 N D = 6 * 3M * 5 times 10^9 = 9 times 10^16$ flops to train. This is
roughly 5 A100 hours. I expect the LLC calculation to take roughly the same
time.

I will then calculate the LLC at each 100th step of training for the induction
heads in layer 1 (classified using devinterp) from the pre-training data
distribution and a Github/code data distribution and compare the relative
difference
$(hat(lambda)_("Github") - hat(lambda)_("Pile")) \/ hat(lambda)_("Pile")$ to see
if the LLC of one of the induction heads has a higher value. This mirrors Figure
3 from their paper. If the authors are correct, it will not.

I plan to spend 4 weeks reproducing Wang et al.'s results: reading their
Appendix and Github for implementation details, cleaning the Pile data, and
writing the training code, and setting up the GPU training infrastructure. Next
I'll spend 1 week performing the analysis described above on the saved snapshots
of model weights from the training run. I'll spend the last week writing the
report.

#pagebreak(weak: true)

_AI usage note: I used Claude to help with literature review, brainstorm
research directions, and estimate compute budget._

#bibliography("refs.bib", style: "ieee")

/*
= archive

== other papers to extend

Baker et al., “Structural Inference: Interpreting Small Language Models with Susceptibilities” (ICLR 2026). A small perturbation of the data distribution, such as shifting the Pile toward GitHub or legal text, induces a first-order change in an observable on a chosen network component. The resulting response matrix separates functional modules such as multigram and induction heads in a 3M-parameter transformer. This is the most direct formalization of how data composition shapes structure

Wang & Murfet, “Patterning: The Dual of Interpretability” (arXiv 2601.13548). In a small language model, re-weighting training data along principal susceptibility directions can accelerate or delay the formation of structure such as the induction circuit. This is very close to your proposed data-composition experiment

== random

2. Finer-grained weight refinements: Calculate wrLLC on QK and OV matrices for
  each attention head. Plot the same plots and see what changes. Con: "see what
  changes" is pretty loose
3. New data refinements. Compute drLLCs on synthetic distributions that isolate
  one pattern type, such as pure Dyck/bracket sequences, repeated-token
  induction sequences, or math/multilingual text. Then check whether each head’s
  drLLC tracks its Table 1 classification. Compute: one big run
  "
  There are similar constructions that we did not analyze but that would make
  natural candidates for follow-up analysis, such as additional correlative
  comparatives (“as ... as”) conditional statements (“if ... then”), cleft
  sentences (“it is ... who”), comparative correlatives (“the [more you
  practice], the [better you get]”), result clauses (“[it was] so [cold] that
  [the lake froze]”)
  "

I plan to train a small language model from scratch. If I had control over the
dataset, perhaps I could alter composition to compare the resulting training
milestones and/or algorithmic structures between the models. Alternatively I
could just compare runs across different random weight initializations. With a
small enough dataset each training run would hopefully take only a few hours on
a single GPU.

@hoogland2025


Dataset options include

- TinyStories (Eldan & Li, 2023)
- Subset of the pile: https://arxiv.org/abs/2101.00027 like @hoogland2025
- others???

possible research questions

- induction-head formation
- ???
- vary some structure of TinyStories dataset??


https://pypi.org/project/devinterp/

Future work: Verification on EleutherAI's Pythia dataset@biderman2023.
*/
