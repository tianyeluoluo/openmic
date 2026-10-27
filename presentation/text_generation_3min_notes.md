# Text generation: three-minute English script

Integrated deck: [midterm.html](midterm.html), **slides 4–7**, section **02 / znjq**.
The [standalone four-slide version](text_generation_3min.html) contains the same content.
This section follows the previous speaker and has no separate title slide.

| Team slide | Topic | Time | Elapsed |
|---|---|---:|---:|
| 4 | The whole-story view | 40 sec | 0:40 |
| 5 | Prompting method | 45 sec | 1:25 |
| 6 | Generation pipeline | 45 sec | 2:10 |
| 7 | Four evaluation dimensions | 50 sec | 3:00 |

## Slide 4: Humor in the whole story

I work on text generation. My working view is that a joke can be treated as a complete story. We need to understand what a character wants, how they react, and how the result changes our expectations.

During optimization, a model may polish separate setups, punchlines, and callbacks until the overall story feels artificial. So I check motivation, cause, and rhythm together. Believable coincidences and inner thoughts can still be useful.

## Slide 5: CFunSet-guided prompt engineering

Our method combines retrieval with prompt engineering. We retrieve examples from the training portion of CFunSet, then identify the listener's initial expectation and what changes it.

Role prompts separate planning, writing, and review. Writing may start from a funny moment or an experience. Adaptation keeps the original mechanism and necessary context. Original writing transfers the mechanism into a new situation. We provide examples through prompts without fine-tuning DeepSeek.

## Slide 6: Generation pipeline

The input contains a topic, audience, and real context. After reference selection, five roles handle direction, audience analysis, writing, delivery, and quality control. These drafting calls use DeepSeek Flash.

In the additional quality loop, DeepSeek V4 Pro reviews the whole script and checks continuity. Pro also rewrites it using the original draft and specific feedback. We review the revision again, retain the best candidate within a revision budget, and record failed attempts.

## Slide 7: Four core evaluation dimensions

Pro uses a prompted one-to-five rubric. Humor considers surprise and comic response. Cultural fit covers Chinese context and audience knowledge. Spoken naturalness checks phrasing aloud. Structure covers coherence, reveal timing, and useful callbacks.

For humor, understandable but weak content stays around three. Four requires a specific comic turn. Problems need a quoted line, a reason, and a repair. We also check realism and continuity. These scores are model judgments, separate from CFunSet's binary labels. Human ratings remain necessary to test whether revisions are actually funnier.

## Presenter reminders (not spoken)

- The timing is a three-minute plan, not a measured recording. Rehearse once at a measured pace with brief pauses.
- The four core dimensions are rubric scores. The independent audit also reports `realism` and runs a separate fact check.
- Independent quality-loop scores use 1–5. Internal five-role QC uses 0–10.
- Actual API model IDs: `deepseek-flash` for initial drafting, `deepseek-v4-pro` for independent review and revision. These calls do not update model weights.
- The ordinary five-role CLI does not automatically invoke the additional Pro quality loop.
- CFunSet contains multiple tasks. Its binary-label records train a separate local classifier, whose probability is a weak auxiliary signal, not a 1–5 audience humor score.
- The whole-story view allows joke-first and experience-first writing, including short jokes.
- Sources: `src/openmic/autogen_v02.py`, `humor_generation.py`, `humor_data.py`, `humor_craft.py`, `humor_quality_loop.py` and `humor_scorer.py`.
