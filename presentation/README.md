# OpenMic midterm presentation

Open `midterm.html` in a browser. The deck is self-contained, uses system fonts,
and needs no server, build step or network connection.

## Slide ownership

| Slide | Owner | Content |
|---|---|---|
| 1 | Team 1234 | Cover |
| 2 | lyy | Goal; defining humor, writing jokes and delivering them |
| 3 | lyy | Five-agent architecture, revision loop and TTS flow |
| 4 | znjq | TODO: content generation |
| 5 | csw / jyb | TODO: speech generation |
| 6 | szq | TODO: evaluation and current progress |
| 7 | szq | TODO: limitations, next steps and closing |

The lyy section describes the local implementation and project goals. The
3–5 minute duration is a target, not a measured result. The remaining sections
are placeholders until their owners supply content.

Slide 2 presents our working challenge analysis, not a universal theory of humor.
Audience ratings and matched text/audio comparisons are proposed validation
methods. Additional speaking notes are in the HTML comment above that slide.

## Present and edit

- Use arrow keys, Space, Page Up/Down or the on-screen arrows to change slides.
- Press E, or use the top-left edit button, to edit headings and placeholder text.
- Press Esc to leave editing mode. Text changes are cached in this browser only.
- Choose **Save HTML** or press Ctrl/Cmd+S to download the edited deck and share it.
- Use **Print / PDF** for a static copy; check the browser's print preview before saving.
- SVG diagram labels and layout are edited in the HTML source.

This is a local file, not a real-time shared editor. For parallel AI editing,
each owner should modify only their commented section in `midterm.html` and
submit the change for integration. Shared CSS and navigation belong to the
integrator. Keep all visible slide text in English and leave unknown results as TODO.

## Design and evidence

The canvas is fixed at 1920 × 1080 and scales uniformly. Use a white background,
system fonts, simple blue/gray diagrams and little text. Keep each page to one
main visual. Avoid adding unverified progress or results to placeholders.

The diagrams on slides 1–3 are explanatory graphics, not application screenshots.
Architecture references: `src/openmic/autogen_v02.py`,
`src/openmic/streamlit_app.py`, and `src/openmic/tts.py`.
The TTS action is user-triggered; the quality gate stops at approval or the
revision limit, and the UI displays the outcome in either case.

## Task 2 text-generation presentation

Open [text_generation.html](text_generation.html) for the standalone text-generation section.
It follows the same white/blue/gray theme, fixed 1920 × 1080 canvas, navigation,
local editing, **Save HTML**, and print controls. It has no external scripts,
fonts or build dependencies. Main explanations are in English; Chinese joke
excerpts retain their original wording.

- **Length:** 9 slides, about 7–8 minutes for one speaker.
- **Speaking plan:** [Chinese notes with per-slide timing](text_generation_notes.md), target 7:40.
- **Content:** CFunSet preparation, reference selection, two writing routes,
  topic expansion, a callback example, evaluation, local training experiments
  and current limitations.
- **Evidence:** [release summary](../docs/TASK2_RELEASE.md) and
  [compact development results](../docs/task2_v39_results.json).

This separate file leaves the shared `midterm.html` and other owners' placeholders
unchanged. An integrator can reuse the text-generation slides when assembling
the team presentation. The optional training experiments are distinguished from
DeepSeek generation, and the four-topic model review is distinguished from human
audience evaluation.
