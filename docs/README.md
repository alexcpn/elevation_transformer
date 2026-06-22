# Slides

This directory contains a Marp-compatible Markdown slide deck:

- `itm_transformer_talk.md`

Example export commands:

```bash
npx @marp-team/marp-cli /ssd/elevation_transformer/slides/itm_transformer_talk.md --pdf --output /ssd/elevation_transformer/slides/itm_transformer_talk.pdf
```

```bash
npx @marp-team/marp-cli /ssd/elevation_transformer/slides/itm_transformer_talk.md --pptx --output /ssd/elevation_transformer/slides/itm_transformer_talk.pptx
```

# PDF

Run this from the root

```
pandoc --pdf-engine=xelatex README.md  -o docs/paper.pdf
```