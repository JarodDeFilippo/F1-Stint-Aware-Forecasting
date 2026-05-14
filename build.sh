# build.sh
pandoc report/report.md -o report/report.pdf \
  --pdf-engine=xelatex \
  --resource-path=report \
  -H report/header.tex \
  --shift-heading-level-by=-1