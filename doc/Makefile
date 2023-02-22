doc.pdf: doc.tex doc.bib
	pdflatex doc.tex
	bibtex doc
	pdflatex supplement.tex
	bibtex supplement
	pdflatex supplement.tex
	pdflatex doc.tex
	pdflatex supplement.tex
	pdflatex doc.tex
	gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook -dPrinted=false -dNOPAUSE -dQUIET -dBATCH -sOutputFile=smalldoc.pdf doc.pdf

doc.ps: doc.dvi
	dvips doc

doc.dvi: doc.tex bib.bib
	latex doc.tex
	bibtex doc
	latex doc.tex
	latex doc.tex
clean:
	rm -f *.log *.dvi *.aux
	rm -f *.blg *.bbl
	rm -f *.eps *.[1-9]
	rm -f src/*.mpx *.mpx

mrproper:
	rm -f *.ps *.pdf
