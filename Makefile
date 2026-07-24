default:	all

.PHONY: clean all mrproper

all:	main.pdf

BIBINPUTS += ".:~/bibs:~/home/"

main.pdf:	*.tex *.bib
		pdflatex main
		(export BIBINPUTS=${BIBINPUTS}:${BIBS}; bibtex main)
		pdflatex main
		makeglossaries main
		pdflatex main
		pdflatex main

clean:
		rm -f *.aux *.bbl *.blg *.log *.out *.pdf

publish:

		scp main.pdf vruiz@dali.hpca.ual.es:public_html/medical_imaging.pdf
