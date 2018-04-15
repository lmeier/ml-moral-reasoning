# conda install -c conda-forge pdfminer3k
from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
import numpy as np
import glob, os, pickle
import math



def txtToTextDict(filename):
    textDict = []
    datafiles = sorted(glob.glob(filename+'*.txt'))
    for txt in datafiles:
        fp = open(txt, 'rb')
        extracted_text = fp.read()
        textDict.append(extracted_text)
    print(textDict)
    return textDict

f1 = '/Users/liammeier/moral-reasoning/AppliedConsequentialism/'
f2 = '/Users/liammeier/moral-reasoning/AppliedDeontology/'
pickle.dump(txtToTextDict(f1), open("consPapersNewest.pkl", "wb"))
pickle.dump(txtToTextDict(f2), open("deonPapersNewest.pkl", "wb"))

