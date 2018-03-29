# conda install -c conda-forge pdfminer3k
from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
import numpy as np
import glob, os, pickle
import math



def pdfToTextDict(filename):
    textDict = []
    datafiles = sorted(glob.glob(filename+'*.pdf'))
    for pdf in datafiles:
        fp = open(pdf, 'rb')
        parser = PDFParser(fp)
        doc = PDFDocument()
        parser.set_document(doc)
        try:
            doc.set_parser(parser)

            doc.initialize('')
            rsrcmgr = PDFResourceManager()
            laparams = LAParams()
            laparams.char_margin = 1.0
            laparams.word_margin = 1.0
            device = PDFPageAggregator(rsrcmgr, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            extracted_text = ''

            for page in doc.get_pages():
                interpreter.process_page(page)
                layout = device.get_result()
                for lt_obj in layout:
                    if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
                    #if isinstance(lt_obj, LTTextBox):
                    #compare performance
                        extracted_text += lt_obj.get_text()

            textDict.append(extracted_text)
        except:
            print("set parser error)")

    return textDict

f1 = '/Users/liammeier/Dropbox/MoralReasoning/AppliedConsequentialism/'
f2 = '/Users/liammeier/Dropbox/MoralReasoning/AppliedDeontology/'
pickle.dump(pdfToTextDict(f1), open("consPapers.pkl", "wb"))
pickle.dump(pdfToTextDict(f2), open("deonPapers.pkl", "wb"))

