# from spacy.language import Language
# import spacy
# import re
# import csv
#
#
# @Language.component("custom_sentencizer")
# def custom_sentencizer(doc):
#     for i, token in enumerate(doc[:-1]):
#         if re.match(r'^ ?\.', token.text) and (doc[i + 1].is_upper or doc[i + 1].is_title):
#             doc[i + 1].is_sent_start = True
#             token.is_sent_start = False
#         elif re.match(r'[0-9]{1,2}\.$', token.text):
#             if not doc[i - 1].is_stop:
#                 token.is_sent_start = True
#                 doc[i + 1].is_sent_start = False
#             else:
#                 token.is_sent_start = False
#                 doc[i + 1].is_sent_start = True
#         elif token.text == '-' and doc[i + 1].text != '-':
#             token.is_sent_start = True
#             doc[i + 1].is_sent_start = False
#         else:
#             doc[i + 1].is_sent_start = False
#     return doc
#
#
# @Language.component('tkndef')
# def def_tokens(doc):
#     patterns = [r'\[\*\*.+?\*\*\]',  # de-identification
#                 r'[0-9]{1,4}[/\-][0-9]{1,2}[/\-][0-9]{1,4}',  # date
#                 r'[0-9]+\-?[0-9]+%?',  # lab/test result
#                 r'[0-9]+/[0-9]+',  # lab/test result
#                 r'([0-9]{1,3} ?, ?[0-9]{3})+',  # number >= 10^3
#                 r'[0-9]{1,2}\+',  # lab/test result
#                 r'[A-Za-z]{1,3}\.',  # abbrv, e.g., pt.
#                 r'[A-Za-z]\.([a-z]\.){1,2}',  # abbrv, e.g., p.o., b.i.d.
#                 r'[0-9]{1,2}h\.',  # time, e.g., 12h
#                 r'(\+[0-9] )?\(?[0-9]{3}\)?[\- ][0-9]{3}[\- ][0-9]{4}',  # phone number
#                 r'[0-9]{1,2}\.'  # Numbered lists
#                 ]
#     for expression in patterns:
#         for match in re.finditer(expression, doc.text):
#             start, end = match.span()
#             span = doc.char_span(start, end)
#             # This is a Span object or None if match
#             # doesn't map to valid token sequence
#             if span is not None:
#                 with doc.retokenize() as retokenizer:
#                     retokenizer.merge(span)
#     return doc
#
#
# if __name__ == '__main__':
#     nlp = spacy.load('en_core_sci_md',
#                      disable=["ner", "attribute_ruler", "lemmatizer"])
#     nlp.add_pipe('tkndef', before='parser')
#     nlp.add_pipe('custom_sentencizer', before='parser')
#
#     prova = """
#     470971328 | AECH | 09071283 | | 6159055 | 5/26/2006 12:00:00 AM | PNUEMONIA | Signed | DIS | Admission Date: 4/22/2006 Report Status: Signed
#
#     Discharge Date: 7/27/2006
#     ATTENDING: CARINE , WALTER MD
#     SERVICE:
#     Medicine Service.
#     ADMISSION INFORMATION AND CHIEF COMPLAINT:
#     Hypoxemic respiratory failure.
#     HISTORY OF PRESENT ILLNESS:
#     The patient is a 57-year-old woman with a past medical history of
#     OSA , asthma , CAD status post CABG. On 8/19/06 , she underwent a
#     right total knee replacement at Dola Elan Hospital .  On
#     8/9/06 , she was discharged to rehabilitation. There , she
#     experienced fever , cough and dyspnea. She was started on
#     vancomycin , ceftazidime , and Flagyl for presumed pneumonia. In
#     the L ED , the patient was afebrile with a temperature of 97.6 ,
#     pulse of 88 , blood pressure 117/70 , oxygen saturation 97% on 6
#     liters nasal cannula. Her exam was notable for crackles in the
#     left base and 1+ lower extremity edema.
#     ADMISSION LABS:
#     Notable for white blood cell count of 20 , hematocrit 3of 5 ,
#     platelets of 442 , 000 , creatinine of 0.6 , and INR of 1.2. Her
#     admission EKG revealed sinus tachycardia of 119 beats per minute ,
#     normal axis , QRS 104 milliseconds , QTC 461 milliseconds , no
#     evidence of atrial enlargement or ventricular hypertrophy , poor
#     R-wave progression , 2 mm ST depressions and T-wave inversions in
#     leads 1 , aVL , V5 , V6 , 1 mm J-point elevation in V3 ( prior EKG
#     showed T-wave inversions in 1 , and aVL with no ST depressions ).
#     Her admission chest x-ray revealed bilateral diffuse patchy
#     opacities.
#     The patient was presumed to have pneumonia versus CHF. She was
#     treated with vancomycin , cefotaxime , levofloxacin , and
#     azithromycin , and was admitted to the Medicine Service for
#     further evaluation and management.
#     PAST MEDICAL HISTORY:
#     1. Left carotid artery stenosis status post CEA.
#     2. Right carotid artery stenosis , status post angioplasty.
#     3. OSA.
#     4. Asthma.
#     5. CAD status post three-vessel CABG in 2004 and subsequent PCI
#     to the ramus in 2005.
#     6. 70-80% RCA stenosis not bypassed during CABG.
#     7. Hypertension.
#     8. CHF , ejection fraction 45-50%.
#     9. AS status post aortic valve replacement.
#     10. Pericarditis removal.
#     11. Diabetes.
#     12. Peripheral vascular disease.
#     MEDICATIONS AT REHAB:
#     1. Vancomycin 1 gram IV q. 12h. , ( first dose 27 of March ).
#     2. Ceftazidime 1 g IV q. 8h. , ( first dose 7/17/06 )
#     3. Flagyl 500 mg IV q. 8h. , ( first dose 7/17/06 .
#     4. Advair 100/50 inhaled b.i.d.
#     5. Aspirin 325 mg p.o. daily.
#     6. Lipitor 80 mg p.o. at bedtime.
#     7. Zetia 10 mg p.o. daily.
#     8. Lopressor 75 mg p.o. q. 6h.
#     9. Lasix 1 tablet p.o. daily.
#     10. Colace 100 mg p.o. b.i.d.
#     11. Multivitamin 1 tab p.o. daily.
#     12. CaCO3 500 mg p.o. daily.
#     13. Cholecalciferol 400 units p.o. daily.
#     14. Ferrous sulfate 300 mg p.o. t.i.d.
#     15. Folic acid 1 mg p.o. daily.
#     16. Avapro 225 mg p.o. daily.
#     17. Lantus 100 units subq daily.
#     18. Lispro sliding scale.
#     19. Coumadin.
#     20. P.r.n. oxycodone , Tylenol , Benadryl , and Metamucil.
#         """
#
#     with open('/Users/landii03/PycharmProjects/myproject/redundancy_transformers/data/abbreviations_list.csv',
#               'r') as f:
#         rd = csv.reader(f)
#         next(rd)
#         myabbrv_dict = {r[0].strip(' '): r[1].strip(' ') for r in rd}
#
#     doc = nlp(re.sub('  +', ' ', prova.replace('\n', ' ')))
#
#     for i, s in enumerate(doc.sents):
#         print(s)
#         for t in s:
#             if t.text.lower() in myabbrv_dict:
#                 print(t)
#                 print(f"{myabbrv_dict[t.text.lower()]}")
#             else:
#                 print(t)
#         print('\n')
