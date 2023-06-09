from enum import Enum


class Diagnosis(str, Enum):
    ### Cytology ###
    AGUS = "AGUS"
    LSIL = "LSIL"
    HSIL = "HSIL"
    ASC_H = "ASC-H"
    NORMAL = "Normal"
    NORMAL_w_blood = "Normal m betennelse eller blod"
    NORMAL_wo_cylinder = "Normal uten sylinder"
    ADC = "ADC"
    SCC = "SCC"
    ACIS = "ACIS"
    ASC_US = "ASC-US"
    Nonconclusive = "Uegnet"  # HPV has a diagnosis called 'uegned' (lowercase)
    #
    METASTASIS = "Metastase"
    CANCER = "Cancer Cervix cancer andre/usp"
    ### Histology ###
    Hist10 = "10"
    Hist100 = "100"
    Hist1000 = "1000"
    Hist8001 = "8001"
    Hist74006 = "74006"
    Hist74007 = "74007"
    Hist74009 = "74009"
    Hist80021 = "80021"
    Hist80032 = "80032"
    Hist80402 = "80402"
    Hist80703 = "80703"
    Hist80833 = "80833"
    Hist81403 = "81403"
    Hist82103 = "82103"
    Hist10700 = "10700"
    Hist21000 = "21000"
    Hist79350 = "79350"
    Hist86666 = "86666"
    Hist9010 = "9010"
    Hist9500 = "9500"
    Hist99 = "99"
    # HPV common
    HPVNegative = "negativ"
    HPVPositive = "positiv"
    HPVUnknown = "uegnet"
    # HPV cobas
    HPVCobas16 = "HPV 16"
    HPVCobas18 = "HPV 18"
    HPVCobasChannel12 = "Cobas Channel 1"
    """Channel collecting 31, 33, 35, 39, 45, 51, 52, 56, 58, 59, 66, and 68"""
    # HPV genXpert
    HPVGenXpert16 = "HPV 16"
    HPVGenXpert18_45 = "HPV pool 18/45"
    HPVGenXpertchannel1 = "genXpert channel 1"
    """Channel collecting 31, 33, 35, 52, 58; 51, 59; 39, 56, 66, 68"""


class ExamTypes(str, Enum):
    Cytology = "cytology"
    Histology = "histology"
    HPV = "HPV"


risk_mapping = {
    Diagnosis.NORMAL: 1,
    Diagnosis.LSIL: 2,
    Diagnosis.ASC_US: 2,
    Diagnosis.ASC_H: 3,
    Diagnosis.HSIL: 3,
    Diagnosis.ACIS: 3,
    Diagnosis.AGUS: 3,
    Diagnosis.SCC: 4,
    Diagnosis.ADC: 4,
    # TODO: Check metastasis and cancer!
    Diagnosis.METASTASIS: None,
    Diagnosis.CANCER: None,
    ##
    Diagnosis.Hist10: 1,
    Diagnosis.Hist100: 1,
    Diagnosis.Hist1000: 1,
    Diagnosis.Hist8001: 1,
    Diagnosis.Hist74006: 2,
    Diagnosis.Hist74007: 3,
    Diagnosis.Hist74009: 3,
    Diagnosis.Hist80021: 1,
    Diagnosis.Hist80032: 3,
    Diagnosis.Hist80402: 3,
    Diagnosis.Hist80703: 4,
    Diagnosis.Hist80833: 4,
    Diagnosis.Hist81403: 4,
    Diagnosis.Hist82103: 4,
    #
    Diagnosis.Hist10700: None,
    Diagnosis.Hist21000: None,
    Diagnosis.Hist79350: None,
    Diagnosis.Hist86666: None,
    Diagnosis.Hist9010: None,
    Diagnosis.Hist9500: None,
    Diagnosis.Hist99: None,
    ##
    # HPV is not mapped to a risk
    Diagnosis.HPVNegative: None,
    Diagnosis.HPVPositive: None,
    Diagnosis.HPVUnknown: None,
    Diagnosis.HPVCobas16: None,
    Diagnosis.HPVCobas18: None,
    Diagnosis.HPVCobasChannel12: None,
    Diagnosis.HPVGenXpert16: None,
    Diagnosis.HPVGenXpert18_45: None,
    Diagnosis.HPVGenXpertchannel1: None,
    # TODO: verify
    Diagnosis.Nonconclusive: None,
    Diagnosis.NORMAL_w_blood: 1,
    Diagnosis.NORMAL_wo_cylinder: 1,
}

assert set(risk_mapping.keys()) == set(Diagnosis), set(risk_mapping.keys()) ^ set(
    Diagnosis
)

HPV_TEST_TYPE_NAMES = {
    1: "HCII",
    2: "HCIII",
    3: "PreTect HPV-Proofer",
    4: "Amplicor",
    5: "PCR-primer",
    6: "Real time PCR",
    7: "Ventana Inform HPV (ISH)",
    8: "ISH andre",
    9: "PAP 13 Tele-lab",
    10: "Paptype13 realtime",
    11: "Cobas 4800 System",
    12: "Abbott RealTime High Risk HPV",
    13: "BD Onclarity HPV Assay",
    14: "Inno Lipa",
    15: "(Ukjent)",
    16: "Abbot Alinity",
    17: "Cobas 6800",
}
"""Mapping from HPV test type code to long form name."""
