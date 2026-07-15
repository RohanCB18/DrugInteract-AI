export const TEST_SET_PAIRS = [
  {
    a: "Warfarin",
    smilesA: "CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=C(OC1=O)C=CC=C2",
    b: "Acetaminophen",
    smilesB: "CC(=O)NC1=CC=C(O)C=C1",
    label: "Class 0 (Increases Effect)"
  },
  {
    a: "Aspirin",
    smilesA: "CC(=O)OC1=CC=CC=C1C(O)=O",
    b: "Fostamatinib",
    smilesB: "CNCC1=CC=C(C=C1)C1=C2CCNC(=O)C3=C2C(N1)=CC(F)=C3",
    label: "Class 1 (Decreases Effect)"
  },
  {
    a: "Loratadine",
    smilesA: "CCOC(=O)N1CCC(CC1)=C1C2=C(CCC3=C1N=CC=C3)C=C(Cl)C=C2",
    b: "Ritonavir",
    smilesB: "CC(C)[C@H](NC(=O)N(C)CC1=CSC(=N1)C(C)C)C(=O)N[C@H](C[C@H](O)[C@H](CC1=CC=CC=C1)NC(=O)OCC1=CN=CS1)CC1=CC=CC=C1",
    label: "Class 1 (Decreases Effect)"
  },
  {
    a: "Aripiprazole",
    smilesA: "COC1=CC=C(C=C1)C(CN(C)C)C1(O)CCCCC1",
    b: "Atorvastatin",
    smilesB: "CC(C)C1=C(C(=O)NC2=CC=CC=C2)C(=C(N1CC[C@@H](O)C[C@@H](O)CC(O)=O)C1=CC=C(F)C=C1)C1=CC=CC=C1",
    label: "Class 3 (Increases Risk)"
  },
  {
    a: "Omeprazole",
    smilesA: "COC1=CC2=C(C=C1)N=C(N2)S(=O)CC1=NC=C(C)C(OC)=C1C",
    b: "Caffeine",
    smilesB: "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    label: "Class 1 (Decreases Effect)"
  },
  {
    a: "Propranolol",
    smilesA: "CC(C)NCC(O)COC1=CC=CC2=C1C=CC=C2",
    b: "Risperidone",
    smilesB: "CC1=C(CCN2CCC(CC2)C2=NOC3=C2C=CC(F)=C3)C(=O)N2CCCCC2=N1",
    label: "Class 4 (Other Interaction)"
  },
  {
    a: "Levothyroxine",
    smilesA: "N[C@@H](CC1=CC(I)=C(OC2=CC(I)=C(O)C(I)=C2)C(I)=C1)C(O)=O",
    b: "Theophylline",
    smilesB: "CN1C2=C(NC=N2)C(=O)N(C)C1=O",
    label: "Class 3 (Increases Risk)"
  },
  {
    a: "Phenytoin",
    smilesA: "O=C1NC(=O)C(N1)(C1=CC=CC=C1)C1=CC=CC=C1",
    b: "Ibuprofen",
    smilesB: "CC(C)CC1=CC=C(C=C1)C(C)C(O)=O",
    label: "Class 3 (Increases Risk)"
  },
  {
    a: "Warfarin",
    smilesA: "CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=C(OC1=O)C=CC=C2",
    b: "Cholecalciferol",
    smilesB: "CC(C)CCC[C@@H](C)[C@@]1([H])CC[C@@]2([H])\\C(CCC[C@]12C)=C\\C=C1\\C[C@@H](O)CCC1=C",
    label: "Class 1 (Decreases Effect)"
  },
  {
    a: "Caffeine",
    smilesA: "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    b: "Fluoxetine",
    smilesB: "CNCCC(OC1=CC=C(C=C1)C(F)(F)F)C1=CC=CC=C1",
    label: "Class 1 (Decreases Effect)"
  },
  {
    a: "Salicylic Acid",
    smilesA: "OC(=O)C1=CC=CC=C1O",
    b: "Ibuprofen",
    smilesB: "CC(C)CC1=CC=C(C=C1)C(C)C(O)=O",
    label: "Class 3 (Increases Risk)"
  },
  {
    a: "Risperidone",
    smilesA: "CC1=C(CCN2CCC(CC2)C2=NOC3=C2C=CC(F)=C3)C(=O)N2CCCCC2=N1",
    b: "Lovastatin",
    smilesB: "[H][C@]12[C@H](C[C@@H](C)C=C1C=C[C@H](C)[C@@H]2CC[C@@H]1C[C@@H](O)CC(=O)O1)OC(=O)[C@@H](C)CC",
    label: "Class 1 (Decreases Effect)"
  },
  {
    a: "Indomethacin",
    smilesA: "COC1=CC2=C(C=C1)N(C(=O)C1=CC=C(Cl)C=C1)C(C)=C2CC(O)=O",
    b: "Dexamethasone",
    smilesB: "[H][C@@]12C[C@@H](C)[C@](O)(C(=O)CO)[C@@]1(C)C[C@H](O)[C@@]1(F)[C@@]2([H])CCC2=CC(=O)C=C[C@]12C",
    label: "Class 3 (Increases Risk)"
  },
  {
    a: "Risperidone",
    smilesA: "CC1=C(CCN2CCC(CC2)C2=NOC3=C2C=CC(F)=C3)C(=O)N2CCCCC2=N1",
    b: "Loratadine",
    smilesB: "CCOC(=O)N1CCC(CC1)=C1C2=C(CCC3=C1N=CC=C3)C=C(Cl)C=C2",
    label: "Class 3 (Increases Risk)"
  },
  {
    a: "Fluoxetine",
    smilesA: "CNCCC(OC1=CC=C(C=C1)C(F)(F)F)C1=CC=CC=C1",
    b: "Levothyroxine",
    smilesB: "N[C@@H](CC1=CC(I)=C(OC2=CC(I)=C(O)C(I)=C2)C(I)=C1)C(O)=O",
    label: "Class 4 (Other Interaction)"
  },
  {
    a: "Diazepam",
    smilesA: "[H][C@@]12CC[C@H](O)[C@@]1(C)CC[C@@]1([H])[C@@]2([H])CCC2=CC(=O)CC[C@]12C",
    b: "Aripiprazole",
    smilesB: "COC1=CC=C(C=C1)C(CN(C)C)C1(O)CCCCC1",
    label: "Class 1 (Decreases Effect)"
  },
  {
    a: "Phenytoin",
    smilesA: "O=C1NC(=O)C(N1)(C1=CC=CC=C1)C1=CC=CC=C1",
    b: "Dexamethasone",
    smilesB: "[H][C@@]12C[C@@H](C)[C@](O)(C(=O)CO)[C@@]1(C)C[C@H](O)[C@@]1(F)[C@@]2([H])CCC2=CC(=O)C=C[C@]12C",
    label: "Class 4 (Other Interaction)"
  },
  {
    a: "Metformin",
    smilesA: "CCOC1=C(C=CC(CC(=O)N[C@@H](CC(C)C)C2=CC=CC=C2N2CCCCC2)=C1)C(O)=O",
    b: "Salicylic Acid",
    smilesB: "OC(=O)C1=CC=CC=C1O",
    label: "Class 3 (Increases Risk)"
  },
  {
    a: "Propranolol",
    smilesA: "CC(C)NCC(O)COC1=CC=CC2=C1C=CC=C2",
    b: "Ritonavir",
    smilesB: "CC(C)[C@H](NC(=O)N(C)CC1=CSC(=N1)C(C)C)C(=O)N[C@H](C[C@H](O)[C@H](CC1=CC=CC=C1)NC(=O)OCC1=CN=CS1)CC1=CC=CC=C1",
    label: "Class 1 (Decreases Effect)"
  },
  {
    a: "Diazepam",
    smilesA: "[H][C@@]12CC[C@H](O)[C@@]1(C)CC[C@@]1([H])[C@@]2([H])CCC2=CC(=O)CC[C@]12C",
    b: "Fostamatinib",
    smilesB: "CNCC1=CC=C(C=C1)C1=C2CCNC(=O)C3=C2C(N1)=CC(F)=C3",
    label: "Class 1 (Decreases Effect)"
  },
  {
    a: "Fluoxetine",
    smilesA: "CNCCC(OC1=CC=C(C=C1)C(F)(F)F)C1=CC=CC=C1",
    b: "Omeprazole",
    smilesB: "COC1=CC2=C(C=C1)N=C(N2)S(=O)CC1=NC=C(C)C(OC)=C1C",
    label: "Class 1 (Decreases Effect)"
  },
  {
    a: "Phenytoin",
    smilesA: "O=C1NC(=O)C(N1)(C1=CC=CC=C1)C1=CC=CC=C1",
    b: "Indomethacin",
    smilesB: "COC1=CC2=C(C=C1)N(C(=O)C1=CC=C(Cl)C=C1)C(C)=C2CC(O)=O",
    label: "Class 3 (Increases Risk)"
  },
  {
    a: "Acetaminophen",
    smilesA: "CC(=O)NC1=CC=C(O)C=C1",
    b: "Caffeine",
    smilesB: "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    label: "Class 1 (Decreases Effect)"
  },
  {
    a: "Fluoxetine",
    smilesA: "CNCCC(OC1=CC=C(C=C1)C(F)(F)F)C1=CC=CC=C1",
    b: "Milnacipran",
    smilesB: "CCN(CC)C(=O)C1(CC1CN)C1=CC=CC=C1",
    label: "Class 3 (Increases Risk)"
  },
  {
    a: "Zolpidem",
    smilesA: "FC(F)OC(Cl)C(F)(F)F",
    b: "Risperidone",
    smilesB: "CC1=C(CCN2CCC(CC2)C2=NOC3=C2C=CC(F)=C3)C(=O)N2CCCCC2=N1",
    label: "Class 4 (Other Interaction)"
  }
];
