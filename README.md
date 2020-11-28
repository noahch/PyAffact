# PyAffact
PyTorch implementation of the Affact Paper

# CMDs
- conda install -c https://www.idiap.ch/software/bob/conda bob.io.image
- conda install -c https://www.idiap.ch/software/bob/conda bob.ip.base
- conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch


# TODO
- Config Evaluation
- Evaluation
- Run Experiments
- Report
- Download unaligned
- Dataset via flag in config
- Each Transform Operation is a custom class.. Compose them by checking enabled flag
- Read Paper and try to evaluate

# Questions
- Server Status
- Erwartung Schlussreport
- Zahl am Anfag der Datasets entfernt
- Reshape or CenterCrop
- Wie oft das selbe Image modifizieren in der Preprocessing pipeline.. --> TEST!
- BBX nachfragen.. für baseline (z.B aligned, oder unaligned mit BBX)
- Beim Evaluieren: Wie transformieren

-> Dataset remove first line.. 
-> Gauss Sigma zu hoch, viel zu blurry