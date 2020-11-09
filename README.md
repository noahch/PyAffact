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
- Random Bounding Box? When is it used?
- Server Status
- Welches BBX
- Scale: Code!=Paper (Im Paper Scale = W/w, im Code Scale = W/tr)
- bbx viel grösser als bbx vom dataset
- talk about evaluation and server
- default parameters aus paper und code nicht ersichtlich
- Erwartung Schlussreport?
- Zahl am Anfag der Datasets entfernt
- Reshape or CenterCrop