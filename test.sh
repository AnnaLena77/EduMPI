#!/bin/bash

# SSH-Verbindung zum Cluster aufbauen und Commands ausführen
ssh -T fdai0231@10.32.47.10 << EOF
    # Kommando auf dem Cluster ausführen, z.B. sbatch
    source ~/.bashrc        # Umgebungsvariablen laden
    cd eduMPI_files
    sbatch remote_bash_eduMPI.sh
EOF
