#!/bin/bash

# ###### Zona de Parámetros de solicitud de recursos a SLURM ############################
#
#SBATCH --job-name=h4_ActGlob		    #Nombre del job
#SBATCH -p medium					    #Cola a usar, Default=short (Ver colas y límites en /hpcfs/shared/README/partitions.txt)
#SBATCH -N 1						    #Nodos requeridos, Default=1
#SBATCH -n 1						    #Tasks paralelos, recomendado para MPI, Default=1
#SBATCH --cpus-per-task=1			    #Cores requeridos por task, recomendado para multi-thread, Default=1
#SBATCH --mem=16000					    #Memoria en Mb por CPU, Default=2048
#SBATCH --time=165:00:00				#Tiempo máximo de corrida, Default=2 horas
#SBATCH --mail-user=a.barrera5@uniandes.edu.co
#SBATCH --mail-type=ALL			
#SBATCH -o Output_ActGlob0.o%j			#Nombre de archivo de salida
#
########################################################################################

# ################## Zona Carga de Módulos ############################################
module load anaconda/python3.9

########################################################################################


# ###### Zona de Ejecución de código y comandos a ejecutar secuencialmente #############
sleep 60
python ActivadoresGlobh4.py
host=`/bin/hostname`
date=`/bin/date`
echo "Acabamos :)"
echo "Corri en la maquina: "$host
echo "Corri el: "$date

########################################################################################

