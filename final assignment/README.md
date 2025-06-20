#  README – EPA141A MORDM Flood Risk Project

##  Team Information

| Name            | Student Number |
|-----------------|----------------|
| Jelle de Boer   | 5403170        |
| Tessa Huijsinga | 5173914        |
| Thijn Keunen    | 5091349        |
| Jet Beelen      | 5409942        |
---

## 📂 Project Files Overview

Below is a list of files that we either created ourselves or modified significantly as part of the project.

| File Name                | Description                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------|
| `dike_model_function.py` | Core model implementation for the dike network and Room for the River. In  this file we added a piece of code to |
|                          | make the new variable HRI it is located from line 341.                                                           |
| `Problem Formulation.py` | We changed the outcome of problem formulation (3) so it gave us the right information.                           |
| `Visual analysis.py`     | The correlation matrix is made here. Also the Extra Tree analysis is done in this file.                          |
| `SOBOL.py`               | File with the SOBOL analysis.                                                                                    |
| `MORDM.ipynb`            | This is the biggest file made. In this notebook the two MORDM's are executed together with the scenario analysis |
|                          | and the analysis to test the robustness. All the visualization of those analysis are also produced here          | 
| `Files for ourself`      | This folder contains exploratory scripts and analysis attempts used during our development process.              |
|                          | While not essential for running or understanding the final model, we chose to keep these files for transparency, |
|                          | and future reference.                                                                                            |
                                                                                                 

---

## ▶ How to Run the Model

### Requirements
Before running the model run the next line in your terminal to make sure all the requirements are installed 

pip install -r requirements.txt

### Things to know about the model
The original files provided with the project are interconnected, and the minor changes we made do not affect their compatibility or functionality.
The newly created files are modular in design: they extract or build upon information from the original files but do not interact directly with one another.
As such, there is no specific order in which these new files need to be executed—they can be run independently based on the analysis step you wish to perform.

