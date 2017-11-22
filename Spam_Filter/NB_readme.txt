Instructions to run

Pre-requisits

1) Before trying to run the script, please make sure all the python package requirements are met using requirements.txt
2) Run the command  "pip install -r requirements.txt" to satisfy all the requirements.

Input

1) The script has a default configuration to run when it is present in the same working directory as the training, testing datasets and the SPAM.labels file.
2) If the script is not in the same working directory, it will ask the user to input the paths to all the required dataset folders and label file.
3) It will accept both absolute and relative paths.

Output

1) The script outputs various metrics on the console output.
2) It also creates a file called NBresults.txt which has all the predictions done during testing phase, while mentioning if they were correct or not.

Extra

1) After the first run, the script saves the training model as a piclkle file so that subsequent runs are quicker.
2) If there is a need to override this behaviour, just delete the pickle file.

