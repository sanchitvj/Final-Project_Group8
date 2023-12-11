# Final-Project_Group8

Install dependencies using  
`pip install -r requirements.txt`  

- Go to `Code` directory. There you will find all the code.  
- To start pre-training first configure hyperparameters in `config.yaml` then run `python3 main.py` in terminal.  
- To start with fine-tuning configure `ft_config.yaml` then run `python3 main_ft.py` in terminal.
- If you wanna use `WandB` logger, make sure you paste key file from your weights and biases account.
- All the saved models are found in `saved_checkponts` folder.
- All checkpoints used for evaluation in app are in 'ckp_app' folder.
- Make sure all your data is in `Code` directory. 

### App
To test the app make sure you are in local environment as it will not work on instance because you are not authorized to modify security group of instance.

Go to terminal, got to `Code` folder and execute below command:
`streamlit run app.py`
