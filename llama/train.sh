#!/bin/zsh
python -m llama_recipes.finetuning \
  	--use_peft \
	--peft_method lora \
	--quantization  \
	--dataset alpaca_dataset \
	--model_name meta-llama/Llama-2-7b-chat-hf \
	--save_model \
	--batch_size_training 12 \
	--num_epochs 10 \
	--val_batch_size 12 \
	--output_dir ./fine-tuned-llama \
	--log_pathname ./log.json



