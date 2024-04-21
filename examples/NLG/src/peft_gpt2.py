from transformers import GPT2Tokenizer, GPT2Model, AutoModelForSeq2SeqLM
#from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
# )
model_name_or_path = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2Model.from_pretrained(model_name_or_path)
#model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)



#model = get_peft_model(model, peft_config)

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)