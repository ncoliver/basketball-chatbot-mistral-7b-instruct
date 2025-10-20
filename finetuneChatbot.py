# -*- coding: utf-8 -*-
"""finetuneChatbot.ipynb
Fixed and simplified version
"""


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
import gradio as gr


# ✅ Fine-tuning function
def fine_tune_chatbot(model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    model_name = "tiiuae/falcon-1b-instruct"  # smaller model recommended

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    dataset = load_dataset("json", data_files="basketball.json")

    def tokenize(example):
        text = example["question"] + "\n" + example["answer"] + tokenizer.eos_token
        tokenized = tokenizer(text, truncation=True, max_length=512, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_ds = dataset.map(tokenize)

    lora_config = LoraConfig(
        r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./chatbot_model",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,  # ✅ keep small first
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit"
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_ds["train"])
    trainer.train()

    # Save LoRA adapter
    model.save_pretrained("./chatbot_model")
    print("✅ LoRA adapter saved to ./chatbot_model")


# ✅ Gradio app function
def create_gradio_app(base_model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, device_map="auto", torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(base_model, "./chatbot_model")

    pipe = gr.ChatInterface(
        fn=lambda msg, history: pipe_generate(msg, history, model, tokenizer),
        title="🏀 Basketball Chatbot",
        description="Ask about basketball concepts fine-tuned on a custom dataset."
    )
    pipe.launch()


# ✅ Helper function for chat generation
def pipe_generate(prompt, chat_history, model, tokenizer):
    from transformers import pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    history_text = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in chat_history])
    full_prompt = f"{history_text}\nUser: {prompt}\nAssistant:"

    output = pipe(full_prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"]
    response = output.split("Assistant:")[-1].strip()
    chat_history.append((prompt, response))
    return response


# ✅ Run the code
if __name__ == "__main__":
    fine_tune_chatbot()
    create_gradio_app()
