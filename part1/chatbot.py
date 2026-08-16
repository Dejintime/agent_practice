from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = r"F:\agent\Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

class MultiTurnCharbot:
    def __init__(self):
        self.history = []
    def generate_response(self, user_input):
        prompt = "对话历史:\n"
        for turn in self.history:
            prompt += f"用户：{turn['user']}\n助手：{turn['assistant']}\n"
        prompt += f"用户：{user_input}\n助手："

        inputs = tokenizer(prompt, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            max_length=500,
            temperature=0.7,
            top_k=50,
            top_p = 0.9,
            num_return_sequences=1,
            eos_token_id = tokenizer.eos_token_id
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("助手：")[-1]
        self.history.append({"user":user_input, "assistant":response})
        return response
    
chatbot = MultiTurnCharbot()
print("欢迎使用聊天机器人！输入 '退出' 停止对话。")

while True:
    user_input = input("\n用户：")
    if user_input.lower() == "退出":
        print("聊天结束，再见！")
        break
    response = chatbot.generate_response(user_input)
    print(f"助手：{response}")
