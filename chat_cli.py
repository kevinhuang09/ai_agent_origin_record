from maiagent_client import MaiAgentClient

NEUROP_ID = "ea6bff29-4f47-474e-9b63-d44d1af06b42"
client = MaiAgentClient(chatbot_id=NEUROP_ID)

print("🧬 NeuroP 神經肽預測助理")
print("輸入 'exit' 離開，'reset' 開新對話\n")

while True:
    user_input = input("👤 你：").strip()
    
    if not user_input:
        continue
    if user_input.lower() == "exit":
        print("👋 再見！")
        break
    if user_input.lower() == "reset":
        client.reset()
        print("🔄 對話已重置\n")
        continue
    
    try:
        reply = client.chat(user_input)
        print(f"\n🤖 NeuroP：{reply}\n")
    except Exception as e:
        print(f"❌ 錯誤：{e}\n")
